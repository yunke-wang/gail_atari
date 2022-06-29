import copy
import glob
import os
import time
from collections import deque
from torch.utils.data import Dataset,DataLoader,TensorDataset
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from utils.disc import Disc_atari
from tqdm import tqdm
from pathlib import Path
from utils.writer import Writer
from tensorboardX import SummaryWriter


def convert_to_onehot(action, num_actions):
    onehot_ac = torch.zeros(action.shape[0], num_actions)
    for i in range(onehot_ac.shape[0]):
       onehot_ac[i, int(action[i])] = 1
    return onehot_ac


def convert_to_num(action, num_actions):
    ac = torch.zeros(action.shape[0], 1)
    for i in range(ac.shape[0]):
        k = action[i].argmax()
        ac[i] = k
    return ac


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def behavior_cloning(actor_critic, state, action):
    bc_optim = optim.Adam(actor_critic.parameters(), lr=1e-4)
    bc_loss = nn.CrossEntropyLoss()
    # bc_dataset = TensorDataset(state, action)
    # train_loader = DataLoader(dataset=bc_dataset, batch_size=128, shuffle=True)
    batch = 128
    iter_max = 10000
    for i in range(iter_max):
    # for j, ba_data in enumerate(train_loader):
        idx = np.random.choice(state.shape[0], batch)
        ba_state = state[idx]
        ba_ac = action[idx]

        # ba_state, ba_ac = ba_data
        # ba_state = Variable(ba_state)
        # ba_ac = Variable(ba_ac)

        ac, _ = actor_critic.bc(ba_state)

        bc_optim.zero_grad()
        loss = bc_loss(ac, torch.max(ba_ac,1)[1])
        loss.backward()
        bc_optim.step()

        if i % 1000 == 0:
            print('BC pre train:{}, loss: {:.2f}'.format(i, loss.cpu().detach().numpy()))
        if i >= iter_max:
            break


def main():
    args = get_args()
    set_seed(args)

    method = 'gail'
    logdir = Path(os.path.abspath(os.path.join('logs/gail_optimal', str(args.env_name), method+'_'+str(args.traj_size))))
    log_dir = os.path.expanduser(logdir)
    eval_log_dir = log_dir + '/{}_{}_{}_eval'.format(args.env_name, method, args.traj_size)
    utils.cleanup_log_dir(eval_log_dir)
    logdir_checkpoint = Path(Path(os.path.abspath(os.path.join(logdir, 'checkpoints'))))
    if logdir.exists():
        print('orinal logdir is already exist.')
    else:
        logdir_checkpoint.mkdir(parents=True)

    writer_tensor = SummaryWriter(log_dir)
    writer = Writer(args.env_name, args.seed, args.prior, args.traj_size, fname=method, folder=str(logdir))
    writer1 = Writer(args.env_name, args.seed, args.prior, args.traj_size, fname=method + str('rollout'),
                     folder=str(logdir))

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, str(logdir), device, False)
    num_actions = envs.action_space.n

    # load data
    name = args.name
    expert_state1 = np.load('demonstrations/new_ppo_{}_state_level1.npy'.format(name))
    expert_action1 = np.load('demonstrations/new_ppo_{}_ac_level1.npy'.format(name))
    expert_state2 = np.load('demonstrations/new_ppo_{}_state_level2.npy'.format(name))
    expert_action2 = np.load('demonstrations/new_ppo_{}_ac_level2.npy'.format(name))

    expert_state1 = torch.FloatTensor(expert_state1)
    expert_action1 = torch.FloatTensor(expert_action1)

    expert_state2 = torch.FloatTensor(expert_state2)
    expert_action2 = torch.FloatTensor(expert_action2)

    ac1 = convert_to_onehot(expert_action1, num_actions)  # one-hot action
    ac2 = convert_to_onehot(expert_action2, num_actions)

    if args.imperfect:
        expert_state = torch.cat((expert_state1, expert_state2),dim=0).to(device)
        ac = torch.cat((ac1,ac2),dim=0).to(device)
        expert_action = torch.cat((expert_action1,expert_action2),dim=0).to(device)
        print('imperfect demonstrations')
    else:
        expert_state = expert_state2.to(device)
        ac = ac2.to(device)
        expert_action = expert_action2.to(device)
        print('optimal demonstrations')

    actor_critic = Policy(envs.observation_space.shape,envs.action_space,base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    # bc_pretrain
    if args.bc:
        behavior_cloning(actor_critic, expert_state, ac)
        eva_reward_bc, eva_var_bc = evaluate(actor_critic, None, args.env_name, args.seed,
                                       args.num_processes, eval_log_dir, device)
        writer_tensor.add_scalar('BC/eva_reward_var', eva_var_bc)
        writer_tensor.add_scalar('BC/eva_reward', eva_reward_bc)

    # Discriminator
    assert args.gail == True
    disc = Disc_atari(num_actions).to(device)
    disc_criterion = nn.BCEWithLogitsLoss()
    disc_optimizer = optim.Adam(disc.parameters(), lr=args.disc_lr)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in tqdm(range(num_updates), dynamic_ncols=True):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            # if j < 10:
            #     gail_epoch = 100  # Warm up
            gail_epoch = 1
            for _ in range(gail_epoch):
                policy_data_generator = rollouts.feed_forward_generator(
                    None, mini_batch_size=1024)

                for policy_batch in policy_data_generator:
                    policy_state, policy_action = policy_batch[0], policy_batch[2]
                    policy_action = convert_to_onehot(policy_action, num_actions).to(device)
                    # random choose or traj choose

                    idx = np.random.choice(expert_action.shape[0], 1024, replace=False)

                    expert_state_batch = expert_state[idx]
                    expert_action_batch = ac[idx]

                    policy_d = disc(policy_state, policy_action)
                    expert_d = disc(expert_state_batch, expert_action_batch, flag=0)

                    disc_optimizer.zero_grad()
                    gail_loss = disc_criterion(policy_d, torch.ones(policy_d.size()).to(device)) \
                                    + disc_criterion(expert_d, torch.zeros(expert_d.size()).to(device))
                    gail_loss.backward()
                    disc_optimizer.step()

                generator_acc = torch.mean((torch.sigmoid(policy_d) > 0.5).float())
                disc_acc = torch.mean((torch.sigmoid(expert_d) < 0.5).float())

            for step in range(args.num_steps):
                action_onehot = convert_to_onehot(rollouts.actions[step], num_actions).to(device)
                rollouts.rewards[step] = disc.predict_reward(
                    rollouts.obs[step], action_onehot, args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # tensorboard log
        writer_tensor.add_scalar('Train/gail_loss', gail_loss.detach().cpu().numpy(), j)
        writer_tensor.add_scalar('Train/value_loss', value_loss, j)
        writer_tensor.add_scalar('Train/action_loss', action_loss, j)
        writer_tensor.add_scalar('Train/true_acc', disc_acc, j)
        writer_tensor.add_scalar('Train/fake_acc', generator_acc, j)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = logdir_checkpoint
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_{}.pt".format(j)))
            torch.save(disc.state_dict(),
                       os.path.join(save_path, args.env_name + "_disc_{}.pth".format(j)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            avg_re = np.mean(episode_rewards)
            avg_std = np.std(episode_rewards)
            tqdm.write(
                "Updates {}, num timesteps {}, FPS {}, {} episodes: mean/median {:.1f}/{:.1f}, min/max {:.1f}/{:.1f}"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), avg_re,
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            writer_tensor.add_scalar('Train/episodes_reward_timestep', avg_re, total_num_steps)
            writer_tensor.add_scalar('Train/episodes_reward_std_timestep', avg_std, total_num_steps)
            writer_tensor.add_scalar('Train/episodes_reward', avg_re, j)
            writer_tensor.add_scalar('Train/episodes_reward_std', avg_std, j)
            writer1.log(total_num_steps, avg_re, avg_std)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and (j % args.eval_interval == 0 or j == num_updates-1)):
            # ob_rms = utils.get_vec_normalize(envs).ob_rms
            eva_reward, eva_var = evaluate(actor_critic, None, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
            writer.log(j, eva_reward, eva_var)
            writer_tensor.add_scalar('Evaluate/eva_reward', eva_reward, j)

    writer_tensor.close()
if __name__ == "__main__":
    main()
