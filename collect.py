import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from evaluation import evaluate
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()


args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cuda:0',
    allow_early_resets=True)

name = args.env_name.split('No')[0]
iter= '9764'
# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, 'ppo' , args.env_name + "_{}.pt").format(iter), map_location='cuda:0')

# evaluate(actor_critic, ob_rms, args.env_name, 1, 1, '/tmp/gym', device='cuda:0')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()

render_func = get_render_func(env)

trajs = []
actions = []
eval_episode_rewards = []
k = 0
reward = 0
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    if k == 0:
        trajs = obs
        actions = action
    else:
        trajs = torch.cat((trajs,obs),dim=0)
        actions = torch.cat((actions, action),dim=0)
    k += 1

    obs, r, done, infos = env.step(action)

    # use random action
    # rand_ac = torch.IntTensor(np.array(env.action_space.sample()).reshape(1,1)).to('cuda:0')
    # obs, r, done, infos = env.step(rand_ac)

    reward += r
    masks.fill_(0.0 if done else 1.0)

    for info in infos:
        if 'episode' in info.keys():
            eval_episode_rewards.append(info['episode']['r'])

    # env.render()
    if render_func is not None:
        render_func('human')
    # print(1)
    if trajs.shape[0] == 5000:
        break

print(k, reward.cpu().detach().numpy()[0], np.mean(eval_episode_rewards))
trajs = trajs.cpu().detach().numpy()
actions = actions.cpu().detach().numpy()

np.save('demonstrations/ppo_{}_state_{}.npy'.format(name, name, iter), trajs)
np.save('demonstrations/ppo_{}_ac_{}.npy'.format(name, name, iter), actions)
