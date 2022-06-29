import numpy as np
import torch


def get_random_traj(expert_state, expert_action, expert_conf, traj_size):
    idx = np.random.randint(0, expert_state.shape[0], traj_size)
    expert_state = expert_state[idx]
    expert_action = expert_action[idx]
    expert_conf = expert_conf[idx]

    return expert_state, expert_action, expert_conf


def get_traj(expert_state, expert_action, expert_conf, traj_size, length):
    length = 100
    num_trajs = traj_size // length

    idx = np.random.choice(expert_state.shape[0] - length, num_trajs, replace=False)
    trajs = []
    trajs_ac = []
    trajs_conf = []
    for i in range(num_trajs):
        traj_ob = expert_state[idx[i]:idx[i] + 2 * length:2]
        traj_ac = expert_action[idx[i]:idx[i] + 2 * length:2]
        traj_conf = expert_conf[idx[i]:idx[i] + 2 * length:2]

        trajs.append(traj_ob)
        trajs_ac.append(traj_ac)
        trajs_conf.append(traj_conf)

    return trajs, trajs_ac, trajs_conf