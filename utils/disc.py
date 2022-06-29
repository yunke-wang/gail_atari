import numpy as np
import torch
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd


def load_data():
    # load data and confidence
    expert_state = np.load('../space_state.npy')
    expert_action = np.load('../space_action.npy')
    expert_conf = np.load('../space_conf.npy')

    # expert_action = expert_action[:, np.newaxis]

    traj_size = 2000
    # choose traj = 2000 #
    idx = np.random.randint(0, expert_state.shape[0], traj_size)
    expert_state = expert_state[idx]
    expert_action = expert_action[idx]
    expert_conf = expert_conf[idx]
    Z = expert_conf.mean()
    return 0


class Disc_atari(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784 + num_actions, 64)
        self.fc2 = nn.Linear(64, 1)

        # self.fc2.weight.data.mul_(0.1)
        # self.fc2.bias.data.mul_(0.0)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self,traj, ac, flag=0):
        #x = traj.permute(0,3,1,2) #get into NCHW format
        if flag == 1:   # expert data
            traj = traj.permute(0,3,1,2)
        # x = traj.permute(2,0,1)
        x = traj / 255.0
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        feat = x.view(-1, 784)
        x = self.fc1(torch.cat((feat, ac), dim=1))
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.forward(state, action)
            # s = torch.sigmoid(d)
            reward = -F.logsigmoid(d)
            # reward = (s + 1e-8).log() - (1 - s + 1e-8).log()

            if np.isnan(reward.cpu().detach().numpy()).any() == True:
                print('inside reward nan')
            # reward = - s.log()
            # if self.returns is None:
            #     self.returns = reward.clone()
            #
            # if update_rms:
            #     self.returns = self.returns * masks * gamma + reward
            #     self.ret_rms.update(self.returns.cpu().numpy())
            #
            # return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # if (a>0.5).any() > 0.5:
            #     print('amazing')
            return reward
