import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Actor(nn.Module):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784 + self.num_actions, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.1)

    def forward(self, traj, ac):
        x = traj / 255.0
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        feat = x.view(-1, 784)
        x = torch.cat((feat,ac), dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        return mu, std, log_std


class Critic(nn.Module):
    def __init__(self, num_actions):
        self.num_actions = num_actions
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784 + self.num_actions, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc2.weight.data.mul_(0.1)
        self.fc2.bias.data.mul_(0.1)

    def forward(self, x, ac):
        x = x / 255.0
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        feat = x.view(-1, 784)
        x = torch.cat((feat, ac), dim=1)
        x = F.tanh(self.fc1(x))
        v = self.fc2(x)

        return v


def get_action(mu, std):
    action = torch.normal(mu, std)
    # action = action.cpu().detach().numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x-mu).pow(2) / (2 * var) \
                    - 0.5 * math.log(2*math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def get_loss(actor, returns, states, ac, actions):  # state-ac pair, actions is the predicted label#
    mu, std, logstd = actor(states, ac)
    log_policy = log_density(actions, mu, std, logstd)
    object = returns * log_policy
    object = object.mean()
    return - object


def train_actor(actor, returns, states, ac, actions, actor_optim):
    loss = get_loss(actor, returns, states, ac, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()

    return loss
