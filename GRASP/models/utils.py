# encoding=utf8



import numpy as np
from collections import deque, namedtuple
import warnings
import random
import numpy as np

import torch
from torch.autograd import Variable

exp = namedtuple('exp', 'state0, action, goal, reward, state1, terminate, action_space')

# Use and modified code in https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py
class AnnealedGaussianProcess():
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=10000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class Memory(object):
    def __init__(self, rm_size=10000):
        self.rm_size = rm_size
        self.observations = deque(maxlen=rm_size)

    def sample(self, batch_size):
        _samples = random.sample(self.observations, batch_size)
        state0_batch = []
        action_batch = []
        goal_batch = []
        reward_batch = []
        state1_batch = []
        terminate_batch = []
        action_space_batch = []
        for _s in _samples:
            state0_batch.append(_s.state0)
            action_batch.append(_s.action)
            goal_batch.append(_s.goal)
            reward_batch.append(_s.reward)
            state1_batch.append(_s.state1)
            terminate_batch.append(0. if _s.terminate else 1.)
            action_space_batch.append(_s.action_space)

        state0_batch = np.array(state0_batch).reshape(batch_size,-1).astype(np.float64)
        action_batch = np.array(action_batch).reshape(batch_size,-1).astype(np.float64)
        goal_batch = np.array(goal_batch).reshape(batch_size,-1).astype(np.float64)        
        reward_batch = np.array(reward_batch).reshape(batch_size,-1).astype(np.float64)
        state1_batch = np.array(state1_batch).reshape(batch_size,-1).astype(np.float64)
        terminate_batch = np.array(terminate_batch).reshape(batch_size,-1).astype(np.float64)

        return state0_batch, action_batch, goal_batch, reward_batch, state1_batch, terminate_batch, action_space_batch

    def append(self, observation):
        self.observations.append(observation)

    def reset(self):
        self.observations = deque(maxlen=self.rm_size)

    def __len__(self):
        return len(self.observations)

def soft_update(target, source, tau_update):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau_update) + param.data * tau_update
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def to_numpy(d):
    return d.cpu().data.numpy().astype(np.float64)

def to_tensor(ndarray, volatile=False, requires_grad=False, device=0):
    return Variable(torch.from_numpy(ndarray).cuda(device=device).type(torch.cuda.DoubleTensor),
                        volatile=volatile,
                        requires_grad=requires_grad)