# encoding=utf8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden1=512, hidden2=256):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        self.fc1 = nn.Linear(self.goal_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + self.state_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, self.action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, xin):
        s,g = xin
        out = self.fc1(g)
        out = self.relu(out)
        c_in = torch.cat([out,s],len(s.shape)-1)
        out = self.fc2(c_in)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden1=512, hidden2=256):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self.fc1_1 = nn.Linear(self.state_dim, hidden1)
        self.fc1_2 = nn.Linear(self.goal_dim, hidden1)
        self.fc2 = nn.Linear(hidden1*2 + self.action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xin):
        s,a,g = xin
        out_1 = self.fc1_1(s)
        out_2 = self.fc1_2(g)
        out_1 = self.relu(out_1)
        out_2 = self.relu(out_2)

        c_in = torch.cat([out_1, out_2 ,a],len(a.shape)-1)
        out = self.fc2(c_in)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out