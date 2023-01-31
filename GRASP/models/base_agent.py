# encoding=utf8

# Used and modified code in https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from GRASP.models.base_models import Actor, Critic
from GRASP.models.utils import Memory, OrnsteinUhlenbeckProcess, hard_update, soft_update, to_numpy, to_tensor
from GRASP.utils.misc import similarity
from GRASP.models.utils import exp

class GRASP_Base(object):
    def __init__(self, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.state_dim =  args.state_dim
        self.action_dim = args.action_dim
        self.goal_dim = args.goal_dim
        self.device = args.rl_device
        
        _cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2
        }
        self.actor = Actor(self.state_dim, self.action_dim, self.goal_dim, **_cfg).double()
        self.actor_target = Actor(self.state_dim, self.action_dim, self.goal_dim, **_cfg).double()
        self.actor_optim = Adam(self.actor.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)

        self.critic = Critic(self.state_dim, self.action_dim, self.goal_dim, **_cfg).double()
        self.critic_target = Critic(self.state_dim, self.action_dim, self.goal_dim, **_cfg).double()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.p_lr, weight_decay=args.weight_decay)

        hard_update(self.actor_target, self.actor) # target, source
        hard_update(self.critic_target, self.critic)
        
        self.memory = Memory(rm_size=args.rmsize)   # ?
        self.memory_imitation = Memory(rm_size=args.rmsize//2)   # ?
        self.random_process = OrnsteinUhlenbeckProcess(size=self.state_dim,     # ?
                                                       theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        self.batch_size = args.bsize
        self.g_prop = args.g_prop
        self.tau_update = args.tau_update
        self.gamma = args.gamma


        self.depsilon = 1.0 / args.epsilon
        self.epsilon = 1.0

        self.warmup = args.warmup
        self.report_steps = args.report_steps
        self.save_per_epochs = args.save_per_epochs

        self.max_episode_length = args.max_episode_length

        self.is_training = True

    def update_policy(self):
        pass

    def cuda_convert(self):
        if len(self.gpu_ids) == 1:
            if self.gpu_ids[0] >= 0:
                with torch.cuda.device(self.gpu_ids[0]):
                    print('model cuda converted')
                    self.cuda()


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def data_parallel(self):
        self.actor = nn.DataParallel(self.actor, device_ids=self.gpu_ids)
        self.actor_target = nn.DataParallel(self.actor_target, device_ids=self.gpu_ids)
        self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids)
        self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids)

    def to_device(self):
        self.actor.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.actor_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))
        self.critic_target.to(torch.device('cuda:{}'.format(self.gpu_ids[0])))

    def observe(self, exp):
        if self.is_training:
            self.memory.append(exp)

    def observe_imitation(self, exp):
        if self.is_training:
            self.memory_imitation.append(exp)

    def random_action(self):
        action = np.random.uniform(0., 1., self.action_dim)
        return action

    def select_action(self, s_t, g_t, decay_epsilon=True):
        # proto action
        proto_action = to_numpy(
            self.actor([to_tensor(s_t, device=self.device), to_tensor(g_t, device=self.device)])
        )

        # Decay proto-action
        proto_action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        proto_action = np.clip(proto_action, 0., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        return proto_action

    def reset(self):
        self.random_process.reset_states()

    def load_weights(self, actor_file, critic_file):
        if dir is None: return

        ml = lambda storage, loc: storage.cuda(self.device)

        self.actor.load_state_dict(
            torch.load(actor_file, map_location=ml)
        )

        self.critic.load_state_dict(
            torch.load(critic_file, map_location=ml)
        )
        print('model weights loaded')


    def save_model(self,output):
        if self.device:
            with torch.cuda.device(self.device):
                torch.save(
                    self.actor.state_dict(),
                    '{}/actor.pt'.format(output)
                )
                torch.save(
                    self.critic.state_dict(),
                    '{}/critic.pt'.format(output)
                )
        else:
            torch.save(
                self.actor.state_dict(),
                '{}/actor.pt'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic.pt'.format(output)
            )

    def seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)




class GRASP_Agent(GRASP_Base):

    def __init__(self, args, k_ratio=0.1):
        super().__init__(args)
        self.criterion = nn.MSELoss()
        self.k_nearest_neighbors = args.knn

    def true_action(self, s_t, g_t, proto_action, action_space):

        _actions = self.search_knn_point(proto_action, action_space, self.k_nearest_neighbors)
        raw_actions = np.array([x['action'] for x in _actions])

        # Equivalent for k=1
        if not isinstance(s_t, np.ndarray):
           s_t = to_numpy(s_t)
        if not isinstance(g_t, np.ndarray):
           g_t = to_numpy(g_t)

        # can also directly use 
        s_t = s_t.reshape(raw_actions.shape[0], self.state_dim)
        g_t = g_t.reshape(raw_actions.shape[0], self.goal_dim)

        raw_actions = to_tensor(raw_actions, device=self.device)
        s_t = to_tensor(s_t, device=self.device)
        g_t = to_tensor(g_t, device=self.device)

        _eval = self.critic([s_t, raw_actions, g_t])

        max_index = np.argmax(to_numpy(_eval), axis=1)
        max_index = max_index.reshape(len(max_index),)

        raw_actions = to_numpy(raw_actions)

        if self.k_nearest_neighbors == 1:
            return raw_actions[max_index], np.array(_actions)[max_index].tolist()
        else:
            raise NotImplementedError

    def cuda_convert(self):
        with torch.cuda.device(self.device):
            print('model to cuda:%d'%(self.device))
            self.cuda()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()


    def select_action(self, s_t, g_t, action_space, decay_epsilon=True):
        proto_action = super().select_action(s_t, g_t, decay_epsilon)
        raw_action, _action = self.true_action(s_t, g_t, proto_action, action_space)
        assert isinstance(raw_action, np.ndarray)
        return raw_action[0], _action[0]

    def random_action(self, action_space):
        proto_action = super().random_action()
        _action = self.search_knn_point(proto_action, action_space, k=1)[0]
        raw_action = _action['action']
        assert isinstance(raw_action, np.ndarray)
        return raw_action, _action

    def select_target_action(self, s_ts, g_ts, action_space_batchs):
        proto_actions = self.actor_target([s_ts, g_ts])
        proto_actions = to_numpy(proto_actions)

        raw_action, _action = self.true_action(s_ts, g_ts, proto_actions, action_space_batchs)
        return raw_action, _action

    def search_knn_point(self, proto_actions, action_spaces, k=1):
        try:
            if len(proto_actions.shape) == 1:
                ind = np.argpartition([similarity(proto_actions, _a['action']) for _a in action_spaces], -k)[-k:].tolist()
                if k == 1:
                    return [action_spaces[ind[0]]]
                else:
                    raise NotImplementedError
            else:
                ind = []
                for proto_action, action_space in zip(proto_actions, action_spaces):
                    ind.append(np.argpartition([similarity(proto_action, _a['action']) for _a in action_space], -k)[-k:].tolist())
                if k == 1:
                    return [a[x[0]] for x,a in zip(ind, action_spaces)]
                else:
                    raise NotImplementedError
        except:
            print(proto_actions, action_spaces)

    def update_policy(self, imitation=False):
        # Sample batch
        state_batch, action_batch, goal_batch, reward_batch, \
        next_state_batch, terminal_batch, ns_action_space_batch = self.memory.sample(self.batch_size)

        next_state_batch = to_tensor(next_state_batch, volatile=True, device=self.device)
        # Default using same goal for consecutive actions estimation
        next_goal_batch = to_tensor(goal_batch, volatile=True, device=self.device)
        raw_action_batch, _ = self.select_target_action(next_state_batch, next_goal_batch, ns_action_space_batch)

        next_q_values = self.critic_target([
            next_state_batch,
            to_tensor(raw_action_batch, volatile=True, device=self.device), 
            next_goal_batch
        ])

        next_q_values.volatile = False

        # next_q_values = 0 if is terminal states
        target_q_batch = to_tensor(reward_batch, device=self.device) + \
                         self.gamma * \
                         to_tensor(terminal_batch.astype(np.float64), device=self.device) * \
                         next_q_values

        # optimize critic
        self.critic.zero_grad()  

        state_batch = to_tensor(state_batch, device=self.device)
        action_batch = to_tensor(action_batch, device=self.device)
        goal_batch = to_tensor(goal_batch, device=self.device)
        q_batch = self.critic([state_batch, action_batch, goal_batch])

        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()  
        self.critic_optim.step() 

        # optimize actor
        self.actor.zero_grad()

        policy_loss = -self.critic([state_batch, self.actor([state_batch, goal_batch]), goal_batch])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        if imitation:
            state_batch, action_batch, goal_batch, reward_batch, \
            _, _, _ = self.memory_imitation.sample(self.batch_size)

            # imitation target
            target_q_batch = to_tensor(reward_batch, device=self.device)

            # optimize critic
            self.critic.zero_grad()  

            state_batch = to_tensor(state_batch, device=self.device)
            action_batch = to_tensor(action_batch, device=self.device)
            goal_batch = to_tensor(goal_batch, device=self.device)
            q_batch = self.critic([state_batch, action_batch, goal_batch])

            value_loss = (torch.maximum(target_q_batch - q_batch, torch.zeros_like(q_batch)) ** 2).mean()
            value_loss.backward()  
            self.critic_optim.step() 

            # optimize actor
            self.actor.zero_grad()

            policy_loss = -self.critic([state_batch, self.actor([state_batch, goal_batch]), goal_batch])
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau_update)
        soft_update(self.critic_target, self.critic, self.tau_update)