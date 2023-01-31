#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import os
dirpath = os.path.dirname(os.path.abspath(__file__))

def init_rl_parser():

        parser = argparse.ArgumentParser(description='RL_default')

        # RL hypers
        parser.add_argument('--state_dim', type=int, default=2048, help='maximum length of trajectory')
        parser.add_argument('--action_dim', type=int, default=2048, help='maximum length of trajectory')
        parser.add_argument('--goal_dim', type=int, default=2048, help='maximum length of trajectory')
        parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
        parser.add_argument('--max_episode_length', type=int, default=10, help='maximum length of trajectory')
        parser.add_argument('--knn', type=int, default=1, help='knn for action selection')
        parser.add_argument('--load', default=False, help='load a trained model')
        parser.add_argument('--load-model-dir', default='', help='folder to load trained models from')
        parser.add_argument('--rl_device', type=int, default=0, help='GPUs to use')
        parser.add_argument('--max_episode', type=int, default=100000, help='maximum #episode.')
        parser.add_argument('--test_episode', type=int, default=1000, help='maximum testing #episode.')
        parser.add_argument('--mode', default='train', type=str, help='train/test')
        parser.add_argument('--hidden1', default=512, type=int, help='hidden num 1st FCN')
        parser.add_argument('--hidden2', default=256, type=int, help='hidden num 2nd FCN')
        parser.add_argument('--c-lr', default=0.0005, type=float, help='critic net lr')
        parser.add_argument('--p-lr', default=0.0001, type=float, help='policy net lr')
        parser.add_argument('--warmup', default=100, type=int, help='steps for filling buffer only')
        parser.add_argument('--bsize', default=32, type=int, help='batch size')
        parser.add_argument('--rmsize', default=10000, type=int, help='memory size')
        parser.add_argument('--g_prop', default=0.1, type=int, help='Proportion of relabeling')
        parser.add_argument('--tau-update', default=0.001, type=float, help='moving average for target network')
        parser.add_argument('--ou_theta', default=0.15, type=float, help='noise hype1')
        parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise hype2')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mean')
        parser.add_argument('--epsilon', default=80000, type=int, help='linear decay of exploration policy')
        parser.add_argument('--seed', default=12, type=int, help='')
        parser.add_argument('--drop-out', default=0.1, type=float, help='dropout rate for dropout layer')
        parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay for L2 Regularization loss')
        parser.add_argument('--report_steps', default=10, type=int, help='report return evern X step')        
        parser.add_argument('--save_per_epochs', default=10, type=int, help='savcat e model every X epochs')

        return parser

def init_tc_parser():
    
        parser = argparse.ArgumentParser(description='TC_default')

        ## Tc_check_api
        parser.add_argument('--rmt_ckpt', default=dirpath + '/model_ckpt/rmt.pt', type=str, help='ckpt for rmt')
        parser.add_argument('--fmt_ckpt', default=dirpath + '/model_ckpt/fmt.pt', type=str, help='ckpt for fmt')
        parser.add_argument('--vocab_ckpt', default=dirpath + '/model_ckpt/mt.retro.vocab.src', type=str, help='ckpt for fmt')
        parser.add_argument('--rmt_device', default=0, type=int, help='GPUs to use')
        parser.add_argument('--fmt_device', default=0, type=int, help='GPUs to use')
        parser.add_argument('--topk', default=50, type=int, help='Max expansion for single-step tc check')
        parser.add_argument('--conf_cut_off', default=0.6, type=float, help='Minimum cutoff for reaction confidence, \
                                                                             recommend > 0.5 for chemical reliability')
        

        return parser

def init_mat_parser():

        parser = argparse.ArgumentParser(description='Material_os')

        ## Material_api
        parser.add_argument('--mat_file', default=dirpath + '/model_ckpt/materials.txt', type=str, help='file for materials')

        return parser

def init_env_parser():
        parser = argparse.ArgumentParser(description='Env_os')

        ## Environment file
        parser.add_argument('--train_file', default=dirpath + '/model_ckpt/train_mol_sample.pkl', type=str, help='sample file for training mol')

        return parser
        
rl_opt = init_rl_parser().parse_args()
tc_opt = init_tc_parser().parse_args()
mat_opt = init_mat_parser().parse_args()
env_opt = init_env_parser().parse_args()