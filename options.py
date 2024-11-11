# -*- coding: utf-8 -*-
"""
@Author: sym
@File: options.py
@Time: 2024/7/15
"""
import argparse
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./pretrain/p2t_small.pth', help='train from checkpoints')
parser.add_argument('--load_P2T', type=str, default='./cpts/ACINet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./datasets/train_2985/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='./datasets/train_2985/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./datasets/train_2985/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='./datasets/test/NLPR/RGB/', help='the test gt images root')
parser.add_argument('--test_depth_root', type=str, default='./datasets/test/NLPR/depth/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='./datasets/test/NLPR/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./cpts/ACINet/', help='the path to save models and logs')
opt = parser.parse_args() 
