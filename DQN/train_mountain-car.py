#!/usr/bin/env python

import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import gym
import dqn

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="MountainCar-v0")
    parser.add_argument("--batch_size","-b",type=int,default=32)
    parser.add_argument("-c","--context",type=str,default="cpu",help="specify cpu or cudnn.")
    parser.add_argument("--tau","-tau",type=float,default=1.0)
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--Nepi","-Ne",type=int,default=3000)
    parser.add_argument("--Nstep","-Ns",type=int,default=200)
    parser.add_argument("--gamma", "-gamma", type=float, default=0.9)
    parser.add_argument("--critic_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--device_id",type=int,default=0)
    parser.add_argument("--Nrep",type=int,default=3000)
    parser.add_argument("--render",type=int,default=1)
    parser.add_argument("--eps",type=float,default=0.3)
    parser.add_argument("--f-critic", "-fc", type=str, default="none", help="specify file name for critic-network.")

    args = parser.parse_args()
    AI = dqn.Agent(args)
    AI.train(args)
