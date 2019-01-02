#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt

import gym

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

class Agent:
    def __init__(self,args):
        q_init = 10
        self.Nepi = args.Nepi # number of episode
        self.Nstep  = args.Nstep # number of time-step
        self.state = 0.0
        self.action = 0.0
        self.env = args.env # object of environment
        self.gamma = args.gamma
        self.tau = args.tau
        self.replay_buffer = list()
        self.Nrep = args.Nrep
        # initialize Q network
        self.Q  = critic_network()
        self.target_Q  = critic_network()
        # initialize Mu network
        self.Mu = actor_network()
        self.target_Mu = actor_network()

    """ member function """
    def critic_network(x,n,test=False):
        # input layer
        with nn.parameter_scope('Affine'):
            h = PF.affine(x,n)
        with nn.parameter_scope("BatchNormalization"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu'):
            h = F.relu(h)
        # hidden layer 1
        with nn.parameter_scope('Affine1'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization1"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu1'):
            h = F.relu(h)
        # hidden layer 2
        with nn.parameter_scope('Affine2'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization2"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu2'):
            h = F.relu(h)
        # output layer
        with nn.parameter_scope('Affine4'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization4"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        return h

    def actor_network(x,n,test=False):
        # input layer
        with nn.parameter_scope('Affine'):
            h = PF.affine(x,n)
        with nn.parameter_scope("BatchNormalization"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu'):
            h = F.relu(h)
        # hidden layer 1
        with nn.parameter_scope('Affine1'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization1"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu1'):
            h = F.relu(h)
        # hidden layer 2
        with nn.parameter_scope('Affine2'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization2"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu2'):
            h = F.relu(h)
        # output layer
        with nn.parameter_scope('Affine4'):
            h = PF.affine(h,n)
        with nn.parameter_scope("BatchNormalization4"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        return h

    def policy(self,s):
        return self.target_Mu(s)

    def updateQ(self,s,a,s_next,a_next):

    def updateMu(self,s,a,s_next,a_next):

    def push_replay_buffer(self,history):
        if len(replay_buffer) <  Nrep :
            self.replay_buffer.append(history)
        elif len(replay_buffer) >= Nrep
            self.replay_buffer.pop(0)
            self.replay_buffer.append(history)

    def train(self,args):
        # Get context.
        from nnabla.ext_utils import get_extension_context
        logger.info("Running in %s" % args.context)
        ctx = get_extension_context(
            args.context, device_id=args.device_id, type_config=args.type_config)
        nn.set_default_context(ctx)
        # init env
        env = gym.make(args.env)
        iepi=0
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            observation = env.reset()
            if args.render == 1 :
                env.render()
            s = rnd.random()
            a = rnd.random()
            noise = rnd.random()
            t,game_over = 0,False
            while(t<self.Nstep):# loop for timestep
                t += 1
                a_next = self.policy(s) + noise
                s_next, reward, done, info = env.step(a_next)
                # update Q-network
                self.push_replay_buffer([s,s_next,a,a_next,reward])
                if len(replay_buffer) % Nrep == 0:
                    self.updateQ(s,a,s_next,a_next)
                # remember current state and action
                s = s_next
                a = a_next
                if game_over == Truse :
                    print("finished a episode.")
                    break
            # update actor-network
            self.updateMu(s,a,s_next,a_next)
        print("Training finished.")

    #plot graph
    def plot(self):
        #x =
        #y =
        #plt.title("Q("+self.s_init+","+self.a_init+")")
        #plt.ylim([2,11])
        #plt.plot(x,y)
        #plt.show()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    #parser.add_argument("--env",type=str,default="MountainCarContinuous-v0")
    parser.add_argument("--env",type=str,default="Pendulum-v0")
    parser.add_argument("--batch_size","-b",type=int,default=32)
    parser.add_argument("-c","--c",type=str,default="cpu",help="specify cpu or cudnn.")
    parser.add_argument("--tau","-tau",type=float,default=0.001)
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--Nepi","-Nepi",type=int,default=3000)
    parser.add_argument("--Nstep","-Nstep",type=int,default=1000)
    parser.add_argument("--gamma","-gamma",type=float,default=0.99)
    parser.add_argument("--device_id",type=int,default=0)
    parser.add_argument("--Nrep",type=int,default=100)
    parser.add_argument("--render",type=int,default=1)

    AI = Agent(args)
    AI.train()
