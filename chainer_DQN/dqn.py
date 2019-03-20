#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt
import random
import math

import gym

import chainer
import chainer.functionas as F
import chainer.Links as L
from chainer import iterators,optmizer

class Qnetwork(chainer.Chain):
    def __init__(self,n_in,hidden_size,n_out):
        super(Qnetwork,self).__init__() # ???
        with self.init_scope():
            self.l1 = L.Linear(n_in,hidden_size)
            self.l2 = L.Linear(hidden_size,n_out)

    def __call__(self,x,t=None,train=True):
        h1 = self.l1(x)
        h2 = F.relu(h1)
        h3 = self.l2(h2)
        h4 = F.relu(h3)
        return F.mean_squared_error（h4,t） if train else F.solfmax(h4)

    #def copy_network(self,obj):
        # copy data

class Agent:
    def __init__(self,args):
        self.Nepi = int(args.Nepi) # number of episode
        self.Nstep  = int(args.Nstep) # number of time-step
        self.state = [0.0,0.0,0.0] # continuous state
        self.action = [0.0] # continuous action
        self.Naction = 3
        self.Nstate  = 2
        self.env = args.env
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = int(args.batch_size)
        self.replay_buffer = list()
        self.Nrep = args.Nrep
        self.eps = args.eps
        self.hidden_neuron = 100
        self.gradient_momentum = 0.95
        self.critic_learning_rate = args.critic_learning_rate
        #self.actor_learning_rate = args.actor_learning_rate
        #self.actor_loss = nn.Variable([1])
        #self.critic_loss = nn.Variable([1])
        # temporal variables
        #self.y = nn.Variable([self.batch_size, self.Naction])
        #self.t = nn.Variable([self.batch_size, self.Nstate])
        # initialize critic network
        #self.Q_input = nn.Variable([self.batch_size, self.Nstate])
        # Q network
        self.Q = Qnetwork(self.Nstate,self.hiden_neuron,self.Naction)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.Q)
        '''
        with nn.parameter_scope("critic"):
            self.Q  = self.critic_network(self.Q_input,self.hidden_neuron)
            self.Q.persistent = True
            self.critic_solver = S.Adam(args.critic_learning_rate)
            #self.critic_solver = S.RMSprop(args.critic_learning_rate,self.gradient_momentum)
            self.critic_solver.set_parameters(nn.get_parameters())
        '''
        self.targetQ = Qnetwork(self.Nstate,self.hidden_neuron,self.Naction)
        #self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        # targetQ
        '''
        self.targetQ_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("target-critic"):
            self.targetQ  = self.critic_network(self.targetQ_input,self.hidden_neuron)
            #self.targetQ.persistent = True
        '''
        self.name="dqn_env%s_Nepi%d_Nstep%d_bs%d"%(self.env,self.Nepi,self.Nstep,self.batch_size)


    ''' member function '''
    '''
    def save_network(self,network,fname):
        with nn.parameter_scope(network):
            nn.save_parameters(network+"_"+fname + '.h5')
            print nn.get_parameters()

    def load_network(self,network,fname):
        #print "network=",network
        with nn.parameter_scope(fname):
            nn.load_parameters(fname)
            src = nn.get_parameters()
        with nn.parameter_scope(network):
            dst = nn.get_parameters()
        for (s_key, s_val), (d_key, d_val) in zip(src.items(), dst.items()):
            d_val.d = s_val.d.copy()
    '''

    def policy(self,s):
        if self.eps > rnd.random() :
            q = self.Q(s)
            a = np.argmax(q)
        else :
            a = rnd.choice(range(self.Naction))
        return a

    def push_replay_buffer(self,history):
        if len(self.replay_buffer) <  self.Nrep :
            self.replay_buffer.append(history)
        elif len(self.replay_buffer) >= self.Nrep :
            self.replay_buffer.pop(0)
            self.replay_buffer.append(history)

    def get_minibatch(self):
        data = range(len(self.replay_buffer))
        index = random.sample(data,self.batch_size)
        return [self.replay_buffer[i] for i in index]

    def updateQ(self):
        minibatch = self.get_minibatch()
        batch_s = np.array([b[0] for b in minibatch])
        batch_s_next = np.array([b[1] for b in minibatch])
        batch_action = np.array([np.array([float(b[2])]) for b in minibatch])
        batch_reward = np.array([np.array([b[3]]) for b in minibatch])
        batch_done   = np.array([np.array([b[4]]) for b in minibatch])
        q  = self.targetQ(batch_s_next)
        for i in range(self.batch_size):
            if batch_done[i] :
                self.y[i] = batch_reward[i]
            else :
                self.y[i] = batch_reward[i] + self.gamma * np .amax(q)
        self.Q(batch_s)
        ''' self.critic_loss = F.mean(F.huber_loss(self.y, self.Q)) '''
        self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        self.critic_loss.forward()
        self.critic_solver.zero_grad()  # Initialize gradients of all parameters to zero.
        self.critic_loss.backward()
        self.critic_solver.update()

    def update_targetQ(self):
        '''
        soft updation by tau
        copy parameter from critic to target-critic
        '''
        '''
        with nn.parameter_scope("critic"):
            src = nn.get_parameters()
        with nn.parameter_scope("target-critic"):
            dst = nn.get_parameters()
        for (s_key, s_val), (d_key, d_val) in zip(src.items(), dst.items()):
            d_val.d = self.tau * s_val.d.copy() + (1.0 - self.tau) * d_val.d.copy()
        '''
        # copy params Q -> targerQ

    def train(self,args):
        # Get context.
        #from nnabla.ext_utils import get_extension_context
        #logger.info("Running in %s" % args.context)
        #ctx = get_extension_context(
        #    args.context, device_id=args.device_id, type_config=args.type_config)
        #nn.set_default_context(ctx)
        # init env
        env = gym.make(self.env)
        iepi=0
        total_reward = 0.0
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            s = env.reset()
            a = rnd.choice(range(self.Naction))
            logger.info("epithod %d total_reward=%f"%(iepi-1,total_reward))
            t,total_reward = 0 , 0.0
            while(t<self.Nstep):# loop for timestep
                t += 1
                a = self.policy(s)
                # step
                s_next, reward, done, info = env.step(a)
                total_reward += (reward + s_next[0])
                # update Q-network
                self.push_replay_buffer([s,s_next,a,reward,done])
                if len(self.replay_buffer) >= self.Nrep :
                    if args.render == 1:
                        env.render()
                    self.updateQ()
                    if iepi % 10 == 0 :
                        self.update_targetQ()
                    #logger.info("epithod %d timestep %d loss=%f"\
                    #            % (iepi, t, self.critic_loss.d))
                    #print self.critic_loss.d
                else:
                    logger.info("epithod %d timestep %d storing replay buffer... "\
                                % (iepi, t))
                s = s_next
                if done == True :
                    #logger.info("finished a episode.")
                    break
            #logger.info("A episode finished.")
        logger.info("Training finished.")
        #self.save_network("target-Q",\
        #                  "nnabla_params_env%s_Nepi%d_Nstep%d_batchsize%d" % \
        #                  (self.env, self.Nepi, self.Nstep, self.batch_size))
