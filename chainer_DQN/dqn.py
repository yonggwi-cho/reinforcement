#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt
import random
import math

import gym

'''
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
import nnabla.experimental.viewers as V
'''
import chainer

class PrintFunc(object):
    def __call__(self, nnabla_func):
        print("==========")
        print(nnabla_func.info.type_name)
        print(nnabla_func.inputs)
        print(nnabla_func.outputs)
        print(nnabla_func.info.args)

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
        self.y = nn.Variable([self.batch_size, self.Naction])
        self.t = nn.Variable([self.batch_size, self.Nstate])
        # initialize critic network
        self.Q_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("critic"):
            self.Q  = self.critic_network(self.Q_input,self.hidden_neuron)
            self.Q.persistent = True
            self.critic_solver = S.Adam(args.critic_learning_rate)
            #self.critic_solver = S.RMSprop(args.critic_learning_rate,self.gradient_momentum)
            self.critic_solver.set_parameters(nn.get_parameters())
        #self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        # targetQ
        self.targetQ_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("target-critic"):
            self.targetQ  = self.critic_network(self.targetQ_input,self.hidden_neuron)
            #self.targetQ.persistent = True
        self.name="dqn_env%s_Nepi%d_Nstep%d_bs%d"%(self.env,self.Nepi,self.Nstep,self.batch_size)

    ''' member function '''
    def critic_network(self,x,n,test=False):
        # input layer
        with nn.parameter_scope("layer1"):
            h = PF.affine(x,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            #h = F.relu(h)
            h = F.tanh(h)
        # hidden layer 1
        #with nn.parameter_scope("layer2"):
        #    h = PF.affine(h,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            #h = F.relu(h)
        #    h = F.tanh(h)
        # output layer
        with nn.parameter_scope("layer3"):
            h = PF.affine(h,self.Naction)
            #h = PF.batch_normalization(h, batch_stat=not test)
        return h

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

    def policy(self,s):
        if self.eps > rnd.random() :
            self.Q_input.d[0] = s
            self.Q.forward()
            a = np.argmax(self.Q.d[0])
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
        self.targetQ_input.d = batch_s_next
        self.targetQ.forward(clear_buffer=True)
        for i in range(self.batch_size):
            if batch_done[i] :
                self.y.d[i] = batch_reward[i]
            else :
                maxQ = np .amax(self.targetQ.d)
                self.y.d[i] = batch_reward[i] + self.gamma * maxQ
        self.Q_input.d = batch_s
        self.Q.forward()
        #print "Q=",self.Q.d
        ''' self.critic_loss = F.mean(F.huber_loss(self.y, self.Q)) '''
        self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        #self.critic_loss.forward(clear_no_need_grad=True)
        self.critic_loss.forward()
        self.critic_solver.zero_grad()  # Initialize gradients of all parameters to zero.
        #self.critic_loss.backward(clear_buffer=True)
        self.critic_loss.backward()
        #print "loss=",self.critic_loss.d
        #logger.info("critic_loss = %f " % critic_loss.d)
        #self.critic_solver.weight_decay(self.critic_learning_rate)  # Applying weight decay as an regularization
        self.critic_solver.update()

    def update_targetQ(self):
        '''
        soft updation by tau
        copy parameter from critic to target-critic
        '''
        with nn.parameter_scope("critic"):
            src = nn.get_parameters()
        with nn.parameter_scope("target-critic"):
            dst = nn.get_parameters()
        for (s_key, s_val), (d_key, d_val) in zip(src.items(), dst.items()):
            d_val.d = self.tau * s_val.d.copy() + (1.0 - self.tau) * d_val.d.copy()
            #d_val.g = self.tau * s_val.g.copy() + (1.0 - self.tau) * d_val.g.copy()
            #print s_key,s_val.d

    def train(self,args):
        # Get context.
        from nnabla.ext_utils import get_extension_context
        logger.info("Running in %s" % args.context)
        ctx = get_extension_context(
            args.context, device_id=args.device_id, type_config=args.type_config)
        nn.set_default_context(ctx)
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
        self.save_network("target-Q",\
                          "nnabla_params_env%s_Nepi%d_Nstep%d_batchsize%d" % \
                          (self.env, self.Nepi, self.Nstep, self.batch_size))


'''
    def train_from_memory(self, args):
        # Get context.
        from nnabla.ext_utils import get_extension_context
        logger.info("Running in %s" % args.context)
        ctx = get_extension_context(
            args.context, device_id=args.device_id, type_config=args.type_config)
        nn.set_default_context(ctx)
        # load params
        self.load_network("target-critic",args.f_critic)
        self.load_network("target-actor",args.f_actor)
        # init env
        env = gym.make(self.env)
        iepi = 0
        while (iepi < self.Nepi): # loop for epithod
            iepi += 1
            s = env.reset()
            a = rnd.random()
            t, game_over = 0, False
            ounoise = ou.OUNoise(self.Naction)
            total_reward = 0.0
            while (t < self.Nstep):  # loop for timestep
                #logger.info("epithod %d timestep %d" % (iepi, t))
                if args.render == 1:
                    env.render()
                # noise = 0.1*(2.0 * rnd.random() - 1.0)
                noise = ounoise.sample()
                t += 1
                if abs(self.policy(s)+noise) <= 2.0:
                    a = self.policy(s) + noise
                else:
                    a = self.policy(s)
                #print("action,noise = %f,%f" % (a, noise))
                s_next, reward, done, info = env.step(a)
                # update Q-network
                self.push_replay_buffer([s, s_next, a, reward, done])
                total_reward += reward
                if len(self.replay_buffer) >= self.Nrep:
                    self.updateQ()
                    self.update_targetQ()
                #logger.info("epithod %d timestep %d critc_loss = %f actor_loss %f "\
                #% (iepi, t,math.sqrt(self.critic_loss.d),self.actor_loss.d))
                # remember current state and action
                s = s_next
                if done == True:
                    #logger.info("finished a episode.")
                    break
            #logger.info("A episode finished.")
            logger.info("epithod %d Total_reward=%f"%(iepi-1,total_reward))
        print("Training finished.")
        self.save_network("target-critic", \
                          "nnabla_params_env%s_Nepi%d_Nstep%d_batchsize%d" % ( \
                              self.env, self.Nepi, self.Nstep, self.batch_size))
        self.save_network("target-actor", \
                          "nnabla_params_env%s_Nepi%d_Nstep%d_batchsize%d" % ( \
                              self.env, self.Nepi, self.Nstep, self.batch_size))

    def train_debug(self,args):
        graph = V.SimpleGraph(verbose=True)
        # Get context.
        from nnabla.ext_utils import get_extension_context
        logger.info("Running in %s" % args.context)
        ctx = get_extension_context(
            args.context, device_id=args.device_id, type_config=args.type_config)
        nn.set_default_context(ctx)
        # init env
        env = gym.make(self.env)
        iepi=0
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            s = env.reset()
            a = rnd.random()
            t, game_over = 0, False
            ounoise = ou.OUNoise(self.Naction)
            while(t<self.Nstep):# loop for timestep
                logger.info("epithod %d timestep %d"%(iepi,t))
                if args.render == 1:
                    env.render()
                #noise = 0.1*(2.0 * rnd.random() - 1.0)
                noise = ounoise.sample()
                t += 1
                a = self.policy(s) + noise
                print("action,noise = %f,%f"%(a,noise))
                s_next, reward, done, info = env.step(a)
                # update Q-network
                self.push_replay_buffer([s,s_next,a,reward,done])
                if len(self.replay_buffer) >= self.Nrep :
                    self.save_network("befor_update")
                    self.updateQ()
                    graph.view(self.targetQ)
                    self.targetQ.visit(PrintFunc())
                    self.update_targetQ()
                    self.save_network("after_update")
                    self.targetQ.visit(PrintFunc())
                    graph.view(self.targetQ)
                # remember current state and action
                s = s_next
                if game_over == True :
                    logger.info("finished a episode.")
                    break
            logger.info("A episode finished.")
        print("Training finished.")
        self.save_network()
'''
