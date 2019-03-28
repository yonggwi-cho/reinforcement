#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import math

import gym

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
import nnabla.experimental.viewers as V

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
        self.env = args.env
        self.Nstate  = 2
        self.Naction = 3
        self.state = [0.0, 0.0] # continuous state
        self.action = [0, 1 ,2] # discrete action
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = int(args.batch_size)
        self.replay_buffer = list()
        self.Nrep = args.Nrep
        self.eps = args.eps
        self.hidden_neuron = 50
        self.gradient_momentum = 0.95
        self.critic_learning_rate = args.critic_learning_rate

        # State-Action Plot's parametes
        self.plim = [-1.2, 0.6]
        self.vlim = [-0.07, 0.07]
        self.N_position = 27
        self.N_velocity = 27
        self.positions = np.linspace(self.plim[0], self.plim[1], num=self.N_position, endpoint=True)
        self.velocities = np.linspace(self.vlim[0], self.vlim[1], num=self.N_velocity, endpoint=True)
        self.fig, self.axs = plt.subplots(1, 1, figsize=(5.8, 5))
        self.cb = None

        #self.actor_learning_rate = args.actor_learning_rate
        #self.actor_loss = nn.Variable([1])
        #self.critic_loss = nn.Variable([1])
        # temporal variables
        self.y = nn.Variable([self.batch_size, self.Naction])
        self.t = nn.Variable([self.batch_size, self.Nstate])
        # initialize critic network
        self.Q_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("critic"):
            #self.h1 = F.tanh(PF.affine(self.Q_input,self.hidden_neuron,name="layer1"))
            #self.Q = PF.affine(self.h1,self.Naction,name="layer2")
            self.Q  = self.critic_network(self.Q_input,self.hidden_neuron)
            self.Q.persistent = True
            #self.critic_solver = S.Adam(args.critic_learning_rate)
            self.critic_solver = S.RMSprop(args.critic_learning_rate,self.gradient_momentum)
            self.critic_solver.set_parameters(nn.get_parameters())
        #self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        # targetQ
        self.targetQ_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("target-critic"):
            self.targetQ  = self.critic_network(self.targetQ_input,self.hidden_neuron)
            #self.th1 = F.tanh(PF.affine(self.targetQ_input,self.hidden_neuron,name="tlayer1"))
            #self.targetQ = PF.affine(self.th1,self.Naction,name="tlayer2")
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
        re = rnd.random()
        if self.eps < re :
            self.targetQ_input.d[0] = s
            self.targetQ.forward(clear_buffer=True)
            a = np.argmax(self.targetQ.d[0])
        else :
            a = rnd.choice(range(self.Naction))
        return self.action[a]

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
            a = int(batch_action[i])
            #print a
            if batch_done[i] :
                self.y.d[i][a] = batch_reward[i]
            else :
                maxQ = np.amax(self.targetQ.d[i])
                self.y.d[i][a] = float(batch_reward[i]) + self.gamma * maxQ
                #print batch_reward[i],self.targetQ.d[i],maxQ, self.y.d[i]
        #print self.y.d
        self.Q_input.d = batch_s
        #print self.Q_input.d.shape,batch_s.shape
        self.Q.forward(clear_buffer=True)
        #print "Q=",self.Q.d
        ''' self.critic_loss = F.mean(F.huber_loss(self.y, self.Q)) '''
        self.critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        self.critic_loss.forward(clear_no_need_grad=True)
        self.critic_solver.zero_grad()  # Initialize gradients of all parameters to zero.
        self.critic_loss.backward(clear_buffer=True)
        #self.critic_loss.backward()
        #logger.info("critic_loss = %f " % critic_loss.d)
        self.critic_solver.weight_decay(self.critic_learning_rate)  # Applying weight decay as an regularization
        self.critic_solver.update()

    def plotQ(self,clear=False):
        # Parameters
        grid_on = True
        v_max = 0. #np.max(self.Q[0, :, :])
        v_min = -30.
        x_labels = ["%.2f" % x for x in self.positions ]
        y_labels = ["%.2f" % y for y in self.velocities]
        titles = "Actions " + u"\u25C0" + ":push_left/" + u"\u25AA" + ":no_push/" + u"\u25B6" + ":push_right"
        Q = np.zeros((self.N_velocity*len(self.action),self.N_position*len(self.action)))
        for s_2 in range(len(self.velocities)):
            for s_1 in range(len(self.positions)):
                self.targetQ_input.d = [self.positions[s_1], self.velocities[s_2]]
                self.targetQ.forward(clear_buffer=True)
                Q_hut = self.targetQ.d
                #print "Q_x:" , self.Q_x.shape , "self.Q.d:", self.Q.d.shape
                for a in range(len(self.action)):
                    Q[3 * s_2 + a, 3 * s_1 + 0] = Q_hut[0][0]
                    Q[3 * s_2 + a, 3 * s_1 + 1] = Q_hut[0][1]
                    Q[3 * s_2 + a, 3 * s_1 + 2] = Q_hut[0][2]
        im = self.axs.imshow(Q, interpolation='nearest', vmax=v_max, vmin=v_min, cmap=cm.jet)
        self.axs.grid(grid_on)
        self.axs.set_title(titles)
        self.axs.set_xlabel('Position')
        self.axs.set_ylabel('Velocity')
        x_start, x_end = self.axs.get_xlim()
        #y_start, y_end = axs.get_ylim()
        self.axs.set_xticks(np.arange(x_start, x_end, 3))
        self.axs.set_yticks(np.arange(x_start, x_end, 3))
        self.axs.set_xticklabels(x_labels, minor=False, fontsize='small', horizontalalignment='left', rotation=90)
        self.axs.set_yticklabels(y_labels, minor=False, fontsize='small', verticalalignment='top')
        self.cb = self.fig.colorbar(im, ax=self.axs)
        #
        plt.show(block=False)

    def plotTQupdate(self):
        #print self.axs
        Q_mesh = np.zeros((self.N_velocity * len(self.action), self.N_position * len(self.action)))
        #print Q_mesh
        for s_2 in range(len(self.velocities)):
            for s_1 in range(len(self.positions)):
                self.targetQ_input.d = [self.positions[s_1], self.velocities[s_2]]
                self.targetQ.forward(clear_buffer=True)
                Q_hut = self.targetQ.d
                #print "Q_x:" , self.Q_x.shape , "self.Q.d:", self.Q.d.shape
                for a in range(len(self.action)):
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 0] = Q_hut[0][0]
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 1] = Q_hut[0][1]
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 2] = Q_hut[0][2]
        self.axs.get_images()[0].set_data(Q_mesh)
        self.axs.draw_artist(self.axs.images[0])
        self.fig.canvas.blit(self.axs.bbox)

    def plotQupdate(self):
        #print self.axs
        Q_mesh = np.zeros((self.N_velocity * len(self.action), self.N_position * len(self.action)))
        #print Q_mesh
        for s_2 in range(len(self.velocities)):
            for s_1 in range(len(self.positions)):
                self.Q_input.d = [self.positions[s_1], self.velocities[s_2]]
                self.Q.forward(clear_buffer=True)
                Q_hut = self.Q.d
                #print "Q_x:" , self.Q_x.shape , "self.Q.d:", self.Q.d.shape
                for a in range(len(self.action)):
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 0] = Q_hut[0][0]
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 1] = Q_hut[0][1]
                    Q_mesh[3 * s_2 + a, 3 * s_1 + 2] = Q_hut[0][2]
        self.axs.get_images()[0].set_data(Q_mesh)
        self.axs.draw_artist(self.axs.images[0])
        self.fig.canvas.blit(self.axs.bbox)

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
        self.plotQ()
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            s = env.reset()
            a = rnd.choice(range(self.Naction))
            t,total_reward = 0 , 0.0
            #while(t<self.Nstep):# loop for timestep
            while True :
                t += 1
                a = self.policy(s)
                # step
                s_next, reward, done, info = env.step(a)
                #if s_next[0] >= -0.2 or s_next[1] >= 0.0:
                if s_next[0] >= 0.6 :
                    reward = 1
                    done = True
                else:
                    reward += 5.*np.abs(s_next[1])
                total_reward += reward
                # update Q-network
                self.push_replay_buffer([s,s_next,a,reward,done])
                #print s_next,reward,done
                #print self.replay_buffer
                if len(self.replay_buffer) >= self.Nrep :
                    if args.render == 1:
                        env.render()
                    self.updateQ()
                    if t % 100 == 0 :
                        self.update_targetQ()
                    #logger.info("epithod %d timestep %d loss=%f"\
                    #            % (iepi, t, self.critic_loss.d))
                    #print self.critic_loss.
                s = s_next
                if done == True :
                    #logger.info("finished a episode.")
                    break
            self.plotQupdate()
            #self.plotTQupdate()
            #logger.info("epithod %d timestep %d storing replay buffer... "\
            #                    % (iepi, t))
            #logger.info("A episode finished.")
            #self.eps *= 0.998
            logger.info("epithod %d total_reward =%f t = %d eps = %f "%(iepi-1,total_reward,t,self.eps))
        logger.info("Training finished.")
        self.save_network("target-Q",\
                          "nnabla_params_env%s_Nepi%d_Nstep%d_batchsize%d" % \
                          (self.env, self.Nepi, self.Nstep, self.batch_size))
