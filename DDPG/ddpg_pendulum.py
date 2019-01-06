#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt
import random

import gym

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

class Agent:
    def __init__(self,args):
        self.Nepi = int(args.Nepi) # number of episode
        self.Nstep  = int(args.Nstep) # number of time-step
        self.state = [0.0,0.0,0.0] # continuous state
        self.action = [0.0] # continuous action
        self.Naction = 1
        self.Nstate  = 3
        self.env = "Pendulum-v0"
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = int(args.batch_size)
        self.replay_buffer = list()
        self.Nrep = args.Nrep
        # initialize critic network
        self.Q_input  = nn.Variable([self.batch_size,self.Nstate+self.Naction])
        with nn.parameter_scope("critic"):
            self.Q  = self.critic_network(self.Q_input,self.Nstate+self.Naction)
        self.targetQ_input = nn.Variable([self.batch_size,self.Nstate+self.Naction])
        with nn.parameter_scope("target-critic"):
            self.targetQ  = self.critic_network(self.targetQ_input,self.Nstate+self.Naction)
        self.critic_solver = S.Adam(args.critic_learning_rate)
        # initialize actor network
        self.Mu_input = nn.Variable([self.batch_size,self.Nstate])
        with nn.parameter_scope("actor"):
            self.Mu = self.actor_network(self.Mu_input,self.Nstate)
        self.targetMu_input = nn.Variable([self.batch_size,self.Nstate])
        with nn.parameter_scope("target-actor"):
            self.targetMu = self.actor_network(self.targetMu_input,self.Nstate)
        self.actor_solver = S.Adam(args.actor_learning_rate)
        # temporal variables
        self.y = nn.Variable([self.batch_size, self.Nstate + self.Naction])
        self.t = nn.Variable([self.batch_size, self.Nstate + self.Naction])


    ''' member function '''
    def critic_network(self,x,n,test=False):
        nn.clear_parameters()
        # input layer
        with nn.parameter_scope('Affine'):
            h = PF.affine(x,n)
        with nn.parameter_scope("BatchNormalization"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu'):
            h = F.relu(h)
        # hidden layer 1
        with nn.parameter_scope('Affine1'):
            h = PF.affine(h,n/2)
        with nn.parameter_scope("BatchNormalization1"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu1'):
            h = F.relu(h)
        # hidden layer 2
        with nn.parameter_scope('Affine2'):
            h = PF.affine(h,n/2)
        with nn.parameter_scope("BatchNormalization2"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        with nn.parameter_scope('Relu2'):
            h = F.relu(h)
        # output layer
        with nn.parameter_scope('Affine4'):
            h = PF.affine(h,1)
        with nn.parameter_scope("BatchNormalization4"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        return h

    def actor_network(self,x,n,test=False):
        # input layer
        nn.clear_parameters()
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
            h = PF.affine(h,1)
        with nn.parameter_scope("BatchNormalization4"):
            h = PF.batch_normalization(h,(1,),0.9,0.0001,not test)
        return h

    def policy(self,s):
        self.targetMu_input.d[0]  = np.array(s)
        self.targetMu.forward()
        return self.targetMu.d[0]

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

    def updateMu(self):
        self.actor_solver.zero_grad()  # Initialize gradients of all parameters to zero.
        minibatch = self.get_minibatch()
        batch_s = np.array([b[0] for b in minibatch])
        self.Mu_input.d = batch_s
        self.Mu.forward()
        self.Q_input.d = np.hstack((batch_s,self.Mu.d))
        self.Q.forward()
        self.y.d = -1.0*self.Q.d
        self.t.d = np.zeros((self.batch_size,self.Nstate+self.Naction))
        actor_loss = F.mean(F.huber_loss(self.y,self.t))
        actor_loss.backward()
        logger.info("actor_loss = %f " % actor_loss.d)
        self.actor_solver.weight_decay(args.actor_learning_rate)  # Applying weight decay as an regularization
        self.actor_solver.update()

    def updateQ(self):
        self.critic_solver.zero_grad()  # Initialize gradients of all parameters to zero.
        minibatch = self.get_minibatch()
        batch_s = np.array([b[0] for b in minibatch])
        batch_s_next = np.array([b[1] for b in minibatch])
        batch_action = np.array([np.array([float(b[2])]) for b in minibatch])

        batch_reward = np.array([np.array([b[3]]) for b in minibatch])
        self.targetMu_input.d =  batch_s
        self.targetMu.forward()
        self.targetQ_input.d = np.hstack((batch_s_next,self.targetMu.d))
        self.targetQ.forward()
        self.y.d = batch_reward + self.gamma * self.targetQ.d
        self.Q_input.d = np.hstack((batch_s,batch_action))
        self.Q.forward() # ??
        critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        critic_loss.backward()
        logger.info("critic_loss = %f " % critic_loss.d)
        self.critic_solver.weight_decay(args.critic_learning_rate)  # Applying weight decay as an regularization
        self.critic_solver.update()

    def update_targetQ(self):
        '''soft update by tau '''
        self.targetQ.d = self.tau * self.Q.d.copy() + (1.0 - self.tau) * self.targetQ.d.copy()
        #print self.targetQ.d

    def update_targetMu(self):
        '''soft update by tau '''
        self.targetMu.d = self.tau * self.Mu.d.copy() + (1.0 - self.tau) * self.targetMu.d.copy()

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
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            observation = env.reset()
            if args.render == 1 :
                env.render()
            s, a, noise = [rnd.random(),rnd.random(),rnd.random()], rnd.random(), rnd.random()
            t, game_over = 0, False
            while(t<self.Nstep):# loop for timestep
                t += 1
                a_next = self.policy(s) + noise
                s_next, reward, done, info = env.step(a_next)
                # update Q-network
                self.push_replay_buffer([s,s_next,a,reward,done])
                if len(self.replay_buffer) % self.Nrep == 0:
                    self.updateQ()
                    self.update_targetQ()
                    self.updateMu()
                    self.update_targetMu()
                # remember current state and action
                s ,a = s_next, a_next
                if game_over == True :
                    logger.info("finished a episode.")
                    break

        print("Training finished.")

    #plot graph
    def plot(self):
        '''
        #x =
        #y =
        #plt.title("Q("+self.s_init+","+self.a_init+")")
        #plt.ylim([2,11])
        #plt.plot(x,y)
        #plt.show()
        '''

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    #parser.add_argument("--env",type=str,default="MountainCarContinuous-v0")
    #parser.add_argument("--env",type=str,default="Pendulum-v0")
    parser.add_argument("--batch_size","-b",type=int,default=32)
    parser.add_argument("-c","--context",type=str,default="cpu",help="specify cpu or cudnn.")
    parser.add_argument("--tau","-tau",type=float,default=0.001)
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--Nepi","-Nepi",type=int,default=3000)
    parser.add_argument("--Nstep","-Nstep",type=int,default=1000)
    parser.add_argument("--gamma", "-gamma", type=float, default=0.99)
    parser.add_argument("--actor_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--critic_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--device_id",type=int,default=0)
    parser.add_argument("--Nrep",type=int,default=100)
    parser.add_argument("--render",type=int,default=1)

    args = parser.parse_args()
    AI = Agent(args)
    AI.train(args)
