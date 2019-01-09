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
import nnabla.experimental.viewers as V

import OUNoise as ou

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
        self.Naction = 1
        self.Nstate  = 3
        self.env = "Pendulum-v0"
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = int(args.batch_size)
        self.replay_buffer = list()
        self.Nrep = args.Nrep
        # initialize critic network
        self.Q_input = nn.Variable([self.batch_size, self.Nstate+self.Naction])
        with nn.parameter_scope("critic"):
            self.Q  = self.critic_network(self.Q_input,self.Nstate+self.Naction)
            self.Q.persistent = True
            self.critic_solver = S.Adam(args.critic_learning_rate)
            self.critic_solver.set_parameters(nn.get_parameters())
        self.targetQ_input = nn.Variable([self.batch_size, self.Nstate + self.Naction])
        with nn.parameter_scope("target-critic"):
            self.targetQ  = self.critic_network(self.targetQ_input,self.Nstate+self.Naction)
            self.targetQ.persistent = True
        # initialize actor network
        self.Mu_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("actor"):
            self.Mu = self.actor_network(self.Mu_input,self.Nstate)
            self.actor_solver = S.Adam(args.actor_learning_rate)
            self.actor_solver.set_parameters(nn.get_parameters())
        self.targetMu_input = nn.Variable([self.batch_size, self.Nstate])
        with nn.parameter_scope("target-actor"):
            self.targetMu = self.actor_network(self.targetMu_input,self.Nstate)
        # temporal variables
        self.y = nn.Variable([self.batch_size, self.Nstate + self.Naction])
        self.t = nn.Variable([self.batch_size, self.Nstate])

    ''' member function '''
    def critic_network(self,x,n,test=False):
        # input layer
        with nn.parameter_scope("layer1"):
            h = PF.affine(x,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # hidden layer 1
        with nn.parameter_scope("layer2"):
            h = PF.affine(h,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # hidden layer 2
        with nn.parameter_scope("layer3"):
            h = PF.affine(h,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # output layer
        with nn.parameter_scope("layer4"):
            h = PF.affine(h,1)
            #h = PF.batch_normalization(h, batch_stat=not test)
        return h

    def actor_network(self,x,n,test=False):
        # input layer
        with nn.parameter_scope("layer1"):
            h = PF.affine(x,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # hidden layer 1
        with nn.parameter_scope("layer2"):
            h = PF.affine(h,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # hidden layer 2
        with nn.parameter_scope("layer3"):
            h = PF.affine(h,n)
            #h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # output layer
        with nn.parameter_scope("layer4"):
            h = PF.affine(h,1)
            #h = PF.batch_normalization(h, batch_stat=not test)
        # normalization for action space
        h = 2.0*F.tanh(h)
        return h

    def save_network(self,fname):
        nn.save_parameters(fname + '.h5')
        print nn.get_parameters()
        with nn.parameter_scope("critic"):
            print nn.get_parameters()

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
        self.t.d =  batch_s
        self.Q_input = F.concatenate(self.t, self.Mu)
        self.Q.forward()
        actor_loss = -1.0*F.mean(self.Q)
        actor_loss.forward()
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
        self.targetMu_input.d =  batch_s_next
        self.targetMu.forward()
        self.targetQ_input.d = np.hstack((batch_s_next,self.targetMu.d))
        self.targetQ.forward()
        self.y.d = batch_reward + self.gamma * self.targetQ.d
        self.Q_input.d = np.hstack((batch_s,batch_action))
        self.Q.forward()
        #critic_loss = F.mean(F.huber_loss(self.y, self.Q))
        critic_loss = F.mean(F.squared_error(self.y, self.Q))
        critic_loss.forward()
        critic_loss.backward()
        logger.info("critic_loss = %f " % critic_loss.d)
        self.critic_solver.weight_decay(args.critic_learning_rate)  # Applying weight decay as an regularization
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

    def update_targetMu(self):
        '''
        soft updation by tau
        copy parameter from actor to target-actor
        '''
        with nn.parameter_scope("actor"):
            src = nn.get_parameters()
        with nn.parameter_scope("target-actor"):
            dst = nn.get_parameters()
        for (s_key, s_val), (d_key, d_val) in zip(src.items(), dst.items()):
            d_val.d = self.tau * s_val.d.copy() + (1.0 - self.tau) * d_val.d.copy()


    def train(self,args):
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
                    #self.save_network("befor_update")
                    self.updateQ()
                    #graph.view(self.targetQ)
                    #self.targetQ.visit(PrintFunc())
                    self.update_targetQ()
                    #self.save_network("after_update")
                    #self.targetQ.visit(PrintFunc())
                    #graph.view(self.targetQ)
                    #sys.exit(0)
                    self.updateMu()
                    self.update_targetMu()
                # remember current state and action
                s = s_next
                if game_over == True :
                    logger.info("finished a episode.")
                    break
            logger.info("A episode finished.")
        print("Training finished.")

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
