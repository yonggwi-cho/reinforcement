#!/usr/bin/env python

import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt

import gym

class Agent:
    def __init__(self,env,method):
        q_init = 10
        self.Nepi = 3000 # number of episode
        self.Nstep  = 1000
        self.state = list()
        self.action = list()
        self.N_next = 2 # number of possibly taken next state
        self.env = env # object of environment
        self.s_init = "none"
        self.a_init = "none"
        self.method = method
        self.Q_history = list()
        # initialize Q network
        self.Q  = list()
        # initialize Mu network
        self.Mu = list()

    """ member function """
    def policy(self,s):
        return self.Mu(s)

    def updateQ(self,s,a,s_next,a_next):

    def updateMu(self,s,a,s_next,a_next):

    def train(self,args):
        env = gym.make(args.env)
        # loop
        iepi=0
        while(iepi<self.Nepi):# loop for epithod
            observation = env.reset()
            iepi  += 1
            s = rnd.random()
            a = rnd.random()
            icount = 0
            game_over = False
            while(icount<self.Nstep) : # loop for timestep
                icount += 1
                a_next = self.policy(s)
                s_next, reward, done, info = env.step(a_next)
                # update Q-network
                self.updateQ(s,a,s_next,a_next)
                # remember current ???
                s = s_next
                a = a_next
                if game_over == Truse :
                    print("finished a episode.")
                    break
            # update Mu-network
            self.updateMu(s,a,s_next,a_next)
        print("Training finished.")

    #plot graph
    def plot(self):
        x = np.arange(len(self.Q_history))
        y = np.array(self.Q_history)
        plt.title("Q("+self.s_init+","+self.a_init+")")
        plt.ylim([2,11])
        plt.plot(x,y)
        plt.show()

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="MountainCarContinuous-v0")
    parser.add_argument()

    AI = Agent(args)
    AI.train()
