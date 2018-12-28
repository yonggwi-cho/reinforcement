#!/usr/bin/env python

'''
Created on 2017/07/28

@author: Yong-Gwi Cho
'''
import numpy as np
import numpy.random as rnd
import sys
import argparse
import matplotlib.pyplot as plt

import sarsa
import ql

class Environment:
    """define environment"""

    def __init__(self):
        self.trans_matrix  = np.array([["s3","s2"],
                                       ["s1","s4"],
                                       ["s4","s1"],
                                       ["none","none"]],
                                      dtype=str)
        self.reward_matrix = np.array([[1.0,0.0],
                                       [-1.0,1.0],
                                       [5.0,-100.0],
                                       [0.0,0.0]],
                                      dtype=float)

    def reward(self,s,a):
        return self.reward_matrix[s][a]

    # transition
    def step(self,s,a):
        return self.trans_matrix[s][a]

class Agent:

    def __init__(self,env,method):
        q_init = 10
        self.Nepi = 3000 # number of episode
        self.Nstep  = 1000
        self.state = [ "s1","s2","s3","s4" ] # list for state space
        self.action = [ "a1", "a2" ] # list for action space
        self.N_next = 2 # number of possibly taken next state
        self.env = env # object of environment
        self.s_init = "none"
        self.a_init = "none"
        self.method = method
        self.Q_history = list()
        # define Q array
        self.Q = np.array([[q_init,q_init],[q_init,q_init],
                           [q_init,q_init],[q_init,q_init]],dtype=float)

    """ member function """
    #policy : randomly choose a1 or a2
    def policy(self,s): # s stards for a state which not used in this policy.
        prob = rnd.rand()
        if prob >= 0.5 :
            return "a1"
        else :
            return "a2"

    def get_nextQlist(self,s,Q):
        Qlist = list()
        for i in range(self.N_next):# loop for possible next actions
            Qlist.append(Q[self.state.index(s)][i])
        return Qlist

    def updateQ(self,s,a,s_next,a_next):
        if self.method == "sarsa" :
            self.Q[self.state.index(s)][self.action.index(a)] \
                    = sarsa.action_value(self.Q[self.state.index(s)][self.action.index(a)],\
                                         self.Q[self.state.index(s_next)][self.action.index(a_next)],\
                                         self.env.reward(self.state.index(s),self.action.index(a)))
        elif self.method == "ql" :
            self.Q[self.state.index(s)][self.action.index(a)] \
                    = ql.action_value(self.Q[self.state.index(s)][self.action.index(a)],\
                                      self.get_nextQlist(s_next,self.Q),\
                                      self.env.reward(self.state.index(s),self.action.index(a)))
        else :
            print("method for updating Q-tabel is not specified.")
            sys.exit(1)

    def train(self,s_init,a_init):

        self.Q_history = list()

        #initial value
        self.s_init = s_init
        self.a_init = a_init

        # loop
        iepi=0
        while(iepi<self.Nepi):# loop for epithod
            iepi  += 1
            s = self.s_init
            a = self.a_init
            icount = 0
            while(icount<self.Nstep) : # loop for timestep
                icount += 1
                s_next = self.env.step(self.state.index(s),self.action.index(a))
                a_next = self.policy(s)

                # update Q-table
                self.updateQ(s,a,s_next,a_next)

                # remember current ???
                s = s_next
                a = a_next

                if s == "s4" :
                    #print("reached s4!!")
                    break

            # store Q[s_init,a_init]
            self.Q_history.append(self.Q[self.state.index(self.s_init)][self.action.index(self.a_init)])

        print("Training was finished.")



    #plot graph
    def plot(self):
        x = np.arange(len(self.Q_history))
        y = np.array(self.Q_history)
        plt.title("Q("+self.s_init+","+self.a_init+")")
        #plt.ylim([0,11])
        plt.plot(x,y)
        plt.show()

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--s_init",default="s1",type=str)
    parser.add_argument("--a_init",default="a1",type=str)
    parser.add_argument("--method",default="ql",type=str)
    args = parser.parse_args()

    env = Environment()
    agent = Agent(env,args.method)
    agent.train(args.s_init,args.a_init)
    agent.plot()

    #agent1 = Agent(env,"ql")
    #agent1.train("s1","a2")
    #agent1.plot()
