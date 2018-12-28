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

    def reward(self,s,sp,a):
        if s == "s1" :
            if a == "a2" and sp == "s2" :
                return 1.0
            elif a == "a1" and sp == "s3":
                return 0.0
            else :
                sys.exit(0)
        elif s == "s2" :
            if a == "a1" and sp == "s1" :
                return -1.0
            elif a == "a2" and sp == "s4" :
                return 1.0
            else :
                sys.exit(0)
        elif s == "s3" :
            if a == "a1" and sp == "s4" :
                return 5.0
            if a == "a2" and sp == "s1" :
                return -100.0
            else :
                sys.exit(0)
        elif s == "s4" :
            return 0.0

    # transition
    def step(self,s,a):
        if s == "s1" :
            if a == "a1" :
                return "s3"
            elif a == "a2" :
                return "s2"
        elif s == "s2" :
            if a == "a1" :
                return "s1"
            elif a == "a2" :
                return "s4"
        elif s == "s3" :
            if a == "a1" :
                return "s4"
            elif a == "a2" :
                return "s1"
        elif s == "s4" :
            print "finished."
            return "s4"

class Agent:

    def __init__(self,env,method):
        q_init = 10
        self.Nepi = 3000 # number of episode
        self.Nstep  = 10000
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
        #print "p=",prob
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
                                         self.env.reward(s,s_next,a) )
        elif self.method == "ql" :
            self.Q[self.state.index(s)][self.action.index(a)] \
                    = ql.action_value(self.Q[self.state.index(s)][self.action.index(a)],\
                                      self.get_nextQlist(s,self.Q),\
                                      self.env.reward(s,s_next,a) )
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
                s_next = self.env.step(s,a)
                a_next = self.policy(s)
                #print i, icount, s_next, s, a_next, a

                # update Q-table
                self.updateQ(s,a,s_next,a_next)

                # remember current ???
                s = s_next
                a = a_next

                # store Q[s_init,a_init]
                self.Q_history.append(self.Q[self.state.index(self.s_init)][self.action.index(self.a_init)])
                #print "Q=", self.Q[self.state.index(self.s_init)][self.action.index(self.a_init)]

                if s == "s4" :
                    #print("reached s4!!")
                    break

        print("Training was finished.")

    #plot graph
    def plot(self):
        x = np.arange(len(self.Q_history))
        y = np.array(self.Q_history)
        plt.title("Q("+self.s_init+","+self.a_init+")")
        plt.ylim([0,11])
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
