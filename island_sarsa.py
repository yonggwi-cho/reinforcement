'''
Created on 2017/07/28

@author: 0145215059
'''
import sarsa
import numpy as np
import numpy.random as rnd
import sys
import matplotlib.pyplot as plt

# list for state space
state = [ "s1","s2","s3","s4" ]

# list for action space
action = [ "a1", "a2" ]

#policy : randomly choose a1 or a2
def policy(s): # s stards for a state which not used in this policy.
    if rnd.rand() > 0.5 :
        return "a1"
    else :
        return "a2"

# setting environment
def reward(s,sp,a):
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
        #print "!!!s4 is final state.!!!"
        return 0.0

# transition
def transition(s,a):
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
        #print "s4 is already final"
        #sys.exit()
        return "s4"

# main
if __name__ == '__main__':
    Nm = 3000 # number of episode
    #Nm = 1 # number of episode
    Nt = 5000
    Q_history = []
    #init value
    s_init = "s1"
    a_init = "a1"
    q = 10.0
    q_next = 10.0
    icount = 0
    q_init = 10.0

    for i in range(Nm):
        s = s_init
        a = a_init
        while(icount<Nt) :
            icount += 1
            s_next = transition(s,a)
            a_next = policy(s)
            #print("->"+s+" r="+str(reward(s,s_next,a)))
            q = sarsa.action_value(q,q_next,reward(s,s_next,a))
            q_next = q
            #print Q_history
            s = s_next
            a = a_next
            #Q_history.append(Q[state.index(s_init)][action.index(a_init)])
            if s == "s4" :
                Q_history.append(q)
                #print("->"+s+" end.")
                break

    # plot
    x = np.arange(len(Q_history))
    y = np.array(Q_history)
    plt.plot(x,y)
    plt.show()