'''
Created on 2017/07/28

@author: Yong-Gwi Cho
'''
import numpy as np
import numpy.random as rnd
import sys
import matplotlib.pyplot as plt
import sarsa
import ql

# list for state space
state = [ "s1","s2","s3","s4" ]

# list for action space
action = [ "a1", "a2" ]

# number of possibly taken next state
N_next = 2

#policy : randomly choose a1 or a2
def policy(s): # s stards for a state which not used in this policy.
    if rnd.rand() >= 0.5 :
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

#
def get_nextQ(s,Q):
    list_Q = list()
    for i in range(N_next):# loop for possible next actions
        list_Q.append(Q[state.index(s)][i])
    return list_Q
# main
if __name__ == '__main__':
    Nm = 3000 # number of episode
    Nt = 10000
    Q_history = list()
    #init value
    s_init = "s1"
    a_init = "a1"
    icount = 0
    q_init = 10.0
    #method = "sarsa" # or "ql"
    method = "ql"

#   define Q array
    Q = np.array([[q_init,q_init],[q_init,q_init],
                    [q_init,q_init],[q_init,q_init]])

    # loop
    for i in range(Nm): # loop for episode
        s = s_init
        a = a_init
        while(icount<Nt) : # loop for timestep
            icount += 1
            s_next = transition(s,a)
            a_next = policy(s)
            #print("->"+s+" r="+str(reward(s,s_next,a)))
            if method == "sarsa" :
                Q[state.index(s)][action.index(a)] \
                    = sarsa.action_value(Q[state.index(s)][action.index(a)],\
                                         Q[state.index(s_next)][action.index(a_next)],\
                                         reward(s,s_next,a) )
            elif method == "ql" :
                list_Q = get_nextQ(s,Q)
                Q[state.index(s)][action.index(a)] \
                    = ql.action_value(Q[state.index(s)][action.index(a)],\
                                      list_Q,\
                                      reward(s,s_next,a) )
            s = s_next
            a = a_next
            Q_history.append(Q[state.index(s_init)][action.index(a_init)])
            if s == "s4" :
                #print("->"+s+" end.")
                break

    # plot
    x = np.arange(len(Q_history))
    y = np.array(Q_history)
    plt.title("Q("+s_init+","+a_init+")")
    plt.ylim([2,11])
    plt.plot(x,y)
    plt.show()