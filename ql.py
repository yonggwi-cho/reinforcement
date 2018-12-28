'''
Created on 2017/08/09

@author: Yong-Gwi Cho
implementation of Q-Learning algorithm

'''
import numpy.random as rnd

# local paramaters
alpha = 0.01
gamma = 0.8

def max_random(q):
    for i in range(len(q)):
        if q[0] != q[i] :
            break
        else :
            return max(q)

    return rnd.choice(q)

def action_value(q, q_next_list, r):
    q_next = max_random(q_next_list)
    return (1.0-alpha)*q + alpha*(r+gamma*q_next)

