'''
Created on 2017/08/09

@author: Yong-Gwi Cho
implementation of Q-Learning algorithm

'''
# local paramaters
alpha = 0.01
gamma = 0.8

def action_value(q, q_next_list, r):
    q_next = max(q_next_list)
    return (1.0-alpha)*q + alpha*(r+gamma*q_next)

