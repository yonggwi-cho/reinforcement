'''
Created on 2017/07/28

@author: Yong-Gwi Cho
'''

# local paramaters
alpha = 0.01
gamma = 0.8

def action_value(q,q_next,r):
    return (1.0-alpha)*q + alpha*(r+gamma*q_next)

