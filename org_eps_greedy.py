#!/usr/bin/env python

import sys
import os

import numpy as np
import numpy.random as rd
#import scipy.stats as st
#import matplotlib.pyplot as plt

#set params
#N = int(sys.argv[1])
#N = 100
Ns = 1
bet = 1
rate = 2
K = 4
eps = 0.1

arms = np.array([0.2,0.3,0.4,0.5])

# setting bandit
def bandit(a):
    # generate random number 0~1.
    p = rd.rand()
    # return win or false by probality a
    if p <= a :
        return True
    else :
        return False

# play
def play(a):
    res = bandit(a)
    if res == True :
        money = bet*rate
    else :
        money = 0.0
    return money

# multi-time search
def search(Nplay,a):
    reward = np.zeros(Nplay)
    for i in range(Nplay) :
        reward[i] = play(a)
    # mean
    sum = 0.0
    for i in range(Nplay):
        sum += reward[i]
    return sum

# eps-greedy argorithm
def eps_greedy(N):
    sum = np.zeros(K)
    mean = np.zeros(K)
    # search * Nx
    for ik in range(K):
        sum[ik] = search(Ns,arms[ik])
        mean[ik] = sum[ik]/float(Ns)

    Nres = np.zeros(K)
    Nres[:] = Ns
    Ncount = 0

    #loop
    while(Ncount < N):
        rnd = rd.rand()
        Ncount += 1
        if eps >= rnd :
            rnd1 = rd.rand()
            iarm = int(rnd1*K)
            if iarm > K-1 :
                iarm = K-1
            sum[iarm] = play(arms[iarm])
            Nres[iarm] += 1
            mean[iarm] = sum[iarm]/float(Nres[iarm])
        else :
            imax = np.argmax(mean)
            sum[imax] += play(arms[imax])
            Nres[imax] += 1
            mean[imax] = sum[imax]/float(Nres[imax])

    return Nres

# main
if __name__ == "__main__":
    N = 10000
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "arms=",arms
    print "Ntrial=",Ns
    print "N=",N
    print "distribution = ",eps_greedy(N)
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
