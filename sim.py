#!/usr/bin/env python

import greedy
#import eps_greedy
import numpy as np
#import matplotlib

# main
if __name__ == "__main__":
    N = 1000
    result = []
    sum = np.zeros(greedy.K)
    for i in range(N):
        result.append(greedy.greedy(i))
        for ik in range(greedy.K):
            sum[ik] += result[i][ik]
    # calc all sum
    sum_all = 0.0
    rate = np.zeros(greedy.K)

    for ik in range(greedy.K):
        sum_all += sum[ik]
    for ik in range(greedy.K):
        rate[ik] = sum[ik]/sum_all
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "arms=",greedy.arms
    print "N=",N
    print "=====greedy argorithm====="
    print "Ntrial = ", greedy.Ns
    print "distribution = ", sum
    print "rate         = ", rate
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"