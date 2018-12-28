#!/usr/bin/env python

from greedy import *

# set eps_greedy pramas
eps = 0.1

# eps-greedy argorithm
def eps_greedy(N):
    sum = np.zeros(K)
    mean = np.zeros(K)

    # search * Nx
    #for ik in range(K):
    #   sum[ik] = search(Ns,arms[ik])
    #   mean[ik] = sum[ik]/float(Ns)

    Nres = np.zeros(K)
    Ncount = 0

    #loop
    while(Ncount < N):
        rnd = rd.rand()
        Ncount += 1
        if eps <= rnd :
            #rnd1 = rd.rand()
            #iarm = int(rnd1*K)
            #if iarm > K-1 :
            #    iarm = K-1
            iarm = rd.choice(range(K),1) # randomly choose
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
    #print "Ntrial=",Ns
    print "N=",N
    print "distribution = ",eps_greedy(N)
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
