import numpy as np
import matplotlib.pyplot as plt
import random

S = np.array([0, 1, 2, 3])
A = np.array([0, 1])
R = np.array([[0, 1], [-1, 1], [5, -100], [0, 0]])
S1 = np.array([[2, 1], [0, 3], [3, 0], [None, None]])

a1_per = 0.5
alpha  = 0.01
gamma  = 0.8
epi    = 3000
q_init = 10

Q = np.zeros(R.shape)
Q += q_init

def pi(a1_per):
    if random.random() <= a1_per: return 0
    return 1

def sarsa():
    for s in S:
        for a in A:
            if S1[s][a] == None: continue
            TD = R[s][a] + gamma * Q[S1[s][a], pi(a1_per)] - Q[s][a]
            Q[s][a] += alpha * TD

result_step = []
result_q = []
for step in range(epi):
    sarsa()
    if (step + 1) % 10 == 0:
        result_step.append(step + 1)
        result_q.append(Q.tolist())

result_q = np.array(result_q).transpose([1, 2, 0])
for s in range(result_q.shape[0]):
    for a in range(result_q[s].shape[0]):
        if S1[s][a] == None: continue
        plt.plot(result_step, result_q[s][a],
                 label='Q(s{}, a{})'.format(s + 1, a + 1))
plt.legend(loc='lower right')
plt.show()
