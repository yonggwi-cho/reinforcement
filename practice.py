"""
Created on 4th Aug. 2017

@author: Fukuyama Masaharu
"""

import numpy as np
import random
import sys

# Reward matrix

R = np.array([

#[-1,-1,-1,-1,0,-1],
[-1,-1,0,-1,0,-1],
#[-1,-1,-1,0,-1,100],
[-1,-1,-1,0,-1,100],
[-1,-1,-1,0,-1,-1],
[-1,0,0,-1,0,-1],
#[-1,0,300,-1,0,-1],
[0,-1,-1,0,-1,100],
[-1,0,-1,-1,0,100],

])

# Initial Q-value
Q = np.zeros((6,6))

LEARNING_COUNT = 1000
GAMMA = 0.8
GOAL_STATE = 5

class QLearning(object):
    def __init__(self):
        return

    def learn(self):
        # set an initial state randomly
        state = self._getRandomState()
        for i in range(LEARNING_COUNT):
            #extract possible actions in state
            possible_actions = self._getPossibleActionsFromState(state)

            #choose an action from possible actions randomly
            action = random.choice(possible_actions)

            #update Q value
            #Q(s,a) = r(s,a) + gamma * max[Q(next_s, possible_actions)]
            next_state = action # action value is the same as next state in this example.
            next_possible_actions = self._getPossibleActionsFromState(next_state)
            max_Q_next_s_a = self._getMaxQvalueFromStateAndPossibleActions(next_state,next_possible_actions)
            Q[state,action] = R[state, action] + GAMMA * max_Q_next_s_a

            state = next_state

            # If an agent reaches a goal state, restart an episode randomly
            if state == GOAL_STATE:
                state = self._getRandomState()

    def _getRandomState(self):
        return random.randint(0,R.shape[0] - 1)

    def _getPossibleActionsFromState(self,state):
        if state < 0 or state >= R.shape[0]: sys.exit("invalid state: %d" % state)
        return list(np.where(np.array(R[state] != -1)))[0]

    def _getMaxQvalueFromStateAndPossibleActions(self,state,possible_actions):
        return max([Q[state][i] for i in (possible_actions)])

    def dumpQvalue(self):
        #print Q.astype(int)
        print(Q)

    def runGreedy(self,start_state = 0):
        print ("===== START =====")
        state = start_state
        while state != GOAL_STATE:
            print ("current state : %d" % state)
            possible_actions = self._getPossibleActionsFromState(state)

            # get best action which maximizes Q value (=  Q(s,a))
            max_Q = 0
            best_action_candidates = []
            for a in possible_actions:
                if Q[state][a] > max_Q:
                    best_action_candidates = [a,]
                    max_Q = Q[state][a]
                elif Q[state][a] == max_Q:
                    best_action_candidates.append(a)

            # get a best action from candidates randomly
            best_action = random.choice(best_action_candidates)
            print ("-> choose action : %d" % best_action)
            state = best_action

        print ("state is %d,GOAL!" % state)


if __name__ == "__main__":
    QL = QLearning()
    QL.learn()
    QL.dumpQvalue()

    for s in range(R.shape[0]-1):
        QL.runGreedy(s)
