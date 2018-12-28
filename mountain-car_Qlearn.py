
import gym
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
#
#
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im = [ axs[0].imshow, axs[1].imshow, axs[2].imshow ]
#
#
class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        # Q-Table
        self.plim = [-1.2, 0.6]
        self.vlim = [-0.07, 0.07]
        self.N_position = 21
        self.N_velocity = 21
        self.positions = np.linspace(self.plim[0], self.plim[1], num=self.N_position, endpoint=True)
        self.velocities = np.linspace(self.vlim[0], self.vlim[1], num=self.N_velocity, endpoint=True)
        self.Q = np.zeros((len(self.actions),self.N_position, self.N_velocity))
        for a in range(len(self.actions)):
            for s_1 in range(len(self.positions)):
                for s_2 in range(len(self.velocities)):
                    self.Q[a,s_1,s_2] = -float('inf')

    def getQ(self, state, action):
        pos = np.argmin(abs(self.positions  - state[0]), axis=0)
        vel = np.argmin(abs(self.velocities - state[1]), axis=0)
        return self.Q[action,pos,vel]

    def learnQ(self, state1, action1, reward, state2):
        # Q-learning: Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        pos = np.argmin(abs(self.positions  - state1[0]), axis=0)
        vel = np.argmin(abs(self.velocities - state1[1]), axis=0)
        Q_n = self.Q[action,pos,vel]
        maxQ = max([self.getQ(state2, a) for a in self.actions])
        if Q_n == -float('inf') or maxQ == -float('inf'):
            self.Q[action, pos, vel] = reward
        else:
            self.Q[action, pos, vel] = Q_n + self.alpha * (reward + self.gamma*maxQ - Q_n)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            # In case there're several state-action max values
            # policy select a random one of them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]
        return action
    #
    def exportQ(self,fname):
        print '--------------------------------------------------'
        print self.Q
        print '--------------------------------------------------'
        print("Export Q-table to {}".format(fname))
        np.save(fname, self.Q)
    #
    def importQ(self,fname):
        print("Import Q-table from {}".format(fname))
        self.Q = np.load(fname + '.npy')
        print '--------------------------------------------------'
        print self.Q
        print '--------------------------------------------------'
    #
    def printQ(self):
        print '--------------------------------------------------'
        print self.Q
    #
    def plotQ(self):
        vmax = -1. #np.max(self.Q[0, :, :])
        vmin = -50.
        im[0] = axs[0].imshow(self.Q[0, :, :],interpolation='nearest',vmax=vmax, vmin=vmin,cmap=cm.jet)
        axs[0].grid(True)
        axs[0].set_title('Action: push_left')
        axs[0].set_ylabel('Position')
        axs[0].set_xlabel('Velocity')
        start, end = axs[0].get_xlim()
        fig.colorbar(im[0], ax=axs[0])

        im[1] = axs[1].imshow(self.Q[1, :, :],interpolation='nearest',vmax=vmax, vmin=vmin,cmap=cm.jet)
        axs[1].grid(True)
        axs[1].set_title('Action: no_push')
        axs[1].set_ylabel('Position')
        axs[1].set_xlabel('Velocity')
        fig.colorbar(im[1], ax=axs[1])

        im[2] = axs[2].imshow(self.Q[2, :, :],interpolation='nearest',vmax=vmax, vmin=vmin,cmap=cm.jet)
        axs[2].grid(True)
        axs[2].set_title('Action: push_right')
        axs[2].set_ylabel('Position')
        axs[2].set_xlabel('Velocity')
        fig.colorbar(im[2], ax=axs[2])

        plt.show(block=False)

    def plotQupdate(self):
        im[0].set_data(self.Q[0, :, :])
        im[1].set_data(self.Q[1, :, :])
        im[2].set_data(self.Q[2, :, :])
        fig.canvas.draw()
#
#
if __name__ == "__main__":
    # ----------------------------------------
    # Define parameters for greedy policy
    epsilon = 0.2   # exploration
    # Define parameters for Q-learning
    alpha = 0.2
    gamma = 0.98
    epoch = 10
    max_steps = 1000
    # ----------------------------------------
    # Actions
    # Type: Discrete(3)
    # Num | Observation
    # 0   | push_left
    # 1   | no_push
    # 2   | push_right
    N_action = 3
    actions = [0,1,2]
    # ----------------------------------------
    # Observation
    # Type: Box(2)
    # Num | Observation | Min   | Max
    # 0   | position    | -1.2  | 0.6
    # 1   | velocity    | -0.07 | 0.07
    N_input = 2
    observation = []
    # ----------------------------------------
    # Define environment/game
    env = gym.make('MountainCar-v0')
    # ----------------------------------------
    # Initialize QLearn object
    AI = QLearn(actions,epsilon=epsilon,alpha=alpha, gamma=gamma)
    # Load pre-trained model
    #AI.importQ('Qtable32121_11000epoch')
    AI.plotQ()
    # ----------------------------------------
    # Train
    for e in range(epoch):
        # Get initial input
        observation = env.reset()

        # Training for single episode
        step = 0
        reward = -1
        game_over = False
        while (not game_over):
            observation_capture = observation
            env.render()

            # Epsilon-Greedy policy
            action = AI.chooseAction(observation)

            # Apply action, get rewards and new state
            observation, reward, done, info = env.step(action)
            if observation[0] >= 0.6:
                reward = 1

            # Refinement of model
            AI.learnQ(observation_capture, action, reward, observation)

            step += 1
            if (step >= max_steps or observation[0] >= 0.6) and done:
                game_over = True
        #
        if reward > 0:
            print("Episode:{} finished after {} timesteps. Reached GOAL!.".format(e,step))
        else:
            print("Episode:{} finished after {} timesteps.".format(e,step))
        #
        AI.plotQupdate()

    # ----------------------------------------
    # Export Q table
    #AI.exportQ('Qtable32121_11000epoch')
    # Some delay
    plt.pause(5)
