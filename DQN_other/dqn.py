import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.solvers as S
import random
import argparse
import gym
import os

from datetime import datetime
from collections import deque
from nnabla.monitor import Monitor, MonitorSeries
from nnabla.ext_utils import get_extension_context

#------------------------------- neural network ------------------------------#
def dnn_network(obs, num_actions, scope):
    n = 50 # number of neuron
    # input layer
    with nn.parameter_scope(scope):
        with nn.parameter_scope("layer1"):
            h = PF.affine(obs,n)
            h = F.tanh(h)
            with nn.parameter_scope("layer2"):
                h = PF.affine(h,num_actions)
        return h

class Network:
    def __init__(self, num_actions, Nstate, batch_size, gamma, lr):
        # infer variable
        self.infer_obs_t = infer_obs_t = nn.Variable((1, Nstate))
        # train variables
        self.obs_t = obs_t = nn.Variable((batch_size, Nstate))
        self.actions_t = actions_t = nn.Variable((batch_size,))
        self.rewards_tp1 = rewards_tp1 = nn.Variable((batch_size,))
        self.obs_tp1 = obs_tp1 = nn.Variable((batch_size, Nstate))
        self.dones_tp1 = dones_tp1 = nn.Variable((batch_size,))

        # inference output
        self.infer_q_t = dnn_network(infer_obs_t, num_actions, scope='q_func')

        # training output
        q_t = dnn_network(obs_t, num_actions, scope='q_func')
        q_tp1 = dnn_network(obs_tp1, num_actions, scope='target_q_func')

        # select one dimension
        a_one_hot = F.one_hot(actions_t.reshape((-1, 1)), (num_actions,))
        q_t_selected = F.sum(q_t * a_one_hot, axis=1, keepdims=True)
        q_tp1_best = F.max(q_tp1, axis=1, keepdims=True)

        # shape transformation
        expanded_r_tp1 = rewards_tp1.reshape((-1, 1))
        expanded_d_tp1 = dones_tp1.reshape((-1, 1))

        # loss calculation
        target = expanded_r_tp1 + gamma * q_tp1_best * (1.0 - expanded_d_tp1)
        self.loss = F.mean(F.huber_loss(q_t_selected, target))

        # optimizer
        self.solver = S.RMSprop(lr, 0.95, 1e-2)

        # weights and biases
        with nn.parameter_scope('q_func'):
            self.params = nn.get_parameters()
        with nn.parameter_scope('target_q_func'):
            self.target_params = nn.get_parameters()

        # set q function parameters to solver
        self.solver.set_parameters(self.params)

    def infer(self, obs_t):
        self.infer_obs_t.d = obs_t
        self.infer_q_t.forward(clear_buffer=True)
        return self.infer_q_t.d

    def train(self, obs_t, actions_t, rewards_tp1, obs_tp1, dones_tp1):
        self.obs_t.d = obs_t
        self.actions_t.d = actions_t
        self.rewards_tp1.d = rewards_tp1
        self.obs_tp1.d = obs_tp1
        self.dones_tp1.d = dones_tp1

        self.loss.forward()
        self.solver.zero_grad()
        self.loss.backward(clear_buffer=True)

        for name, variable in self.params.items():
            # gradient clipping by norm
            grad = 10.0 * variable.grad / np.sqrt(np.sum(variable.g ** 2))
            variable.grad = grad

        self.solver.update()

    def update_target(self):
        for key in self.target_params.keys():
            self.target_params[key].data.copy_from(self.params[key].data)

    def save(self, path):
        nn.save_parameters(path)

    def load(self, path):
        nn.load_parameters(path)
#-----------------------------------------------------------------------------#


#---------------------------- replay buffer ----------------------------------#
class Buffer:
    def __init__(self, maxlen=10 ** 5, batch_size=32):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=maxlen)

    def add(self, obs_t, action_t, reward_tp1, obs_tp1, done_tp1):
        done = 1.0 if done_tp1 else 0.0
        experience = dict(obs_t=obs_t, action_t=action_t,
                          reward_tp1=reward_tp1, obs_tp1=obs_tp1,
                          done_tp1=done_tp1)
        self.buffer.append(experience)

    def sample(self):
        experiences = random.sample(self.buffer, self.batch_size)
        obs_t = []
        actions_t = []
        rewards_tp1 = []
        obs_tp1 = []
        dones_tp1 = []
        for experience in experiences:
            obs_t.append(experience['obs_t'])
            actions_t.append(experience['action_t'])
            rewards_tp1.append(experience['reward_tp1'])
            obs_tp1.append(experience['obs_tp1'])
            dones_tp1.append(experience['done_tp1'])
        return {
            'obs_t': obs_t,
            'actions_t': actions_t,
            'rewards_tp1': rewards_tp1,
            'obs_tp1': obs_tp1,
            'dones_tp1': dones_tp1
        }
#-----------------------------------------------------------------------------#


#----------------------- epsilon-greedy exploration --------------------------#
class EpsilonGreedy:
    def __init__(self, num_actions, init_value, final_value, duration):
        self.num_actions = num_actions
        self.base= init_value - final_value
        self.init_value = init_value
        self.final_value = final_value
        self.duration = duration

    def get(self, t, greedy_action):
        decay = t / self.duration
        if decay > 1.0:
            decay = 1.0
        epsilon = (1.0 - decay) * self.base + self.final_value
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        return greedy_action
#-----------------------------------------------------------------------------#

#-------------------------- training loop ------------------------------------#
def train(env, network, buffer, exploration, logdir):
    monitor = Monitor(logdir)
    reward_monitor = MonitorSeries('reward', monitor, interval=1)

    step = 0
    while step <= 10 ** 7:
        obs_t = env.reset()
        reward_t = 0.0
        done_tp1 = False

        while not done_tp1:
            # infer q values
            q_t = network.infer(obs_t)[0]

            # epsilon-greedy exploration
            action_t = exploration.get(step, np.argmax(q_t))

            # move environment
            obs_tp1, reward_tp1, done_tp1, info_tp1 = env.step(action_t)
            reward_t += reward_tp1

            # clip reward between [-1.0, 1.0]
            #clipped_reward_tp1 = np.clip(reward_tp1, -1.0, 1.0)

            # store transition
            buffer.add(obs_t, action_t, reward_tp1, obs_tp1, done_tp1)

            # update parameters
            if step > 10000 and step % 4 == 0:
                batch = buffer.sample()
                network.train(
                    batch['obs_t'], batch['actions_t'],
                    batch['rewards_tp1'], batch['obs_tp1'],
                    batch['dones_tp1'])

            # synchronize target parameters with the latest parameters
            if step % 10000 == 0:
                network.update_target()

            # save parameters
            if step % 10 ** 6 == 0:
                network.save(os.path.join(logdir, 'model_{}.h5'.format(step)))

            step += 1
            obs_t = obs_tp1

        # record metrics
        reward_monitor.add(step, reward_t)
#-----------------------------------------------------------------------------#


def main(args):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id=str(args.device))
        nn.set_default_context(ctx)

    # environment
    env = gym.make(args.env)
    if args.render == 1 :
        env.render()
    num_actions = env.action_space.n
    Nstate = len(env.observation_space.high)

    # action-value function built with neural network
    network = Network(num_actions, Nstate,args.batch_size, args.gamma, args.lr)
    if args.load is not None:
        network.load(args.load)

    # replay buffer for experience replay
    buffer = Buffer(args.buffer_size, args.batch_size)

    # epsilon-greedy exploration
    exploration = EpsilonGreedy(num_actions, args.epsilon, 0.1,
                                args.schedule_duration)

    # prepare log directory
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    logdir = os.path.join('logs', args.logdir + '_' + date)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # start training loop
    train(env, network, buffer, exploration, logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="MountainCar-v0")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--buffer-size', type=int, default=10 ** 5)
    parser.add_argument('--epsilon', type=int, default=0.5)
    parser.add_argument('--schedule_duration', type=int, default=10 ** 6)
    parser.add_argument('--logdir', type=str, default='experiment')
    parser.add_argument('--load', type=str)
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--render',type=int,default=1)
    args = parser.parse_args()
    main(args)
