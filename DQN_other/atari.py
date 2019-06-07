import gym

class AtariWrapper:
    def __init__(self, env, render=False):
        self.env = env
        self.render = render
        self.queue = get_deque()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.sum_of_rewards = 0.0
        # to restart episode when life is lost
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        processed = preprocess(obs)
        self.queue.append(processed)
        if self.render:
            self.env.render()

        # for episodic life
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives

        self.sum_of_rewards += reward
        if done:
            info['reward'] = self.sum_of_rewards
        return np.array(list(self.queue)), reward, done, info

    def reset(self):
        # for episodic life
        if self.was_real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()

        self.queue = get_deque()
        processed = preprocess(obs)
        self.queue.append(processed)

        self.sum_of_rewards = 0.0
        return np.array(list(self.queue))
#-----------------------------------------------------------------------------#
