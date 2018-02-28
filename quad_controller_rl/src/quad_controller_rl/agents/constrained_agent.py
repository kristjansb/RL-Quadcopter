"""Constrained agent for controlling position only."""

import os
import numpy as np
import pandas as pd
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl import util

class ConstrainedAgent(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            'stats_{}'.format(util.get_timestamp())
        ) # path to CSV file
        self.stats_columns = ['episode', 'total_reward'] # stats to save
        self.episode_num = 1
        print('Saving stats {} to {}.'.format(self.stats_columns, self.stats_filename))

        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = 3
        self.action_size = 3
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size
        ))
        self.state_range = self.task.observation_space.high[0:3] - self.task.observation_space.low[0:3]
        self.action_range = self.task.action_space.high[0:3] - self.task.action_space.low[0:3]

        # Policy parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size)).reshape(1, -1))  # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions"""
        return state[0:3]
    
    def postprocess_action(self, action):
        """Return complete action vector"""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:3] = action
        return complete_action

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns) # single row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename)) # write header first time only

    def step(self, state, reward, done):
        # Reduce state vector
        state = self.preprocess_state(state)

        # Transform state vector
        state = (state - self.task.observation_space.low[0:3]) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector

        # Choose an action
        action = self.act(state)
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            self.learn()
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        # Return complete action vector
        return self.postprocess_action(action)

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        #print(action)  # [debug: action vector]
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        print("RandomPolicySearch.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                self.count, score, self.best_score, self.noise_scale))  # [debug]
        #print(self.w)  # [debug: policy parameters]
