"""Deep Deterministic Policy Gradient agent."""

from keras import layers, models, optimizers
from keras import backend as K
from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
from collections import namedtuple
import pandas as pd
import numpy as np
import os
import random

Experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])

class DDPG(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        # Print debug statements
        self.debug = True

        # Load / save parameters
        self.load_weights = False  # try to load weights from previously trained model
        self.save_weights_every = 10  # save weights every N episodes, None to disable
        self.model_dir = util.get_param('out')  # you can use a separate subdirectory for each task and/or neural net architecture
        # self.model_name = "takeoff-sim"
        # self.model_name = "hover-sim"
        self.model_name = "landing-sim"
        self.model_ext = ".h5"
        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir,
                "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir,
                "{}_critic{}".format(self.model_name, self.model_ext))
            if self.debug:
                print("Actor filename :", self.actor_filename)  # [debug]
                print("Critic filename:", self.critic_filename)  # [debug]
        
        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            'stats_{}_{}'.format(self.model_name, util.get_timestamp())
        )  # path to CSV file
        if self.model_name == "landing-sim":
            self.stats_columns = ['episode', 'total_reward', 
                                  'final_position_z', 'previous_position_z',
                                  'final_velocity_z', 'previous_velocity_z',
                                  'final_acceleration_z', 'previous_acceleration_z',
                                  'final_action', 'previous_action']  # stats to save
        else:
            self.stats_columns = ['episode', 'total_reward']  # stats to save
        self.print_progress_every = 5
        self.episode_num = 0
        if self.debug:
            print('Saving stats {} to {}.'.format(self.stats_columns, self.stats_filename))
        
        # Task (environment) information
        self.task = task
        if self.model_name == 'takeoff-sim':
            # Take off task state space
            self.state_size = 1  # take-off task z position
            self.state_low = self.task.observation_space.low[2]
            self.state_high = self.task.observation_space.high[2]
            self.action_size = 1  # take-off task z thrust
            self.action_low = self.task.action_space.low[2]
            self.action_high = self.task.action_space.high[2]
        if self.model_name == 'landing-sim':
            # Take off task state space
            self.state_size = 3  # take-off task z position
            self.state_low = np.array([self.task.observation_space.low[2],
                                       self.task.observation_space.low[9],
                                       self.task.observation_space.low[12]])
            self.state_high = np.array([self.task.observation_space.high[2],
                                        self.task.observation_space.high[9],
                                        self.task.observation_space.high[12]])
            self.action_size = 1  # take-off task z thrust
            self.action_low = self.task.action_space.low[2]
            self.action_high = self.task.action_space.high[2]
        else:
            self.state_size = np.prod(self.task.observation_space.shape)
            self.state_low = self.task.observation_space.low
            self.state_high = self.task.observation_space.high
            self.action_size = np.prod(self.task.action_space.shape)
            self.action_low = self.task.action_space.low
            self.action_high = self.task.action_space.high
        
        self.state_range = self.state_high - self.state_low     
        self.action_range = self.action_high - self.action_low

        # Actor (policy) model
        if self.model_name == "takeoff-sim":
            self.actor_lr = 1e-4
        elif self.model_name == "landing-sim":
            self.actor_lr = 1e-3
        else:
            self.actor_lr = 1e-4
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, 
            self.action_high, learning_rate=self.actor_lr)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, 
            self.action_high, learning_rate=self.actor_lr)

        # Critic (value) model
        if self.model_name == "takeoff-sim":
            self.critic_lr = 1e-5
        elif self.model_name == "landing-sim":
            self.critic_lr = 1e-5
        else:
            self.critic_lr = 1e-5
        self.critic_local = Critic(self.state_size, self.action_size, learning_rate=self.critic_lr)
        self.critic_target = Critic(self.state_size, self.action_size, learning_rate=self.critic_lr)

        # Print Actor / Critic NN architectures
        if self.debug:
            self.actor_local.model.summary()
            self.critic_local.model.summary()
        
        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                if self.debug:
                    print("Model weights loaded from file!")  # [debug]
            except Exception as e:
                if self.debug:
                    print("Unable to load model weights from file!")
                    print("{}: {}".format(e.__class__.__name, str(e)))
        
        if self.save_weights_every:
            if self.debug:
                print("Saving model weights ", "every {} episodes".format(
                    self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]
        
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size, theta=0.15, sigma=0.3)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 32
        # self.buffer_size = 1000000
        # self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # for soft update of target parameters

        # Score tracker and learning parameters
        self.best_score = -np.inf

        # Episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.last_native_state = None
        self.total_reward = 0.0
        self.count = 0
        self.episode_num += 1

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions"""
        if self.model_name == "takeoff-sim":
            return state[2]
        elif self.model_name == "hover-sim":
            return state
        elif self.model_name == "landing-sim":
            position_z = state[2]
            velocity_z = state[9]
            acceleration_z = state[12]
            return [position_z, velocity_z, acceleration_z]
        else:
            return state
    
    def postprocess_action(self, action):
        """Return complete action vector"""
        if self.model_name == "takeoff-sim":
            complete_action = np.zeros(self.task.action_space.shape)
            complete_action[2] = action
        if self.model_name == "hover-sim":
            complete_action = action
        if self.model_name == "landing-sim":
            complete_action = np.zeros(self.task.action_space.shape)
            complete_action[2] = action
        else:
            complete_action = action
        return complete_action
    
    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample() # Add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next state actions and Q values from target networks
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) # custom training function

        # Soft update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        # Learn by random policy search, using a reward-based score
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score
        # Print debug info to track progress
        if self.debug:
            if self.count % self.print_progress_every == 0:
                print("DDPG.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), total_reward = {:4.2f}".format(
                        self.count, score, self.best_score, self.total_reward))  # [debug]

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
    
    def step(self, state, reward, done):
        # Reduce state vector
        state = self.preprocess_state(state)
        native_state = state

        # Transform state vector
        state = (state - self.state_low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            # print('Step: {:4d}, Position: {}, Force: {}'.format(self.count, state[0][0:3], action[0][0:3]))  # debug
            self.count += 1

        # Learn if enough samples are in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        # Learn, if at end of episode
        if done:
            # Save model weights at regular intervals
            if self.save_weights_every and self.episode_num % self.save_weights_every == 0:
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                if self.debug:
                    print("Model weights saved at episode {}.".format(self.episode_num))  # [debug]
            # Write episode stats
            if self.model_name == "landing-sim":
                self.write_stats([self.episode_num, self.total_reward, 
                                  native_state[0], self.last_native_state[0],
                                  native_state[1], self.last_native_state[1],
                                  native_state[2], self.last_native_state[2],
                                  action[0][0], self.last_action[0][0]])
            else:
                self.write_stats([self.episode_num, self.total_reward])
            if self.debug:
                print("Final State: {}\nFinal Action: {}\nTotal Reward: {}".format(native_state, action, self.total_reward))
            self.reset_episode_vars()

        # Save off current action and state
        self.last_state = state
        self.last_action = action
        self.last_native_state = native_state

        # Return complete action vector
        return self.postprocess_action(action)

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns) # single row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename)) # write header first time only


class Actor:
    """Actor (Policy) model"""
    
    def __init__(self, state_size, action_size, action_low, action_high, learning_rate=1e-4):
        """Initialize parameters and build the model.

        Params
        ------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.lr = learning_rate

        # Initialize other variables here

        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build an actor (policy) model that maps states -> actions."""
        # Define input layer (state)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        if self.state_size == 1:  # Constrainted to Z position
            net = layers.Dense(units=4, activation='relu')(states)
            net = layers.Dense(units=8, activation='relu')(net)
            net = layers.Dense(units=8, activation='relu')(net)
            net = layers.Dense(units=4, activation='relu')(net)
        elif self.state_size == 3:  # Constrained to (X, Y, Z) position
            net = layers.Dense(units=16)(states)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = layers.Dense(units=32)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = layers.Dense(units=64)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
            net = layers.Dense(units=64)(net)
            net = layers.BatchNormalization()(net)
            net = layers.Activation('relu')(net)
        elif self.state_size == 10:  # Includes position, orientation, velocity
            net = layers.Dense(units=32, activation='relu')(states)
            net = layers.Dense(units=32, activation='relu')(net)
            net = layers.Dense(units=64, activation='relu')(net)
            net = layers.Dense(units=128, activation='relu')(net)
        else:
            net = layers.Dense(units=32, activation='relu')(states)
            net = layers.Dense(units=32, activation='relu')(net)
            net = layers.Dense(units=64, activation='relu')(net)
            net = layers.Dense(units=128, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', 
            name='raw_actions')(net)
        
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (value) model."""

    def __init__(self, state_size, action_size, learning_rate=1e-5):
        """Initialize parameters and build model.

        Params
        ------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate

        # Initialize any other variables here

        # Build the model
        self.build_model()
    
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        if self.state_size == 1:  # Constrained to Z position
            # Add hidden layers for state pathway
            net_states = layers.Dense(units=4, activation='relu')(states)
            net_states = layers.Dense(units=8, activation='relu')(net_states)

            # Add hidden layers for action pathway
            net_actions = layers.Dense(units=4, activation='relu')(actions)
            net_actions = layers.Dense(units=8, activation='relu')(net_actions)
        elif self.state_size == 3:  # Constrained to (X, Y, Z) position
            # Add hidden layers for state pathway
            net_states = layers.Dense(units=16, activation='relu')(states)
            net_states = layers.Dense(units=32, activation='relu')(net_states)

            # Add hidden layers for action pathway
            net_actions = layers.Dense(units=16, activation='relu')(actions)
            net_actions = layers.Dense(units=32, activation='relu')(net_actions)
        elif self.state_size == 10:  # Includes position, orientation, velocity
            # Add hidden layers for state pathway
            net_states = layers.Dense(units=32, activation='relu')(states)
            net_states = layers.Dense(units=64, activation='relu')(net_states)
            net_states = layers.Dense(units=128, activation='relu')(net_states)

            # Add hidden layers for action pathway
            net_actions = layers.Dense(units=32, activation='relu')(actions)
            net_actions = layers.Dense(units=64, activation='relu')(net_actions)
            net_actions = layers.Dense(units=128, activation='relu')(net_actions)
        else:
            # Add hidden layers for state pathway
            net_states = layers.Dense(units=32, activation='relu')(states)
            net_states = layers.Dense(units=64, activation='relu')(net_states)
            net_states = layers.Dense(units=128, activation='relu')(net_states)

            # Add hidden layers for action pathway
            net_actions = layers.Dense(units=32, activation='relu')(actions)
            net_actions = layers.Dense(units=64, activation='relu')(net_actions)
            net_actions = layers.Dense(units=128, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an addition function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
