import numpy as np
import random
from collections import namedtuple, deque, defaultdict
from keras import layers, models, optimizers
from keras import backend as K
from keras.initializers import RandomUniform
import copy


class DDPG:
    """Initialize an Agent object.

    Params
    ======
        env (obj): environment to use, for example an OpenAI gym environment
        env_type (str): 'openai' or 'copter'
        random_seed (int): random seed
        max_eps (int): maximum number of episodes to run
        max_steps (int): maximum number of steps to take per episode if there is no hardcoded episode terminator
        train_every (str or int): set to 'step', 'episode', or int to train after every step, episode, or n number of steps
        decay_noise (str): program to decay noise. set to False, 'exp', 'linear', or 'success'
    """

    def __init__(self, env, env_type, random_seed, max_eps, max_steps, train_every, decay_noise):
        self.env = env
        self.env_type = env_type  # Either 'openai' or 'copter'
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        self.seed = random_seed
        self.ep_num = 0  # Counter for episode number
        self.step_num = 1  # Counter for step number
        self.max_eps = max_eps  # Maximum number of episodes to run
        self.max_steps = max_steps  # Maximum number of steps per episode
        self.scale_state_vector = True  # True to apply min/max scaling on state vector
        self.record = defaultdict(list)  # Stores records of all states, actions, and rewards
        self.train_every = train_every  # Training frequency: 'step', 'episode', or int value (every n steps).
        self.decay_noise = decay_noise  # Set to False, 'exp', 'linear', or 'success' (default is False)

        # Actor (Policy) Model
        self.actor_local = Actor(state_size=self.state_size,
                                 action_size=self.action_size,
                                 action_low=self.action_low,
                                 action_high=self.action_high,
                                 seed=random_seed)

        self.actor_target = Actor(state_size=self.state_size,
                                  action_size=self.action_size,
                                  action_low=self.action_low,
                                  action_high=self.action_high,
                                  seed=random_seed)

        # Critic (Value) Model
        self.critic_local = Critic(state_size=self.state_size,
                                   action_size=self.action_size,
                                   seed=random_seed)

        self.critic_target = Critic(state_size=self.state_size,
                                    action_size=self.action_size,
                                    seed=random_seed)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2

        self.initial_noise_scale = 1
        self.noise_scale = self.initial_noise_scale
        self.noise_decay_rate = 0.99
        self.noise = OUNoise(self.action_size,
                             random_seed,
                             self.exploration_mu,
                             self.exploration_theta,
                             self.exploration_sigma)

        # Replay memory
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 128  # minibatch size
        self.sparse_reward_weight = False  # weighting factor for prioritized replay
        self.subsample_size = None  # size of subsample to use for prioritized replay
        self.memory = ReplayBuffer(self.buffer_size,
                                   self.batch_size,
                                   self.sparse_reward_weight,
                                   random_seed,
                                   self.subsample_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset(self, env, env_type, random_seed, max_eps, max_steps, train_every, decay_noise):
        """Reset agent to its initial state."""

        self.__init__(env, env_type, random_seed, max_eps, max_steps, train_every, decay_noise)

    def reset_episode(self):
        self.noise.reset()  # reset the OUNoise
        self.ep_num += 1  # increment the episode counter by 1
        self.step_num = 1  # reset the step counter to 1

    def increment_step(self):
        """Increment the step counter by 1."""
        self.step_num += 1

    def decay_noise_scale(self, done):
        """Decay the noise scale according to a specified program. This is called by the step method.
        Linear decay scales the action by p = ep_num/max_eps and the noise by 1 - p. Exponential decay
        reduces initial_noise_scale by noise_decay_rate^ep_num. The last option is to decay when there
        is a successful episode. Here a success is defined as the episode terminating before max_steps
        is reached (i.e., if the agent reaches a goal), so it doesn't work for say Pendulum, but works
        for MountainCar. But this can be easily adapted for other defs of success."""

        # Linear decay
        if self.decay_noise == 'linear':
            p = self.ep_num / self.max_eps
            self.noise_scale = 1 - p

        # Exponential decay
        if self.decay_noise == 'exp':
            self.noise_scale = self.initial_noise_scale * (self.noise_decay_rate ** self.ep_num)

        # Decay upon successful episode
        if self.decay_noise == 'success':
            if done and (self.step_num < self.max_steps):
                self.noise_scale *= self.noise_decay_rate

    def store_record(self, state, action, reward, done):
        """Store experience in record for analysis."""

        if self.env_type == 'openai':
            for i in range(len(state)):
                self.record['State ' + str(i)].append(state[i])
            for i in range(len(action)):
                self.record['Action ' + str(i)].append(action[i])
        elif self.env_type == 'copter':
            dims = ['x', 'y', 'z', 'phi', 'theta', 'psi']
            for i in range(len(dims)):
                self.record[dims[i]].append(state[i])
            for i in range(len(action)):
                self.record['Action ' + str(i)].append(action[i])
            for i in range(len(self.env.sim.v)):
                self.record['Lin_v ' + dims[i]].append(self.env.sim.v[i])
            for i in range(len(self.env.sim.angular_v)):
                self.record['Ang_v ' + str(i)].append(self.env.sim.angular_v[i])

        self.record['Reward'].append(reward)
        self.record['Done'].append(done)
        self.record['Noise scale'].append(self.noise_scale)
        self.record['Episode'].append(self.ep_num)
        self.record['Step'].append(self.step_num)

    def step(self, state, action, reward, next_state, done):
        # Save experience/reward in record for later analysis (do this before scaling state vector)
        self.store_record(state, action, reward, done)

        # Save experience/reward in Replay Buffer
        self.memory.add(state, action, reward, next_state, done)

        # Train depending on frequency desired (every ep, every step, every n steps)
        if self.train_every == 'episode':
            if done or (self.step_num == self.max_steps):
                # Learn, if enough samples are available in memory
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)
        elif self.train_every == 'step':
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
        elif type(self.train_every) == int:
            if self.step_num % self.train_every == 0:
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

        # Increment step count by 1
        self.increment_step()

        # Decay noise scale
        self.decay_noise_scale(done)

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""

        # Scale the state vector if desired.
        if self.scale_state_vector:
            state = self.scale_state(state)

        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        # Decay noise if desired (see decay_noise_scale method).
        if self.decay_noise == 'linear':
            action = action * (1 - self.noise_scale) + (self.noise_scale * self.noise.sample())
        else:
            action += self.noise_scale * self.noise.sample()

        return list(np.clip(action, self.action_low, self.action_high))

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = rewards + γ * critic_target(next_states, actor_target(next_states))
        """
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        if self.scale_state_vector:
            states = self.scale_state(states)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        if self.scale_state_vector:
            next_states = self.scale_state(next_states)

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def scale_state(self, state):
        """Min-max scaler for state vector."""

        mins = np.array([self.env.observation_space.low[i] for i in range(self.env.observation_space.shape[0])])
        maxes = np.array([self.env.observation_space.high[i] for i in range(self.env.observation_space.shape[0])])

        return np.divide(state - mins, maxes - mins)

    def train_agent(self, threshold=False):
        """Train agent in environment. Environment can be either 'copter' or 'openai'."""

        scores = []
        scores_deque = deque(maxlen=10)

        if threshold:
            trained = False
            while trained == False:
                for _ in range(1, self.max_eps + 1):
                    state = self.env.reset()  # start a new episode
                    self.reset_episode()
                    score = 0
                    while True:
                        action = self.act(state)
                        if self.env_type == 'copter':
                            next_state, reward, done = self.env.step(action)
                        elif self.env_type == 'openai':
                            next_state, reward, done, _ = self.env.step(action)
                        self.step(state, action, reward, next_state, done)
                        state = next_state
                        score += reward
                        if done:
                            scores.append(score)
                            scores_deque.append(score)
                            if self.env_type == 'copter':
                                print(
                                    "Episode {}/{}, Score = {:.1f}, Noise = {:.3f}, Time = {:.2f}, Final State: ({:.2f}, {:.2f}, {:.2f})"
                                    .format(self.ep_num, self.max_eps, score, self.noise_scale,
                                            self.env.sim.time, state[0], state[1], state[2]))
                            elif self.env_type == 'openai':
                                print("Episode {}/{}, Score = {:.1f}, Noise = {:.3f}"
                                      .format(self.ep_num, self.max_eps, score, self.noise_scale))
                            break

                    if self.ep_num % 10 == 0:
                        if np.mean(scores_deque) < threshold:
                            print('Agent failed to learn or got stuck in local minimum: Resetting networks and trying again.')
                            self.reset(self.env, self.env_type, self.seed, max_eps=self.max_eps,
                                       max_steps=self.max_steps,
                                       train_every=self.train_every, decay_noise=self.decay_noise)
                            break
                    if self.ep_num == self.max_eps:
                        print('Agent trained succesfully!')
                        trained = True
                        break

        else:
            for _ in range(1, self.max_eps + 1):
                state = self.env.reset()  # start a new episode
                self.reset_episode()
                score = 0
                while True:
                    action = self.act(state)
                    if self.env_type == 'copter':
                        next_state, reward, done = self.env.step(action)
                    elif self.env_type == 'openai':
                        next_state, reward, done, _ = self.env.step(action)
                    self.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        scores.append(score)
                        if self.env_type == 'copter':
                            print(
                                "Episode {}/{}, score = {:.1f}, Noise = {:.3f}, Time = {:.2f}, Final State: ({:.2f}, {:.2f}, {:.2f})"
                                .format(self.ep_num, self.max_eps, score, self.noise_scale,
                                        self.env.sim.time, state[0], state[1], state[2]))
                        elif self.env_type == 'openai':
                            print("Episode {}/{}, score = {:.1f}, Noise = {:.3f}"
                                  .format(self.ep_num, self.max_eps, score, self.noise_scale))
                        break

        return scores

    def watch_agent(self):
        """Watch a trained agent's performance in an environment."""

        self.ep_num = 0

        scores = []
        for _ in range(1, self.max_eps + 1):
            state = self.env.reset()  # start a new episode
            self.reset_episode()
            score = 0
            while True:
                action = self.act(state)
                if self.env_type == 'copter':
                    state, reward, done = self.env.step(action)
                elif self.env_type == 'openai':
                    state, reward, done, _ = self.env.step(action)
                    if self.ep_num % 10 == 0:
                        self.env.render()
                score += reward
                if done:
                    scores.append(score)
                    if self.env_type == 'copter':
                        print(
                            "Episode {}/{}, score = {:.1f}, Noise = {:.3f}, Time = {:.2f}, Final State: ({:.2f}, {:.2f}, {:.2f})"
                            .format(self.ep_num, self.max_eps, score, self.noise_scale,
                                    self.env.sim.time, state[0], state[1], state[2]))
                    elif self.env_type == 'openai':
                        print("Episode {}/{}, score = {:.1f}, Noise = {:.3f}"
                              .format(self.ep_num, self.max_eps, score, self.noise_scale))
                    break

        return scores


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, seed):
        """Initialize parameters and build model.

        Params
        ======
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
        self.seed = seed

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        kernel_init_hl1 = RandomUniform(minval=-1 / np.sqrt(400), maxval=1 / np.sqrt(400), seed=self.seed)
        net = layers.Dense(units=400, activation='relu', kernel_initializer=kernel_init_hl1)(states)
        kernel_init_hl2 = RandomUniform(minval=-1 / np.sqrt(300), maxval=1 / np.sqrt(300), seed=self.seed)
        net = layers.Dense(units=300, activation='relu', kernel_initializer=kernel_init_hl2)(net)
        kernel_init_hl3 = RandomUniform(minval=-1 / np.sqrt(200), maxval=1 / np.sqrt(200), seed=self.seed)
        net = layers.Dense(units=200, activation='relu', kernel_initializer=kernel_init_hl3)(net)

        #  Add final output layer with sigmoid activation
        kernel_init_out = RandomUniform(minval=-3e-3, maxval=3e-3, seed=self.seed)
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', kernel_initializer=kernel_init_out,
                                   name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=1e-3)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        kernel_init_hl1 = RandomUniform(minval=-1 / np.sqrt(400), maxval=1 / np.sqrt(400), seed=self.seed)
        net_states = layers.Dense(units=400, activation='relu', kernel_initializer=kernel_init_hl1)(states)

        # Combine state and action pathways
        state_action = layers.Concatenate()([net_states, actions])

        # Add hidden layer(s) for combined state/action pathway
        kernel_init_hl2 = RandomUniform(minval=-1 / np.sqrt(300), maxval=1 / np.sqrt(300), seed=self.seed)
        net = layers.Dense(units=300, activation='relu', kernel_initializer=kernel_init_hl2)(state_action)
        kernel_init_hl3 = RandomUniform(minval=-1 / np.sqrt(200), maxval=1 / np.sqrt(200), seed=self.seed)
        net = layers.Dense(units=200, activation='relu', kernel_initializer=kernel_init_hl3)(net)

        # Add final output layer to produce action values (Q values)
        kernel_init_out = RandomUniform(minval=-3e-3, maxval=3e-3, seed=self.seed)
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=kernel_init_out)(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=1e-2)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, sparse_reward_weight, seed, subsample_size=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size  # mini-batch size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.sparse_reward_weight = sparse_reward_weight
        self.seed = random.seed(seed)
        self.subsample_size = subsample_size  # subsample size for priority replay if replay buffer is large

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.sparse_reward_weight:
            if self.sparse_reward_weight >= 1:
                self.sparse_reward_weight = 1 - 1e-8
            # Use a subsample of the memory if the memory is large to avoid slow performance
            if self.subsample_size is not None:
                if self.subsample_size > len(self.memory):  # This will only be True at the very beginning.
                    sorted_mem = sorted(self.memory, key=lambda x: x.reward, reverse=True)
                else:
                    subsample = random.sample(self.memory, k=self.subsample_size)
                    sorted_mem = sorted(subsample, key=lambda x: x.reward, reverse=True)
                prob = np.array([self.sparse_reward_weight ** i for i in range(len(sorted_mem))])
                prob /= np.sum(prob)
                indices = np.random.choice(np.arange(len(self.memory)), replace=False, size=self.batch_size, p=prob)
                sample_batch = np.array(sorted_mem)[indices]
                sample_batch = [self.experience(state, action, reward, next_state, done) for
                                state, action, reward, next_state, done in sample_batch]
            else:
                sorted_mem = sorted(self.memory, key=lambda x: x.reward, reverse=True)
                prob = np.array([self.sparse_reward_weight ** i for i in range(len(sorted_mem))])
                prob /= np.sum(prob)
                indices = np.random.choice(np.arange(len(self.memory)), replace=False, size=self.batch_size, p=prob)
                sample_batch = np.array(sorted_mem)[indices]
                sample_batch = [self.experience(state, action, reward, next_state, done) for
                                state, action, reward, next_state, done in sample_batch]
            return sample_batch
        else:
            return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
