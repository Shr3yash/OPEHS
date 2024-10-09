import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers
import random

class EnergyHarvestingEnv(gym.Env):
    def __init__(self):
        super(EnergyHarvestingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.state = None
        self.steps_taken = 0

    def reset(self):
        self.state = np.random.randint(1, 100)
        self.steps_taken = 0
        return self.state

    def step(self, action):
        reward = 0
        if action == 0:  # Action: Optimize energy
            reward = self.state * 0.1
        elif action == 1:  # Action: Ignore energy
            reward = -1
        elif action == 2:  # Action: Over-harvest
            reward = -self.state * 0.2

        self.steps_taken += 1
        done = self.steps_taken >= 10
        self.state = np.clip(self.state + np.random.randint(-10, 10), 0, 100)
        return self.state, reward, done, {}

class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=1, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PPOAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=1, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='softmax'))
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, 1])
        probabilities = self.model.predict(state).flatten()
        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape:
            probabilities = self.model(states)
            indices = tf.range(len(actions))
            selected_probabilities = tf.gather(probabilities, actions, batch_dims=1)
            old_probabilities = tf.gather(probabilities, actions, batch_dims=1)
            ratios = selected_probabilities / (old_probabilities + 1e-10)
            loss = -tf.reduce_mean(ratios * advantages)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

env = EnergyHarvestingEnv()
dqn_agent = DQNAgent(action_size=env.action_space.n)
ppo_agent = PPOAgent(action_size=env.action_space.n)

episodes = 100
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = dqn_agent.act(np.reshape(state, [1, 1]))
        next_state, reward, done, _ = env.step(action)
        dqn_agent.remember(np.reshape(state, [1, 1]), action, reward, np.reshape(next_state, [1, 1]), done)
        state = next_state
    if len(dqn_agent.memory) > 32:
        dqn_agent.replay(32)

for episode in range(episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action = ppo_agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    ppo_agent.train(np.array(states), np.array(actions), np.array(rewards), np.array(rewards))
