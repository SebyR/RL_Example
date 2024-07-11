import numpy as np
import pickle
from h_env import hEnv

# Define the number of episodes and steps per episode
num_episodes = 150

# Initialize environment
env = hEnv(use_controll=True, render=True)

# Initialize dataset

observations = []
actions = []
rewards = []
terminals = []

done = False
# Collect data
for episode in range(num_episodes):
    done = False
    obs, _ = env.reset()
    while not done:
        # Random action for data collection (you can use a policy if you have one)
            action = env.controll()

            # Step the environment
            next_obs, reward, done, _, _ = env.step(action)

            # Append to dataset
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)

            obs = next_obs
            # print(np.array(obs).shape)

data_file = 'collected_data.pkl'

# Save the collected data
with open(data_file, 'wb') as f:
    data_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals
    }
    pickle.dump(data_dict, f)
