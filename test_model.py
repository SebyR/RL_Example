import pickle
import numpy as np
from d3rlpy.dataset import MDPDataset
from h_env import hEnv
import time
import d3rlpy


env = hEnv(use_controll=False, render=True)
model = d3rlpy.algos.CQLConfig(observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(), batch_size=32).create(device="cuda:0")

model.build_with_env(env)

model.load_model('rl_model.pt')

num_episodes = 50

done = False
# Collect data
for episode in range(num_episodes):
    done = False
    obs, _ = env.reset()
    while not done:
        # Random action for data collection (you can use a policy if you have one)
            obs = np.reshape(obs, [1, 3, 64, 64])
            action = model.predict(obs)
            print(action)
            # Step the environment
            next_obs, reward, done, _, _ = env.step(action.flatten())

            obs = next_obs