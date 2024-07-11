import pickle
import numpy as np
from d3rlpy.dataset import MDPDataset
from h_env import hEnv
import time
import d3rlpy

# # Load the dataset
data_file = 'collected_data.pkl'

# Load the collected data
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)
    observations = data_dict['observations']
    actions = data_dict['actions']
    rewards = data_dict['rewards']
    terminals = data_dict['terminals']

print(np.array(observations).shape)

# Initialize environment
env = hEnv(use_controll=False, render=False)


mdp_dataset = MDPDataset(observations=np.array(observations, dtype=np.uint8),
                         actions=np.array(actions,dtype=np.float32),
                         rewards=np.array(rewards,dtype=np.float32),
                         terminals=np.array(terminals,dtype=np.float32))


time.sleep(10)


model = d3rlpy.algos.CQLConfig(observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(), batch_size=32).create(device="cuda:0")

# # Pretrain the algorithm
model.build_with_dataset(mdp_dataset)

model.load_model('rl_model_pret.pt')

# model.fit(mdp_dataset, n_steps=10000)
#
# # # Save the pretrained model
# model.save_model('rl_model_pret.pt')

#model.fit(dataset= mdp_dataset,n_steps=200, n_steps_per_epoch=10)

model.fit_online(env, n_steps=100000, n_steps_per_epoch=100)

model.save_model('rl_model.pt')