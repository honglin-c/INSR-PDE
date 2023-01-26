import argparse
import torch
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="path to saved checkpoint")
args = parser.parse_args()

checkpoint_folder = parser.input

# Opening JSON file
file_path = os.path.join(checkpoint_folder, f"config.json")
f = open(file_path)
# returns JSON object as  a dictionary
config = json.load(f)

dt = config['dt']  
vel = config['vel']  
dim = config['dim']
n_timesteps = config['n_timesteps']

v = vel * np.ones((dim, 1))

file_path = os.path.join(checkpoint_folder, 'results', f"t{0:03d}.npy.npz")
data = np.load(file_path)
x0 = data['arr_0']


for ti in range(1, n_timesteps):
  file_path = os.path.join(checkpoint_folder, 'results', f"t{ti:03d}.npy.npz")
  data = np.load(file_path)
  xi = data['arr_0']

  xi_gt = x0 + v