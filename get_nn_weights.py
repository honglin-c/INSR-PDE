# python calculate_model_size.py -i neuralAdvect/checkpoints/neuAdv_ex1hf20sr1024/model/ckpt_step_t030.pth -k field_state_dict
import os
import random
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help="path to saved pickle file")
parser.add_argument('-k', '--key', type=str, required=True, help="key in saved pickle")
parser.add_argument('-o', '--output', type=str, required=True, help="path to save")
args = parser.parse_args()

# load_path = args.input
all_paths = sorted(os.listdir(args.input))
# all_paths = [x for x in all_paths if 'projection' in x]
max_t = min(100, len(all_paths))
all_paths = all_paths[:max_t]
all_paths = [os.path.join(args.input, x) for x in all_paths]

all_weights_dict = {}
for i, path in enumerate(all_paths):
    # print(path)

    checkpoint = torch.load(path)
    state_dict = checkpoint[args.key]

    for k, v in state_dict.items():
        if k not in all_weights_dict:
            all_weights_dict[k] = []
        all_weights_dict[k].append(v.numpy())

for k, v in all_weights_dict.items():
    all_weights_dict[k] = np.stack(v, axis=0)


out_dir = args.output
os.makedirs(out_dir, exist_ok=True)

save_path = os.path.join(out_dir, "nn_weights_list.npz")
np.savez(save_path, *all_weights_dict)

N_DRAW = 20
NUM_COLORS = 20
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('gist_rainbow')

for k, v in all_weights_dict.items():
    print(k, v.shape)

    v = v.reshape(v.shape[0], -1)

    indices = sorted(random.sample(list(range(v.shape[-1])), k=min(N_DRAW, v.shape[-1])))
    x = list(range(v.shape[0]))

    fig, ax = plt.subplots()
    for i, idx in enumerate(indices):
        lines = ax.plot(x, v[:, idx])
        lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
    fig.tight_layout()

    save_path = os.path.join(out_dir, f"{k}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
