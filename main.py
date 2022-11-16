import os
from config import Config

# create experiment config containing all hyperparameters
cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

# initialize (t = 0)
model.initialize()

# start time integration
for t in range(1, cfg.n_timesteps + 1):
    print(f"time step: {t}")
    model.step()
