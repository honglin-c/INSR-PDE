import os
from config import Config

# create experiment config containing all hyperparameters
cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)

# start time integration
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        model.initialize()
    else:
        model.step()

    model.write_output(output_folder)

# save time records
with open(os.path.join(cfg.exp_dir, "time_records.txt"), "w") as f:
    for record in model.time_records:
        print(record, file=f)

with open(os.path.join(cfg.exp_dir, "iter_records.txt"), "w") as f:
    for record in model.iter_records:
        print(record, file=f)
