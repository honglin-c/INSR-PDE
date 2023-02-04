# INSR-PDE

Draw nn weights trajectory, e.g.,
```bash
python get_nn_weights.py -i checkpoints/advect1D_ex1/model -k net_field -o nn_weights_advect1D
```

Testing nn weights extrapolation on advection 1D, e.g.,
```bash
python extrapolate_weights_advect.py advection --tag advect1D_ex1 -g 0
```

Testing nn weights extrapolation on fluid 2D, e.g.,
```bash
python extrapolate_weights_fluid.py fluid --tag fluid2d_tlgnM -g 0
```
