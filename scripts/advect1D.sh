# python main.py advection \
#     --tag icml_adv1D_base_5K \
#     --init_cond example1 \
#     --num_hidden_layers 2 \
#     --hidden_features 20 \
#     -sr 5000 \
#     --dt 0.05 \
#     -T 20 \
#     --max_n_iters 5000 \
#     -g 0 &

# python main.py advection \
#     --tag icml_adv1D_extra_10K \
#     --init_cond example1 \
#     --num_hidden_layers 2 \
#     --hidden_features 20 \
#     -sr 5000 \
#     --dt 0.05 \
#     -T 20 \
#     --max_n_iters 10000 \
#     -g 3 \
#     -we &

# python main.py advection \
#     --tag icml_adv1D_baseRP \
#     --init_cond example1 \
#     --num_hidden_layers 2 \
#     --hidden_features 20 \
#     -sr 5000 \
#     --dt 0.05 \
#     -T 20 \
#     -g 2 \
#     --optim_type plateau &

# python main.py advection \
#     --tag icml_adv1D_extraRP \
#     --init_cond example1 \
#     --num_hidden_layers 2 \
#     --hidden_features 20 \
#     -sr 5000 \
#     --dt 0.05 \
#     -T 20 \
#     -g 3 \
#     --optim_type plateau \
#     -we

python main.py advection \
    --tag icml_adv1D_base_large_lr1e-5 \
    --init_cond example1 \
    --num_hidden_layers 3 \
    --hidden_features 256 \
    -sr 5000 \
    --dt 0.05 \
    -T 20 \
    --lr 1e-5 \
    -g 2 &

python main.py advection \
    --tag icml_adv1D_base_small \
    --init_cond example1 \
    --num_hidden_layers 2 \
    --hidden_features 20 \
    -sr 5000 \
    --dt 0.05 \
    -T 20 \
    -g 3