python main.py \
    --exp_name elasticity2D_stretch_68_3 \
    --num_hidden_layers 3 \
    --hidden_features 68 \
    -sr 100 \
    -vr 5 \
    --t_range 4.0 \
    -g 1 \
    --max_n_iter 10000 \
    --early_stop \
    --lr 1e-4 \
    --gravity_g ' 0.0' \
    --enable_bound_top 1 \
    --enable_bound_bottom 1 \
    --ratio_arap 1e0 \
    --ratio_volume 1e0 \
    --vis_frequency 1000