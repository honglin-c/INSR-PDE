python main.py \
    --exp_name elasticity2D_collide_plane_2e1 \
    --num_hidden_layers 3 \
    --hidden_features 68 \
    -sr 100 \
    -vr 5 \
    -vr_time 11 \
    --t_range 4.0 \
    -g 1 \
    --max_n_iter 100000 \
    --early_stop \
    --lr 2e-4 \
    --enable_collision_plane 1 \
    --gravity_g ' -1.0' \
    --ratio_collision 2e1 \
    --ratio_arap 1e-1 \
    --ratio_volume 0e0 \
    --ratio_main 1e-1 \
    --plane_height ' -2.0' \
    --vis_frequency 1000