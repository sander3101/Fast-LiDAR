#!/bin/bash

num_steps_values=(32 64)

for num_steps in "${num_steps_values[@]}"; do
    echo "Running sampling with $num_steps timesteps"
    accelerate launch inpaint_and_save.py \
    --checkpoint logs/diffusion/kitti_360/spherical-1024/r2dm_2_to_8_upsampling_continous/models/diffusion_0000300000.pth \
    --num_steps $num_steps \
    --output_folder sampling_results/quarter_sampling/upsampling_varied/

    if [ $? -eq 0 ]; then
    echo "Sampling of $num_steps timesteps was succesfull!"
    else
        echo "Sampling with $num_steps timesteps has failed"
        exit 1
    fi
done