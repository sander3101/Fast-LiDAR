# !/bin/bash

# Script automatically trains a model, samples all test data, and calculates evaluation metrics.
# It is recommended to use tmux or other tools for running the script even when idle from the computer to avoid timeouts

echo "Running training for model r2dm_only_pepper_and_upsample_continous"
export CUDA_VISIBLE_DEVICES="0,1"

accelerate launch train.py

if [ $? -eq 0 ]; then
    echo "Training of r2dm_only_pepper_and_upsample_continous was succesfull!"
else
    echo "Training of r2dm_only_pepper_and_upsample_continous has failed"
    exit 1
fi


num_steps_values=(1 2 4 8 16 32 64 128)



for num_steps in "${num_steps_values[@]}"; do
    echo "Running sampling with $num_steps timesteps"
    accelerate launch inpaint_and_save.py \
    --checkpoint logs/diffusion/kitti_360/spherical-1024/r2dm_only_pepper_and_upsample_continous/models/diffusion_0000300000.pth \
    --num_steps $num_steps \
    --output_folder sampling_results/quarter_sampling/only_pepper_and_upsampling/

    if [ $? -eq 0 ]; then
        echo "Sampling of $num_steps timesteps was succesfull!"
    else
        echo "Sampling with $num_steps timesteps has failed"
        exit 1
    fi
done


for num_steps in "${num_steps_values[@]}"; do
    echo "Running evaluation with $num_steps timesteps"
    python evaluate_inpaint.py \
    --result_dir sampling_results/quarter_sampling/only_pepper_and_upsampling/densification_results_resamplesteps_$num_steps/

    if [ $? -eq 0 ]; then
    
        echo "Evaluation of $num_steps timesteps was succesfull!"
    else
        echo "Evaluation with $num_steps timesteps has failed"
        exit 1
    fi
done
