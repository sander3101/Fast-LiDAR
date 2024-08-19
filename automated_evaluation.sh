#!/bin/bash

num_steps_values=(1 2 4 8 16)

for num_steps in "${num_steps_values[@]}"; do
    echo "Running evaluation with $num_steps timesteps"
    python evaluate_inpaint.py \
    --target_dir sampling_results/quarter_sampling/upsampling_varied/densification_targets_resamplesteps_$num_steps/

    if [ $? -eq 0 ]; then
    echo "Evaluation of $num_steps timesteps was succesfull!"
    else
        echo "Evaluation with $num_steps timesteps has failed"
        exit 1
    fi
done