#!/bin/bash
# How to use in bash: 1. chmod +x run_all.sh 2. ./run_all.sh
# device: A800 80G

# Group 1
echo "Running Group 1"
python results_full_history/Grad_Dependent_Nonlinear/60d/experiment_run.py &
python results/Grad_Dependent_Nonlinear/60d/experiment_run.py &
wait
echo "Group 1 completed."

# Group 2
echo "Running Group 2"
python results/Grad_Dependent_Nonlinear/80d/experiment_run.py &
python results/Grad_Dependent_Nonlinear/20d/experiment_run.py &
wait
echo "Group 2 completed."

# Group 3
echo "Running Group 3"
python results_full_history/Grad_Dependent_Nonlinear/80d/experiment_run.py &
python results_full_history/Grad_Dependent_Nonlinear/40d/experiment_run.py &
wait
echo "Group 3 completed."

# Group 4
echo "Running Group 4"
python results/Grad_Dependent_Nonlinear/40d/experiment_run.py &
python results_full_history/Grad_Dependent_Nonlinear/20d/experiment_run.py &
wait
echo "Group 4 completed."