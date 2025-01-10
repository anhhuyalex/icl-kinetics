# Differential learning kinetics govern the transition from memorization to generalization during in-context learning
Code for [Differential learning kinetics govern the transition from memorization to generalization during in-context learning](https://arxiv.org/abs/2412.00104)

1. Create a Python virtual environment using `./setup.sh`
2. Activate the environment: `source icl/bin/activate` 
3. To train models and generate data from figures, use `main.sh`.
     We use `slurm` to submit jobs. Use the command `sbatch --array=0-99 main.sh` to submit, say, 100 jobs in parallel.
     Modify the `experiment_name` variable in `main.sh` to select the experiment to generate data for, as they use different configurations. You can choose among ["fig2bc", "fig2d", "fig2e+6d", "fig6a+A7, "fig6b+A4", "figA5", "fig6c", "figA6", "fig6e", "figA8"]
     In some experiments we use `wandb` for tracking. Make sure you have setup your `wandb` environment.
     
4. To plot figures, run `python analysis.py --experiment_name=${experiment_name}` where experiment_name is chosen among ["fig2bc", "fig2d", "fig2e+6d", "fig6a+A7, "fig6b+A4", "figA5", "fig6c", "figA6", "fig6e", "figA8"]