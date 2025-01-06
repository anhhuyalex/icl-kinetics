#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --output l2l-%J.log
#SBATCH -o slurms/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
source icl/bin/activate
mkdir -p cache # for storing data
  
# figure 2b, 2c
for i in {0..10}; do python -u main_icl.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10  --arch causal_transformer_embed --is_layer_norm True  --num_hidden_features 512 --is_equalize_classes False --len_context 100 --burstiness 1 --L 2 --epsilon 0.0 --fileprefix transformer1layer_lr_0.01_no_posenc --wandb_log --wandb_project l2l --experiment_name fig2bc ; done 
# figure 2d
for i in {0..10}; do python -u main_icl.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10  --arch causal_transformer_embed --is_layer_norm True  --num_hidden_features 512 --is_equalize_classes False --len_context 100 --burstiness 1 --L 2 --epsilon 0.0 --fileprefix transformer1layer_lr_0.01_no_posenc --wandb_log --wandb_project l2l --experiment_name fig2d ; done 
# figure 2e
for i in {0..10}; do python -u main_icl.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10  --arch causal_transformer_embed --is_layer_norm True  --num_hidden_features 512 --is_equalize_classes False --len_context 100 --burstiness 1 --L 2 --epsilon 0.0 --fileprefix transformer1layer_lr_0.01_no_posenc --wandb_log --wandb_project l2l --experiment_name fig2e ; done 


# figure A2, A3
for i in {0..10}; do python -u mlp.py --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr 0.01 --wd 1e-10 --arch mlp --num_hidden_features 512 --L 2 --epsilon 0.0 --fileprefix mlp --no-wandb_log --wandb_project l2l --experiment_name fig5 ; done

# 12-2,120-20,1200-200,12000-2000
# for i in {0..10}; do WANDB_MODE=offline ./l2l/bin/python3 -u mlp.py --data ./cache --fileprefix transformer1layer_lr_$1_no_posenc --no-position_encoding --SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID --batch-size 128 --optimizer SGD --lr $1 --epsilon 0.0 --L 2 --epochs 100 --arch mlp --wandb_log --wandb_project l2l --is_data_continuous_sampled True --wandb_group_name gmm_mat17_mlp_vs_k_nolayernorm ; done
# WANDB_MODE=offline ./l2l/bin/python3 -u gmm_transformer.py --data ./cache --fileprefix transformer1layer_lr_no_posenc --no-position_encoding --SLURM_ARRAY_TASK_ID 10 --batch-size 128 --optimizer SGD --lr 0.01 --epsilon 0.1 --burstiness 1 --K 120 --L 10 --epochs 100 --arch m2causal_transformer --temperature 100 --len_context 100 --wandb_log --wandb_project l2l --wandb_group_name gmm_mar8_0322am_fig10_m2causal_transformer_bursty_1_temp100