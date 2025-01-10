#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import importlib

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import attention

import utils
import numpy as np
import random

import wandb 

parser = argparse.ArgumentParser(description='GMM L2L Training with Sequence Model')
parser.add_argument('--SLURM_ARRAY_TASK_ID', default=1, type=int,
                    help='SLURM_ARRAY_TASK_ID')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')

# Optimizer
parser.add_argument('--epochs', default=90, type=int,  
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')                         
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', default='SGD', type=str, 
                    choices = ['SGD', 'Adam'],
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp',
                    help='model architecture (default: mlp)')
parser.add_argument('--num_attn_layers', default=1, type=int,
                    help='number of transformer layers')
parser.add_argument('--temperature', default=None, type=float,
                    help='temperature for transformer')
parser.add_argument('--is_layer_norm', default="True", type=str,
                    help='whether to use layer norm in transformer')
parser.add_argument('--num_hidden_features', default=1, type=int,
                    help='num_hidden_features')

# Data
parser.add_argument('--num_tasks', default=1, type=int,
                    help='number of tasks')
parser.add_argument("--is_equalize_classes", default="False", type=str,
                    help='whether to equalize classes')
parser.add_argument('--len_context', default=1, type=int,
                    help='number of in-context images in sequence')
parser.add_argument('--burstiness', default=0, type=int,
                    help='burstiness of task sequence')
parser.add_argument('--K', default=2, type=int,
                    help='number of classes in GMM')
parser.add_argument('--L', default=2, type=int,
                    help='number of labels in GMM')
parser.add_argument('--epsilon', default=0.1, type=float,
                    help='epsilon for GMM')
parser.add_argument('--prob_mix_up', default=0.0, type=float,
                    help='probability of mix up')

parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')  
parser.add_argument('--experiment_name', default=None, type=str,
                    help='experiment name')
parser.add_argument(
            '--fileprefix',  default="", type=str, action='store') 
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="l2l",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_group_name",type=str,default="",
    help="wandb project name",
)

args = parser.parse_args()

# assert args.K % args.L == 0, "K must be divisible by L"
if args.seed is None:
    args.seed = np.random.randint(0, 10000000)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up hyperparameters for different experiments
if args.experiment_name == "fig2bc":
    Ks = np.linspace( (250),  (5e4), 10).astype(int) * 2
    args.K = Ks[args.SLURM_ARRAY_TASK_ID % len(Ks)] 
    args.weight_decay = 1e-10
    args.prob_mix_up = -1.0 
    args.len_context = 100
    args.epochs = 1000
    args.wandb_log = True
elif args.experiment_name == "fig2d": 
    args.K = 20000
    args.weight_decay = 1e-10 
    args.prob_mix_up = -1.0
    args.len_context = 100
    args.epochs = 1000
    args.wandb_log = True
elif args.experiment_name == "fig2e+6d": # Transience
    args.K = 10000
    args.weight_decay = 3e-3 # transient ICL dynamics require higher attention weight decay
    args.prob_mix_up = -1.0 
    args.len_context = 100
    args.epochs = 50000 # transient ICL dynamics require more epochs
    args.wandb_log = True
elif args.experiment_name == "fig6a+A7": # K^* vs. N
    Ns = [ 50, 100, 150, 200, 250 ]
    args.len_context = Ns [args.SLURM_ARRAY_TASK_ID % len(Ns)]
    Ks = np.logspace( np.log10(250),  np.log10(2e5), 30).astype(int) * 2 
    args.K = Ks[args.SLURM_ARRAY_TASK_ID // len(Ns)]    
    args.weight_decay = 1e-10
    args.prob_mix_up = -1.0 
    args.epochs = 2000 # long context lengths require more epochs
    args.wandb_log = False
elif args.experiment_name == "fig6b+A4": # 
    Ns = [ 50, 100, 150, 200, 250 ]
    args.len_context = Ns [args.SLURM_ARRAY_TASK_ID % len(Ns)]
    args.K = -1 # negative K means we take the limit K->infinity 
    args.weight_decay = 1e-10 
    args.prob_mix_up = -1.0
    args.epochs = 2000 # long context lengths require more epochs
    args.wandb_log = True
elif args.experiment_name == "figA5": 
    Ns = [ 50, 100, 150, 200, 250 ]
    args.len_context = Ns [args.SLURM_ARRAY_TASK_ID % len(Ns)]
    args.K = -1 # negative K means we take the limit K->infinity 
    args.weight_decay = 1e-10 
    args.prob_mix_up = -1.0
    args.epochs = 2000 # long context lengths require more epochs
    args.wandb_log = True
    args.arch = "multilayer_transformer"
    args.num_attn_layers = 2
elif args.experiment_name == "fig6c": # 
    args.len_context = 100
    args.K = 9666 
    args.weight_decay = 1e-10
    args.prob_mix_up = -1.0
    args.epochs = 1000  
    args.wandb_log = True
elif args.experiment_name == "figA6": # vary weight decay to look at transient ICL dynamics
    args.len_context = 100 
    args.K = 15000 
    weight_decay = np.logspace( -3, -2, 20) 
    args.weight_decay = weight_decay[args.SLURM_ARRAY_TASK_ID % len(weight_decay)]
    args.prob_mix_up = -1.0
    args.epochs = 10000 # transient ICL dynamics require more epochs
    args.wandb_log = True
elif args.experiment_name == "fig6e": # vary contextual statistics: equalize 0s and 1s vs. not
    args.len_context = 100
    Ks = np.logspace( np.log10(2500),  np.log10(25000), 10).astype(int) * 2  
    args.K = Ks[args.SLURM_ARRAY_TASK_ID % len(Ks)]
    args.weight_decay = 1e-10
    args.prob_mix_up = -1.0
    args.epochs = 1000
    args.wandb_log = True
    
    is_equalize_classes = ["True", "False"]
    args.is_equalize_classes = is_equalize_classes[args.SLURM_ARRAY_TASK_ID // len(Ks)]
elif args.experiment_name == "figA8": # vary contextual statistics: equalize 0s and 1s
    args.len_context = 100
    args.K = 31336
    args.weight_decay = 1e-10
    args.prob_mix_up = -1.0
    args.epochs = 1000
    args.wandb_log = True
    
    args.is_equalize_classes = "True"
else:
    raise ValueError(f"experiment_name {args.experiment_name} not recognized")


# set default values for some arguments
D = 63 # match omniglot input dim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local Rank for distributed training
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)
print("args:\n",vars(args))
# setup weights and biases (optional)
if local_rank==0 and args.wandb_log: # only use main process for wandb logging
    YOUR_WANDB_HOST = "https://api.wandb.ai" # change this to your wandb host
    wandb.login(host=YOUR_WANDB_HOST) # need to configure wandb environment beforehand
    wandb_model_name = f"{args.fileprefix}_K_{args.K}_num_tasks_{args.num_tasks}"
    wandb_config = vars(args)
    args.wandb_group_name = f"{args.experiment_name}"
    print("wandb_id:",wandb_model_name)
    wandb.init(
        project=args.wandb_project,
        name=wandb_model_name,
        config=wandb_config,
        resume="allow",
        group=args.wandb_group_name,
    )
    wandb.config.local_file_dir = wandb.run.dir 
else:
    record = {
        "args": vars(args),
        "logs": []
    }


class SequenceGMM(torch.utils.data.Dataset):
    def __init__(self, K, D, burstiness, epsilon = 0.1,
                 len_context = 1,
                 is_full_length = False,
                 is_equalize_classes = False,
                len_data = 60000,
                prob_mix_up = 0.0,
                 seed = None ):
        rng = np.random.default_rng(seed)
        
        # if K < 40000:
        self.len_context = len_context
        self.D = D
        if K < 0: # negative K means we take the limit K->infinity
            self.mu_K = torch.randn(self.len_context , self.D) * (1.0 / np.sqrt(self.D)) # shape: (K, D)
            self.K = K
        else:
            mu_K = rng.standard_normal((K, D)) * (1.0 / np.sqrt(D)) # shape: (K, D) 
            self.mu_K = torch.tensor(mu_K)
            self.K = K 
        # self.mu_K = torch.randn(K, D) * (1.0 / np.sqrt(D)) # shape: (K, D) 
        self.burstiness = burstiness
        
        
        self.epsilon = epsilon
        self.prob_mix_up = prob_mix_up
        
        self.len_data = len_data
        self.is_full_length = is_full_length 
        if is_equalize_classes=="True":
            K_classes = np.random.permutation(self.K) # shape: (K)
            self.K0 = K_classes[:K//2]
            self.K1 = K_classes[K//2:] 
        self.is_equalize_classes = is_equalize_classes

    def __len__(self):
        return self.len_data

    def __getitem__(self, task: int):
        # get batch of data from base mnist loader 
        len_seq = np.random.randint(2, self.len_context+1) if self.is_full_length == False else self.len_context 
        # show at least one query item
        
        if self.burstiness == 1:
            # while True:
            if self.K < 0:
                targets = np.random.permutation(self.len_context)
                targets[-1] = np.random.choice(targets[:-1]) # resample the last target
            else:
                if self.prob_mix_up > 0:
                    # negative target means mix-up
                    targets = np.where (np.random.rand(len_seq) < self.prob_mix_up, 
                        -1-np.random.choice (len_seq, size=len_seq, replace=False),
                        np.random.choice(self.K, size=len_seq, replace=True))
                elif self.is_equalize_classes == "True":
                    class_0 = np.random.choice(self.K0, size=len_seq//2, replace=True) 
                    class_1 = np.random.choice(self.K1, size=len_seq//2, replace=True)
                    targets = np.concatenate([ 
                        class_0, 
                        class_1 
                    ]) 

                    # get number of times seq_target appears in the sequence
                    seq_target_id = np.random.choice(self.len_context)
                    num_seq_target = np.sum(targets == targets[seq_target_id]) - 1 # exclude the last item (target)
                    if num_seq_target < 1:
                
                        target_class = np.random.choice([0, 1], size=1, replace=False) 
                        if target_class == 0:
                            class_0[-1] = np.random.choice(class_0[:-1]) # resample the last target
                            targets = np.concatenate([ 
                                class_0[:-1], 
                                class_1,
                                class_0[-1:]
                            ]) 
                        else:
                            class_1[-1] = np.random.choice(class_1[:-1]) # resample the last target
                            targets = np.concatenate([ 
                                class_0, 
                                class_1[:-1],
                                class_1[-1:]
                            ])
                    else:
                        targets = np.concatenate([ 
                            targets[:seq_target_id], 
                            targets[seq_target_id+1:], 
                            targets[seq_target_id:seq_target_id+1]
                        ])

                    
                    
                    ordering = np.random.permutation(len_seq - 1) 
                    targets[:-1] = targets[ordering]
                else:
                    targets = np.random.choice(self.K, size=len_seq, replace=True)  

                if self.is_equalize_classes != "True":
                    seq_target = targets[-1] 
                    # get number of times seq_target appears in the sequence
                    num_seq_target = np.sum(targets == seq_target) - 1 # exclude the last item (target)
                    if num_seq_target < 1:
                        targets[-1] = np.random.choice(targets[:-1]) # resample the last target
                
                
        elif self.burstiness == 0:
            targets = np.random.choice(self.K, size=len_seq, replace=True)  
            
        elif self.burstiness == -1:
            if self.K < 0:
                targets = np.random.permutation(self.len_context)
            else:
                targets = np.random.choice(self.K, size=len_seq, replace=False)
        
        if self.K < 0: # negative K means we take the limit K->infinity
            self.mu_K = torch.randn(self.len_context, self.D) * (1.0 / np.sqrt(self.D)) # shape: (K, D)
            samples = self.mu_K[targets]
        elif self.prob_mix_up > 0:
            samples = torch.zeros((len_seq, self.D), dtype=self.mu_K.dtype)
            original_samples = (targets >= 0)
            samples[original_samples] = self.mu_K[targets[original_samples]]
            mix_up_samples = torch.tensor(targets < 0)
            samples[mix_up_samples] = torch.randn(torch.sum(mix_up_samples), self.D, dtype=self.mu_K.dtype) * (1.0 / np.sqrt(self.D) )
            if targets[-1] < 0: # the seq_target is mixed up
                # there should be exactly one matching target in the sequence
                samples[-1] = (samples[:-1])[targets[:-1] == targets[-1]] 

        else:
            samples = self.mu_K[targets]  
        
        if self.epsilon>0:
            samples += self.epsilon * torch.randn(*samples.shape) * (1.0 / np.sqrt(self.D))  
            samples /= np.sqrt(1 + self.epsilon**2) # approximately normalize, shape: (len_seq, D)
        
        targets = torch.tensor(targets)   # shape: (len_seq)
       
        # pad samples so that all sequences have the same length (len_context)
        if len_seq < self.len_context: 
            pad = torch.zeros((self.len_context - len_seq, self.D))
            samples = torch.cat([pad, samples], dim=0)
            pad_target = torch.zeros((self.len_context - len_seq))-1
            targets = torch.cat([pad_target, targets], dim=0)
        # return support and query sets
        return samples.type(torch.float32), targets.type(torch.int64)  

 

importlib.reload(attention)
# define the model, optimizer, and scheduler, and criterion
if args.arch == "causal_transformer_embed":
    nheads = 1 # np.clip(args.num_hidden_features // 8, 1, 8)
    model = attention.CausalTransformer(x_dim=D+args.L, 
                                                        model_size=64,  
                                  num_classes=args.L,                 
                                  mlp_dim=args.num_hidden_features,
                                  is_layer_norm=args.is_layer_norm
                                  ).to(device)
elif args.arch == "multilayer_transformer":
    nheads = 1 # np.clip(args.num_hidden_features // 8, 1, 8)
    model = attention.MultilayerCausalTransformer(x_dim=D+args.L, 
                                                  num_attn_layers=args.num_attn_layers,
                                                        model_size=64,  
                                  num_classes=args.L,                     
                                  mlp_dim=args.num_hidden_features,
                                  is_layer_norm=args.is_layer_norm
                                  ).to(device)
 
    
if args.optimizer == 'SGD':
    opt_grouped_parameters = [
        {"params": model.to_embedding.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": model.qkv.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": model.mlp.parameters(), "lr": args.lr, "weight_decay": 1e-10}
    ]
    optimizer = torch.optim.SGD(opt_grouped_parameters,  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),  
                            lr=args.lr, 
                            weight_decay=args.weight_decay
                            )
else:
    raise ValueError("optimizer not recognized")

criterion = nn.LogSigmoid().to(device)


# reset seed for data loading
args.data_seed = int(time.time())
random.seed(args.data_seed)
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# define the dataset
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.batch_size}
use_cuda = not args.no_cuda and torch.cuda.is_available()
if use_cuda:
    cuda_kwargs = {'num_workers': args.workers,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

 

train_dataset = SequenceGMM(K = args.K, D = D, 
                            burstiness=args.burstiness,
                            epsilon = args.epsilon, len_context = args.len_context, 
                            is_full_length = True,
                            len_data = 60000,
                            seed=10,
                            prob_mix_up = args.prob_mix_up,
                            is_equalize_classes=args.is_equalize_classes
                            ) 
if args.is_equalize_classes == "True":
    K0 = train_dataset.K0
    K1 = train_dataset.K1
icl_dataset = SequenceGMM(K = min(20000, args.K), D = D, 
                          burstiness=1,
                          epsilon = args.epsilon, len_context = args.len_context, 
                          is_full_length = True, len_data = 2000,
                        seed=100, prob_mix_up = args.prob_mix_up,
                        is_equalize_classes=args.is_equalize_classes)


iwl_dataset = SequenceGMM(K = args.K, D = D, 
                          burstiness=-1,
                          epsilon = args.epsilon, len_context = args.len_context, 
                          is_full_length = True, len_data = 2000,
                          prob_mix_up = args.prob_mix_up,
                          is_equalize_classes=args.is_equalize_classes,
                        seed=10)
iwl_dataset.mu_K = train_dataset.mu_K # use the same mu_K for train and val 
train_sampler = None
val_sampler = None 

train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
    **train_kwargs)

icl_loader = torch.utils.data.DataLoader(
    icl_dataset, sampler=val_sampler, shuffle=(val_sampler is None), **test_kwargs)
      
iwl_loader = torch.utils.data.DataLoader(
    iwl_dataset, sampler=val_sampler, shuffle=(val_sampler is None), **test_kwargs)
print ("Made datasets")
 
class LoadProjsPermutations(torch.utils.data.Dataset):
    def __init__(self, num_tasks, D, len_data):
        self.num_tasks = num_tasks 
        if args.is_all_tasks_seen != "True":
            self.input_projs = torch.randn(num_tasks, D, D) / np.sqrt(D) 
        else:
            self.input_projs = [1]
        self.output_permutations = {}
        for task in range(num_tasks): 
            if args.K < 0:
                self.output_permutations[task] = np.concatenate([np.arange(args.L) for _ in range(args.len_context // args.L)] )
            else:
                self.output_permutations[task] = np.concatenate([np.arange(args.L) for _ in range(args.K // args.L)] )
            np.random.shuffle(self.output_permutations[task])
            self.output_permutations[task] = torch.tensor(self.output_permutations[task])

        self.len_data = len_data
        
    def __len__(self):
        return self.len_data

    def __getitem__(self, i: int):
        if args.K < 0: 
            self.output_permutations = {0: np.random.choice(args.L, size=args.len_context, replace=True)}
            np.random.shuffle(self.output_permutations[0])
            return self.input_projs[i % self.num_tasks], self.output_permutations[i % self.num_tasks]
        else:
            return self.input_projs[i % self.num_tasks], self.output_permutations[i % self.num_tasks]
    
seen_projs_permutations_dataset = LoadProjsPermutations(args.num_tasks, D=D, len_data=60000) 
if args.is_equalize_classes == "True":
    seen_projs_permutations_dataset.output_permutations[0][K0] = 0
    seen_projs_permutations_dataset.output_permutations[0][K1] = 1
seen_projs_permutations_loader = torch.utils.data.DataLoader(seen_projs_permutations_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True) 
print ("Made seen_projs_permutations_loader")

def get_random_projs_and_permuted_outputs(images, target, input_proj, output_permutation, args, device):
    """
    images: shape (batch_size, len_context, 28*28), sequence of images, padded with zeros if sequences of different lengths
    target: shape (batch_size, len_context), sequence of labels, padded with -1 if sequences of different lengths
    input_proj: shape (batch_size, 28*28, 28*28), batch of sampled random projection matrices
    output_permutation: shape (batch_size, 2), batch of sampled permutations of output labels
    """
    # move data to the same device as model
    images = images.to(device, non_blocking=True) # shape: (batch_size, len_context, D)
    target = target.to(device, non_blocking=True).type(torch.int64) # shape: (batch_size, len_context)
    input_proj = input_proj.to(device, non_blocking=True)[:images.size(0)] # shape: (batch_size, 28*28, 28*28)
    output_permutation = output_permutation.to(device, non_blocking=True)[:images.size(0)] # shape: (batch_size, 10) 
    
    # permute target labels, clip target since padded items are marked -1
    permuted_target = torch.gather(output_permutation, dim=1, index=torch.clip(target, min=0)) # permute target labels, clip target since padded items are marked -1
        
    seq_target = permuted_target[:,-1].clone() # * 2 - 1 # convert to -1, 1
    permuted_target = F.one_hot(permuted_target, num_classes=args.L) # one hot encode permuted target labels
    # zero out the label of the last sequence, network has to predict this
    permuted_target[:,-1,:] = 0 
    B, N = torch.where(target==-1)
    permuted_target[B, N, :] = 0 # zero out the label of the padded items
    permuted_target = permuted_target.float() # shape: (batch_size, len_context, 10)
    seq = torch.cat([images, permuted_target], dim=-1) # shape: (batch_size, len_context, D+2 or D+1)
    return seq, seq_target

def validate_gradient_descent(epoch, val_loader, projs_permutations_loader, model, args, criterion, device):
    # seq_lens = list(range(1, args.len_context+1, 5)) 
    seq_lens = [args.len_context]
    if seq_lens[-1] != args.len_context: 
        seq_lens.append(args.len_context)
    test_losses = {seq_len: utils.AverageMeter('Loss', ':.4e') for seq_len in seq_lens}
    test_top1 = {seq_len: utils.AverageMeter('Acc@1', ':6.2f') for seq_len in seq_lens}
    model.eval() # switch to eval mode
    phi_xt_list = []
    labels_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            input_proj, output_permutation = next(iter(projs_permutations_loader))
            
            for seq_len in seq_lens:
                temp_images = torch.zeros((images.shape[0], args.len_context, images.shape[2]), device=device) # shape: (batch_size, seq_len, D)
                temp_images[:, -seq_len:, :] = images[:, -seq_len:, :]
                temp_target = torch.zeros((target.shape[0], args.len_context), device=device) - 1 # shape: (batch_size, seq_len)
                temp_target[:, -seq_len:] = target[:, -seq_len:]
                # print ("temp_images", temp_images.shape, "temp_target", temp_target.shape)
                seq, seq_target = get_random_projs_and_permuted_outputs(temp_images, temp_target, input_proj, output_permutation, args, device)
               
                output, _, _ = model(seq) # shape: (batch_size, 10) 
                
                # loss = criterion(output.squeeze(1), seq_target.float())
                loss = -criterion(output.squeeze(1) * (seq_target*2-1)).mean()
                
                test_losses[seq_len].update(loss.item(), target.size(0))
                # acc1 = utils.accuracy(output, seq_target, topk=[1])
                # test_top1[seq_len].update(acc1[0], target.size(0))
                acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
                test_top1[seq_len].update(acc1, target.size(0))
 
    return test_losses, test_top1, phi_xt_list, labels_list



import pickle
# import matplotlib.pyplot as plt
exp_name = f"./cache/{args.experiment_name}_{time.time()}.pkl"
for epoch in range(args.epochs):
    model.train() # switch to train mode
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')

    end = time.time()
    phi_xt_list = []
    labels_list = []
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input_proj, output_permutation = next(iter(seen_projs_permutations_loader))
        
        #print ("images", images.shape, "target", target.shape, "input_proj", input_proj.shape, "output_permutation", output_permutation[0])
        
        with torch.no_grad():
            seq, seq_target = get_random_projs_and_permuted_outputs(images, target, input_proj, output_permutation, args, device)
         
        output, _, _ = model(seq) 
        loss = -criterion(output.squeeze(1) * (seq_target*2-1)).mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0)) 
        # acc1 = utils.accuracy(output, (seq_target), topk=[1])
        # print ("output", output.shape, output[0], seq_target[0], loss, acc1, model.temperature)
        # top1.update(acc1[0], target.size(0))
        acc1 = torch.mean(((output.squeeze(1) * (seq_target*2-1)) > 0).float()).item()
        top1.update(acc1, target.size(0))
 
    # step scheduler
    # scheduler.step()

    # save metrics
    icl_val_losses, icl_val_top1, icl_phi_xt_list, icl_labels_list = validate_gradient_descent(epoch, icl_loader, seen_projs_permutations_loader, model, args, criterion, device)
    
    iwl_val_losses, iwl_val_top1, iwl_phi_xt_list, iwl_labels_list = validate_gradient_descent(epoch, iwl_loader, seen_projs_permutations_loader, model, args, criterion, device)
    logs = {
            "train_loss": losses.avg,
            "train_top1": top1.avg,
            "epoch": epoch,
        }
    for seq_len, d in icl_val_losses.items():
        logs[f"icl_val_losses{seq_len}"] = d.avg 
    for seq_len, d in icl_val_top1.items(): 
        logs[f"icl_val_top1{seq_len}"] = d.avg 
    for seq_len, d in iwl_val_losses.items():
        logs[f"iwl_val_losses{seq_len}"] = d.avg 
    for seq_len, d in iwl_val_top1.items():
        logs[f"iwl_val_top1{seq_len}"] = d.avg
    print(logs) 
    
    if args.wandb_log:
        wandb.log(logs)
    else:
        record["logs"].append(logs)
    
  

    if logs[f"icl_val_top1{args.len_context}"] > 0.99: # early stopping
        break

if args.wandb_log != True:
    with open(exp_name, "wb") as f:
        pickle.dump(record, f)
  