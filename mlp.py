#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import attention

import utils
import numpy as np
import random

import wandb 
import pickle

 
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
parser.add_argument('--num_hidden_features', default=1, type=int,
                    help='num_hidden_features')   

# Data
parser.add_argument('--K', default=2, type=int,
                    help='number of classes in GMM')
parser.add_argument('--prob_new_K', default=0.0, type=float,
                    help='probability of new K in GMM')
parser.add_argument('--L', default=2, type=int,
                    help='number of labels in GMM')
parser.add_argument('--epsilon', default=0.1, type=float,
                    help='epsilon for GMM')  
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
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_group_name",type=str,default="stability",
    help="wandb project name",
)
    

args = parser.parse_args()

assert args.K % args.L == 0, "K must be divisible by L"
if args.seed is None:
    args.seed = int(time.time())

# set seed for data set generation
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
  
if args.experiment_name == "figa2a3":
    Ks = np.logspace(np.log10(250), np.log10(5e4), 10).astype(int)*2 # len: 10
    # Ks = np.linspace( (500),  (20000), 10).astype(int)*2 # len: 15
    args.K = Ks[args.SLURM_ARRAY_TASK_ID % len(Ks)]  
    args.prob_new_K = -1
    args.weight_decay = 1e-10 
    args.wandb_log = False
    
    
# args.K = np.random.choice(Ks)
prob_new_Ks = np.arange(0.01, 0.11, 0.01) # len: 10
args.prob_new_K = -1 # prob_new_Ks[args.SLURM_ARRAY_TASK_ID % len(prob_new_Ks)]
# weight_decay = [ 1e-10  ]
args.weight_decay = 1e-10# weight_decay[(args.SLURM_ARRAY_TASK_ID // len(Ks)) % len(weight_decay)]

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
    print(f"wandb {args.wandb_project} run")
    wandb.login(host='https://stability.wandb.io') # need to configure wandb environment beforehand
    wandb_model_name = f"{args.fileprefix}_K_{args.K}_prob_new_K_{args.prob_new_K}"
    wandb_config = vars(args)
    args.wandb_group_name = f"{args.experiment_name}_seed_{args.seed}"
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

 
class GMM(torch.utils.data.Dataset):
    def __init__(self, K, D, epsilon = 0.1,
                 prob_new_K = 0.0,  
                 len_data = 500,
                 seed=None ):
        rng = np.random.default_rng(seed)
        mu_K = rng.standard_normal((K, D)) * (1.0 / np.sqrt(D)) # shape: (K, D) 
        self.mu_K = torch.tensor(mu_K)
        # self.mu_K = torch.randn(K, D) * (1.0 / np.sqrt(D)) # shape: (K, D) 
        self.K = K 
        self.D = D
        self.epsilon = epsilon
        self.prob_new_K = prob_new_K
 
        self.len_data = len_data
        K_to_L_dict = np.concatenate([np.arange(args.L) for _ in range(args.K // args.L)] )
        rng.shuffle(K_to_L_dict)
        self.K_to_L_dict = torch.tensor(K_to_L_dict)
        self.rng = rng

    def __len__(self):
        return self.len_data

    def __getitem__(self, task: int):
        # get batch of data from base mnist loader 
        # show at least one query item 
        task = task % self.K

        # sample from GMM
        if self.prob_new_K > 0 and self.rng.random() < self.prob_new_K:
            samples = torch.tensor(self.rng.standard_normal((self.D,)) * (1.0 / np.sqrt(self.D)) ) # shape: (D,)
        else:
            samples = self.mu_K[task] 

        if self.epsilon > 0: 
            self.epsilon * self.rng.standard_normal(self.D) * (1.0 / np.sqrt(self.D))  
            samples /= np.sqrt(1 + self.epsilon**2) # approximately normalize, shape: (len_seq, D)
        
        return samples.float(), self.K_to_L_dict[task] 

 
importlib.reload(attention)
if args.arch == "mlp":
    model = attention.MLP(x_dim=63,   
                                  mlp_dim=args.num_hidden_features).to(device)
else: 
    raise ValueError("model not recognized")

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),  
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

  
train_dataset = GMM(K = args.K, D = D,  
                            epsilon = args.epsilon,
                            prob_new_K = args.prob_new_K,
                            len_data = 60000,
                            seed= args.data_seed)

val_dataset = train_dataset
 
train_sampler = None
val_sampler = None 

train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, shuffle=(train_sampler is None),
    **train_kwargs)

val_loader = torch.utils.data.DataLoader(
    val_dataset, sampler=val_sampler, shuffle=(val_sampler is None), **test_kwargs)
       
print ("Made datasets")
def validate_gradient_descent(epoch, val_loader, model, args, criterion, device):
    test_losses = utils.AverageMeter('Loss', ':.4e') 
    test_top1 = utils.AverageMeter('Acc@1', ':6.2f')
    model.eval() # switch to eval mode
    phi_xt_list = []
    labels_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader): 
            images = images.to(device, non_blocking=True) # shape: (batch_size, D)
            target = target.to(device, non_blocking=True).type(torch.int64) # shape: (batch_size
             
            output = model(images) # shape: (batch_size, 10)
            if (epoch % 3 == 0) and (i < 10):
                phi_xt_list.extend(output.squeeze(1).detach().cpu().numpy())
                labels_list.extend(target.detach().cpu().numpy())
            
            loss = -criterion(output.squeeze(1) * (target*2-1)).mean()
            
            test_losses.update(loss.item(), target.size(0))
            acc1 = torch.mean(((output.squeeze(1) * (target*2-1)) > 0).float()).item()
            test_top1.update(acc1, target.size(0))
 
    return test_losses, test_top1, phi_xt_list, labels_list

exp_name = f"./cache/{args.experiment_name}_{time.time()}.pkl"
save_constants_interval = 10
for epoch in range(args.epochs):
    model.train() # switch to train mode
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
 
    phi_xt_list = []
    labels_list = []
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
         
        images = images.to(device, non_blocking=True) # shape: (batch_size, D)
        target = target.to(device, non_blocking=True).type(torch.int64) # shape: (batch_size)

        output = model(images) 

        if (epoch % save_constants_interval == 0) and (i < 10):
            phi_xt_list.extend(output.squeeze(1).detach().cpu().numpy())
            labels_list.extend(target.detach().cpu().numpy()) 

        loss = -criterion(output.squeeze(1) * (target*2-1)).mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0)) 
        acc1 = torch.mean(((output.squeeze(1)  * (target*2-1)) > 0).float()).item()
        top1.update(acc1, target.size(0))

    logs = {
            "train_loss": losses.avg,
            "train_top1": top1.avg,
            "epoch": epoch,
        }
    
    val_losses, val_top1, val_phi_xt_list, val_labels_list = validate_gradient_descent(epoch, val_loader, model, args, criterion, device)
    logs["val_loss"] = val_losses.avg 
    logs["val_top1"] = val_top1.avg
     
    print(logs)
    if (epoch % save_constants_interval == 0):
        logs["train_phi_xt_list"] = dict(phi_xt_list=np.asarray(phi_xt_list),
                                                                        labels_list=np.asarray(labels_list))
        logs["val_phi_xt_list"] = dict(phi_xt_list=np.asarray (val_phi_xt_list),
                                                                        labels_list=np.asarray(val_labels_list))
    
    if args.wandb_log:
        wandb.log(logs)
    else:
        record["logs"].append(logs)
    
 
    # save phi_xt_list_epoch 
    if epoch % 100 == 0:
        with open(exp_name, "wb") as f:
            pickle.dump(record, f)

 
with open(exp_name, "wb") as f:
    pickle.dump(record, f)
 