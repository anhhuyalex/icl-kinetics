import wandb
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import argparse
from glob import glob
import pickle
from collections import defaultdict
import copy

sns.set(context='talk', font_scale=1.0,font = "Arial",color_codes=True, palette='deep', style='ticks', rc={'mathtext.fontset': 'cm', 'xtick.direction': 'in','ytick.direction': 'in', 'axes.linewidth': 1.5, 'figure.dpi':70, 'text.usetex':True, 'font.size':24})

parser = argparse.ArgumentParser(description='GMM L2L Training with Sequence Model')
parser.add_argument('--wandb_entity', type=str, default="", help='wandb entity')
parser.add_argument('--wandb_project', type=str, default="", help='wandb project')
parser.add_argument('--experiment_name', default=None, type=str,
                    help='experiment name')
args = parser.parse_args()

args.wandb_group_name = args.experiment_name # set group name to experiment name
# util functions
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains juke hence will be faster to allocate than zeros
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result

if args.experiment_name == "fig2bc": # plot ICL accuracy vs. K
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})

    # Download specific data from each run
    K = [] 
    icl_val_top1, iwl_val_top1 = [], []

    for run in runs:
        if "train_loss" not in run.summary:
            continue

        try:
            K.append (run.config["K"])
            icl_val_top1.append(run.summary[f"icl_val_top1{run.config['len_context']}"]) 
            iwl_val_top1.append(run.summary[f"iwl_val_top1{run.config['len_context']}"])
        except Exception as e:  # something is wrong, print error and continue
            print(e)
    datf = pd.DataFrame({ "K": K, 
                        "icl_val_top1": icl_val_top1, "iwl_val_top1": iwl_val_top1
                        })
    def plot(): 
        df = datf 
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots()
        sns.scatterplot( data=df, y="icl_val_top1", x= "K", ax = ax )
        plt.title("Fig 2b: ICL accuracy vs. K")
        plt.show()
        
        fig, ax = plt.subplots()  
        sns.scatterplot( data=df, y="iwl_val_top1", x= "K", ax = ax ) 
        plt.title("Fig 2c: IWL accuracy vs. K")
        plt.show()
    plot()
    
elif (args.experiment_name == "fig2d"): # plot ICL accuracy vs. epochs
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})

    # Download specific data from each run
    K = [] 
    icl_val_top1 = []
    iterations = []
    num_iters_per_epoch = 469 # number of iterations per epoch (len_data=60000/batch_size=128)
    for run in runs:
        if "train_loss" not in run.summary:
            continue
        try:
            e = np.array(run.history()['epoch']) * num_iters_per_epoch
            num_epochs = len(e)
            iterations.extend(e)
            K.extend ([run.config["K"]] * num_epochs)
            icl_val_top1.extend(run.history()[f'icl_val_top1{run.config["len_context"]}'])
        except Exception as e:  # something is wrong, print error and continue
            print(e)
    datf = pd.DataFrame({"K": K, 
                        "icl_val_top1": icl_val_top1,
                        "iterations": iterations
                        })
    def plot():
        df = datf 
        sns.set_theme(style="whitegrid")
        fig,ax_epochs = plt.subplots()
        sns.lineplot( data=df, y="icl_val_top1", x= "iterations", hue="K", ax = ax_epochs )
        plt.title(f"Fig 2d: ICL accuracy vs. iterations")
        plt.show()
    plot()
    
elif args.experiment_name == "fig2e+6d": # plot ICL accuracy vs. epochs for transient experiments
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})

    # Download specific data from each run
    K = [] 
    icl_val_top1, iwl_val_top1 = [], []
    iterations = []
    num_iters_per_epoch = 469 # number of iterations per epoch (len_data=60000/batch_size=128)
    for run in runs:
        if "train_loss" not in run.summary:
            continue

        try:
            e = np.array(run.history()['epoch']) * num_iters_per_epoch
            num_epochs = len(e)
            iterations.extend(e)
            K.extend ([run.config["K"]] * num_epochs)
            icl_val_top1.extend(run.history()[f'icl_val_top1{run.config["len_context"]}'])
            iwl_val_top1.extend(run.history()[f'iwl_val_top1{run.config["len_context"]}'])
        except Exception as e:  # something is wrong, print error and continue
            print(e)
    datf = pd.DataFrame({"K": K, 
                        "icl_val_top1": icl_val_top1, "iwl_val_top1": iwl_val_top1,
                        "iterations": iterations
                        })
    def plot():
        df = datf 
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(1,1, figsize=(5, 4.5))
        sns.lineplot( data=df, y="icl_val_top1", x= "iterations", ax = ax, color = 'C3', label="ICL")
        sns.lineplot( data=df, y="iwl_val_top1", x= "iterations", ax = ax_c2, color = "C0", label="IWL")
        fig2e = "(Transient dynamics)" 
        ax.legend(loc="lower center", prop={'size': 14},frameon=False)
        plt.title(f"Fig 2e: ICL accuracy vs. iterations {fig2e}")
        plt.show()
        
        fig, ax = plt.subplots(1,1, figsize=(5, 4.5))
        sns.scatterplot( data=df, x="icl_val_top1", y= "iwl_val_top1", ax = ax, color = 'C3', label="ICL vs. IWL") 
        ax.legend(loc="lower center", prop={'size': 14},frameon=False)
        plt.title(f"Fig 6D: ICL vs. IWL")
        plt.show()
        
    plot()
elif args.experiment_name == "figa2a3":
    fnames = glob(f"./cache/{args.experiment_name}*")
    cmap = plt.get_cmap('tab10')(range(10))
    Ks, i_K, K_count = {}, 0, defaultdict(int)
    Ks = np.unique([int(f.split("_")[f.split("_").index("K")+1]) for f in fnames])
    Ks = {K:i for i, K in enumerate(Ks)}
    # fig_logits, ax_logits = plt.subplots(1,1, figsize=(8, 8))
    # fig_lm, ax_lm = plt.subplots(5, 2, figsize=(20, 30))
    fig_c1, ax_c1 = plt.subplots(1, 1, figsize=(8, 8))
    fig_c2, ax_c2 = plt.subplots(1, 1, figsize=(8, 8))
    fig_c3, ax_c3 = plt.subplots(1, 1, figsize=(8, 8))
    c1, c2, c3 = [], [], []
    c1_time, c1_K = [], []
    run_ids = []
    for run_i, fname in enumerate(fnames):
        logits, time, labels_list, K_logits = [], [], [], [] 
        # if i > 8: break
        try:
            with open(fname, 'rb') as f:
                r = pickle.load(f)
                K = (r["args"]["K"]) 
                if len(r["logs"])!=20000: continue
        except Exception as e:
            print(e)
            continue

        
        
        for t, l in enumerate(r["logs"]):
            #if K_count [K] >= 5: break
            if not ("train_phi_xt_list" in l): continue
            logits.extend(l["train_phi_xt_list"]["phi_xt_list"])
            time.extend([t] * len(l["train_phi_xt_list"]["phi_xt_list"]))
            labels_list.extend(l["train_phi_xt_list"]["labels_list"])
            K_logits.extend ([K] * len(l["train_phi_xt_list"]["phi_xt_list"])) 
            
            # Compute c1 \equiv \langle \sigma(-\phi^+)\rangle_{\phi^+}
            positive_labels = np.where(l["train_phi_xt_list"]["labels_list"] == 1)
            positive_logits = l["train_phi_xt_list"]["phi_xt_list"][positive_labels]
            c1 += [sigmoid(-positive_logits).mean()]
             
            # Compute $c_2 \equiv 1 - \langle \sigma(-\phi^+)^2\rangle_{\phi^+}/c_1$  
            c2 += [1 - (1 - ((sigmoid(-positive_logits))**2).mean()/c1[-1])]

            # Compute $c_3 \equiv \langle e^{-\phi_+} \rangle_{\phi_+}$
            c3 += [np.exp(-positive_logits).mean()]
            
            c1_time += [(t+1)*np.ceil(60000/128)]
            c1_K += [K]
            run_ids +=  [run_ids]
          
    c1_c2_pd = pd.DataFrame({
        "c1": c1,
        "1-c2": c2,
        "c3": c3,
        "time": c1_time, 
        "K": c1_K,
        "run_ids": run_ids
    })

    sns.lineplot(data=c1_c2_pd, x="time", y="c1", hue="K", ax = ax_c1)
    ax_c1.set_title(f"c_1")
    ax_c1.set_yscale('log')
    ax_c1.set_xscale('log')
    ax_c1.set_title(f"Supp Fig A2a: c1 vs. iterations") 
    
    sns.lineplot(data=c1_c2_pd, x="time", y="1-c2", hue="K", ax = ax_c2) 
    ax_c2.set_title(f"c_2")
    ax_c2.set_yscale('log')
    ax_c2.set_xscale('log')
    ax_c2.set_title(f"Supp Fig A2b: c2 vs. iterations")
    
    sns.lineplot(data=c1_c2_pd, x="time", y="c3", hue="K", ax = ax_c3)
    ax_c3.set_title(f"c_3")
    ax_c3.set_yscale('log')
    ax_c3.set_xscale('log')
    ax_c3.set_title(f"Supp Fig A2c: c3 vs. iterations")
    
    c1_c2_pd.to_csv("./cache/c1c2c3_1e7iters_K_1e5.csv")
    plt.show()
elif args.experiment_name == "fig6a+A7": # ICL accuracy vs. Context length 
    """
    For each $N$, we determine $K^{*}$ by fitting a sigmoidal curve to ICL performance as a
    model (equation 1) at varying $N$ and take the limit $K \rightarrow \infty$ by resampling our dataset $\mathcal{D}$ at every
    training iteration. 
    
    We determine $t_{\text {ICL }}$ as the epoch at which ICL accuracy exceeds $95 \%$. 
    
    We train $\approx 100$ seeds for each $N$ to obtain the full distribution of $t_{\text {ICL }}$. Figure 6 confirms a linear
    """
    cmap = plt.get_cmap('tab10')(range(10)) 
    
    fnames = glob(f"./cache/{args.experiment_name}*")
    
    # get $t_{\text {ICL }}$ as a function of $N$ and $K$
    def get_record(fnames, icl_val_top1_list, Ks, Ns, epoch_above_thres):
        for i, fname in enumerate(fnames): 
            try:
                with open(fname, 'rb') as f:
                    r = pickle.load(f)
                    K = (r["args"]["K"]) 
                    N = (r["args"]["len_context"])
            except Exception as e:
                print(e)
            
            icl_val_top1 = [l[f"icl_val_top1{N}"] for l in r["logs"]][-1] 
            try:
                # get first epoch where ICL accuracy is above 0.95
                epoch_above_thres.append([l[f"icl_val_top1{N}"] > 0.95 for l in r["logs"]].index(True))
            except:
                epoch_above_thres.append(len(r["logs"]))
        
            icl_val_top1_list.append(icl_val_top1)
            Ks.append(np.log10(K))
            Ns.append(N)
            
        return icl_val_top1_list, Ks, Ns, epoch_above_thres
    
    # Fit sigmoidal curve to ICL accuracy vs. K for each N
    from scipy.optimize import curve_fit
    def fsigmoid(x, a, b, c, d):
        return (d / (1.0 + np.exp(-a*(x-b)))+c)
    
    icl_val_top1_list, Ks, Ns, epoch_above_thres = get_record(fnames, [], [], [], [])
    df = pd.DataFrame({"K": Ks, "N": Ns, "icl_val_top1": icl_val_top1_list, "epoch_above_thres": epoch_above_thres})
    
    N_fit, intercept_fit = [], []
    for i, N in enumerate([50, 75, 100, 150, 200, 250]): 
        fig, axis = plt.subplots(1,1,figsize=(5,4.5)) 
        axis.set_xlabel(r"$K$")
        axis.set_ylabel("ICL accuracy")
        
        df_N = df[df["N"]==N]
        axis.semilogx( 10**df_N["K"], df_N["icl_val_top1"], 'C0.', mfc = 'w',mew = 2, label="Data")
        popt, pcov = curve_fit(fsigmoid, df_N["K"], df_N["icl_val_top1"], method='dogbox', bounds=([-1., 2.,0., 0.],[100, 5.,1.,100.]))
        # plot sigmoidal curve fit
        xdata = np.linspace(np.min( df_N["K"]), np.max( df_N["K"]), 100)
        ydata = fsigmoid(xdata, *popt)
        sns.lineplot(x=10** xdata, y=ydata, ax=axis, color="red", label="Sigmoidal Fit")
        axis.legend(loc="center right", prop={'size': 10},frameon=False)
        fig.tight_layout()
        axis.set_title(f"Fig A7: ICL accuracy vs. K for $N={N}$")
        
        N_fit.append((N))
        intercept_fit.append(10**popt[1])
        plt.show()
    
    # Plot $K^{*}$ as a function of $N$
    fig, axis = plt.subplots(1,1,figsize=(5,4.5)) 
    axis.set_xlabel(r"$N$")
    axis.set_ylabel(r"$K^*$")
    axis.semilogx(N_fit, intercept_fit, 'C0.', mfc = 'w', mew = 5,label="Data")
    axis.set_yscale("log")
    fig.tight_layout()
    
    # fit linear curve to data
    (m, b), (SSE,), *_ = np.polyfit(np.log10(N_fit), np.log10(intercept_fit), 1, full=True)
    x = np.linspace(np.min(N_fit), np.max(N_fit), 100)
    y = 10**(m*np.log10(x) + b)
    axis.plot(x, y, 'C3-', label="Linear Fit")
    predicted_slope = 1/0.7 # nu=0.7 (See caption of Fig 6A)
    y_pred = 10**(predicted_slope*np.log10(x) + b-0.04) # we do not make predictions for the intercept
    axis.plot (x, y_pred, 'C4--', label="Predicted Fit")
    axis.legend(loc="center right", prop={'size': 10},frameon=False)
    axis.set_title(f"Fig 6A: $K^*$ vs. $N$")
    plt.show()
elif args.experiment_name == "fig6b+A4": # t_ICL vs. Context length as K->infty
    """
    The median time taken to acquire ICL $\left(t_{\text {ICL }}\right)$ 
    scales linearly as a function of $N$ and the distribution of $t_{\text {ICL }}$ is long-tailed. 
    """
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    # Download specific data from each run
    Ns, iterations = [], []
    num_iters_per_epoch = 469
    for i, run in enumerate(runs):
        if "train_loss" not in run.summary:
            continue
        run_history = run.history() 
        N = run.config["len_context"]
        # get first epoch where ICL accuracy is above 0.95
        e = run_history[run_history[f"icl_val_top1{N}"] > 0.95]["epoch"].min() 
        Ns.append(N)
        
        if np.isnan(e)==True:
            iterations.append(run.config["epochs"]*num_iters_per_epoch)
        else:
            iterations.append(e*num_iters_per_epoch)
            
    df = pd.DataFrame({"N": Ns, "iterations": iterations})
    for i, N in enumerate([50, 100, 150, 200, 250]): 
        df_N = copy.deepcopy(df[df["N"]==N])
        fig, axis = plt.subplots(1,1,figsize=(5,4.5)) 
        axis.set_xlabel(r"$t_{ICL}$ (in iterations $\times 10^5)$")
        df_N["iterations"] = df_N["iterations"]/1e5
        sns.histplot(data=df_N, x="iterations", kde=True, ax=axis, bins=15, stat="probability")
        axis.set_title(r"Fig A4: Histogram tICL for $N=$ " + str(N))
        plt.show()
        
    # plot t_ICL vs. N
    medians = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Create individual violin plots 
    for N in [50, 100, 150, 200, 250]:
        df_N = copy.deepcopy(df[df["N"]==N])
        df_N["iterations"] = df_N["iterations"]/1e5
        # Create violin plot
        sns.violinplot(data= df_N, y="iterations", 
                       x = N, 
                    ax=ax, 
                   inner="box", cut=0, width=0.8, native_scale=False)
        # Calculate and store median
        median = df_N["iterations"].median()
        medians.append((N, median))
        
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$t_{{ICL}}$ (in iterations $\times 10^5)$")
    
    # Convert medians list to numpy array for line fitting
    medians = np.array(medians)

    # Fit a straight line to the medians
    # slope, intercept, r_value, p_value, std_err = stats.linregress(medians[:, 0], medians[:, 1])
    (m, b), (SSE,), *_ = np.polyfit(medians[:, 0], medians[:, 1], 1, full=True)
    x = np.array([50, 100, 150, 200, 250])
    line = m * x + b
    
    # Plot the fitted line
    ax.plot(range(len(x)), line, color='blue', linestyle='--', label='Linear Fit')
    ax.plot(range(len(x)), medians[:, 1], 's', color='red',marker=".",mfc = 'w', mew = 5, label='Medians')
    # Adjust layout and save
    ax.legend(loc="upper left", prop={'size': 14},frameon=False)
    ax.set_title(r"Fig 6B: $t_{ICL}$ vs. $N$")
    plt.show()
elif args.experiment_name == "figA5": 
    """
    For multilayer transformers
    The median time taken to acquire ICL $\left(t_{\text {ICL }}\right)$ 
    scales linearly as a function of $N$ and the distribution of $t_{\text {ICL }}$ is long-tailed. 
    """
    """
    The median time taken to acquire ICL $\left(t_{\text {ICL }}\right)$ 
    scales linearly as a function of $N$ and the distribution of $t_{\text {ICL }}$ is long-tailed. 
    """
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    # Download specific data from each run
    Ns, iterations = [], []
    num_iters_per_epoch = 469
    for i, run in enumerate(runs):
        if "train_loss" not in run.summary:
            continue
        run_history = run.history() 
        N = run.config["len_context"]
        # get first epoch where ICL accuracy is above 0.95
        e = run_history[run_history[f"icl_val_top1{N}"] > 0.95]["epoch"].min() 
        Ns.append(N)
        
        if np.isnan(e)==True:
            iterations.append(run.config["epochs"]*num_iters_per_epoch)
        else:
            iterations.append(e*num_iters_per_epoch)
            
    df = pd.DataFrame({"N": Ns, "iterations": iterations})
    for i, N in enumerate([50, 100, 150, 200, 250]): 
        df_N = copy.deepcopy(df[df["N"]==N])
        fig, axis = plt.subplots(1,1,figsize=(5,4.5)) 
        axis.set_xlabel(r"$t_{ICL}$ (in iterations $\times 10^5)$")
        df_N["iterations"] = df_N["iterations"]/1e5
        sns.histplot(data=df_N, x="iterations", kde=True, ax=axis, bins=15, stat="probability")
        axis.set_title(r"Fig A5: 2-layer Transformer Histogram tICL for $N=$ " + str(N))
        plt.show()
        
    # plot t_ICL vs. N
    medians = []
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    # Create individual violin plots 
    for N in [50, 100, 150, 200, 250]:
        df_N = copy.deepcopy(df[df["N"]==N])
        df_N["iterations"] = df_N["iterations"]/1e5
        # Create violin plot
        sns.violinplot(data= df_N, y="iterations", 
                       x = N, 
                    ax=ax, 
                   inner="box", cut=0, width=0.8, native_scale=False)
        # Calculate and store median
        median = df_N["iterations"].median()
        medians.append((N, median))
        
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$t_{{ICL}}$ (in iterations $\times 10^5)$")
    
    # Convert medians list to numpy array for line fitting
    medians = np.array(medians)

    # Fit a straight line to the medians
    # slope, intercept, r_value, p_value, std_err = stats.linregress(medians[:, 0], medians[:, 1])
    (m, b), (SSE,), *_ = np.polyfit(medians[:, 0], medians[:, 1], 1, full=True)
    x = np.array([50, 100, 150, 200, 250])
    line = m * x + b
    
    # Plot the fitted line
    ax.plot(range(len(x)), line, color='blue', linestyle='--', label='Linear Fit')
    ax.plot(range(len(x)), medians[:, 1], 's', color='red',marker=".",mfc = 'w', mew = 5, label='Medians')
    # Adjust layout and save
    ax.legend(loc="upper left", prop={'size': 14},frameon=False)
    ax.set_title(r"Fig A5: 2-layer transformer $t_{ICL}$ vs. $N$")
    plt.show()
elif args.experiment_name == "fig6c":
    """
    C: bimodal performance at $K=9666$
    """
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    # Download specific data from each run
    iterations = []
    num_iters_per_epoch = 469
    for i, run in enumerate(runs):
        if "train_loss" not in run.summary:
            continue
        run_history = run.history() 
        N = run.config["len_context"]
        # get first epoch where ICL accuracy is above 0.95
        e = run_history[run_history[f"icl_val_top1{N}"] > 0.95]["epoch"].min() 
        
        if np.isnan(e)==True:
            iterations.append(run.config["epochs"]*num_iters_per_epoch)
        else:
            iterations.append(e*num_iters_per_epoch)
            
    df = pd.DataFrame({"iterations": iterations})
    fig, axis = plt.subplots(1,1,figsize=(5,4.5))
    sns.histplot(data=df, x="iterations", kde=True, ax=axis, bins=15, stat="probability")
    axis.set_ylabel("Count")
    axis.set_xlabel(r"$t_{{ICL}}$ (in iterations $\times 10^5)$")
    axis.set_title(r"Fig 6C: Histogram tICL for $K=9666$")
    plt.show()
elif args.experiment_name == "figA6": # vary weight decay to look at transient ICL dynamics
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    iterations = []
    wds = []
    iterations_per_epoch=469
    summary_icl_val_top1, summary_wds, summary_iters, summary_ttransient, summary_color, summary_marker = [], [], [], [], [], []
    is_color_already_plot = {}
    def find_epoch_below_threshold(run_history, run):
        """
        Find the first epoch where icl_val_top1100 drops below 0.6 after step 1000.
        
        Parameters:
        run_history (pd.DataFrame): DataFrame containing '_step' and 'icl_val_top1100' columns
        
        Returns:
        int or None: The first step where the condition is met, or None if no such step exists
        """
        if run_history['_step'].iloc[-1] == 400: return 6e6/iterations_per_epoch, '+', 'red'
        # Filter the dataframe to rows with _step > 1000
        filtered_df = run_history[run_history['_step'] > 1000]
        
        # Find the first row where icl_val_top1100 is below 0.6
        low_performance_row = filtered_df[filtered_df[f'icl_val_top1{run.config['len_context']}'] < 0.9]
        
        # Return the first step that meets the criteria, or None if no such step exists
        if not low_performance_row.empty: 
            return low_performance_row['_step'].iloc[0], 'o', 'blue'
        else: 
            return 5e6/iterations_per_epoch, 'x', 'orange'


    for i, run in enumerate(runs):
        weightdecay = run.config["weight_decay"]
        if "train_loss" not in run.summary:
            continue
        run_history = run.history()
        summary_icl_val_top1.append(run.summary[f"icl_val_top1{run.config['len_context']}"]) 
        summary_wds.append(weightdecay)
        summary_iters.append(run.summary["_step"] * iterations_per_epoch) 
        epoch_transient, marker, sum_color = find_epoch_below_threshold(run_history, run)
        if epoch_transient is None:
            epoch_transient = run.summary["_step"]  
        summary_ttransient.append(epoch_transient* iterations_per_epoch)
        summary_color.append(sum_color)
        summary_marker.append(marker)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        unique_markers = set (summary_marker)
        labels = {
            'o': "transient",
            'x': "no transience",
            "+": "no ICL"
        }
        # Plot the data in each marker group
        for um in unique_markers:
            mask = np.array(summary_marker) == um
            plt.scatter(x=np.array(summary_wds)[mask], y=np.array(summary_ttransient)[mask], marker=um, label=labels[um], color=np.array(summary_color)[mask])
        
        ax.set_xticks([1e-3, 2e-3, 3e-3, 4e-3, 5e-3 ])
        ax.set_xticklabels([1,2,3,4,5])
        ax.set_yticks([1e6, 2e6, 3e6, 4e6, 5e6, 6e6])
        
        ax.set_yticklabels([r'$1\times 10^6$', r'$2\times 10^6$', r'$3\times 10^6$', r'$4\times 10^6$', r'$\infty$', 'N/A'])
        plt.xlim(0.8e-3,6e-3)
        plt.xlabel (r"Weight Decay ($\times 10^{-3}$)")
        plt.ylabel (r"Transient iteration")
        plt.ylim(0, 6.5e6)

        # plt.legend(loc=(1.1,0.1))
        ax.legend(loc="center right", prop={'size': 9},frameon=True)
        ax.set_title(f"Fig A6: Transient dynamics, varying weight decay")
        plt.show()
elif args.experiment_name == "fig6e": # vary contextual statistics: equalize 0s and 1s vs. not
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    
    Ks, icl_val_top1_list = [], []
    for i, run in enumerate(runs):
        if "train_loss" not in run.summary:
            continue
        
        Ks.append(run.config["K"])
        icl_val_top1_list.append(run.summary[f"icl_val_top1{run.config['len_context']}"])
        is_equalize_classes = run.config["is_equalize_classes"]
        
    df = pd.DataFrame({"K": Ks, "icl_val_top1": icl_val_top1_list, "is_equalize_classes": is_equalize_classes})
    
    # Plot fraction of runs with ICL acquisition vs. K
    fig, axis = plt.subplots(1,1,figsize=(5,4.5)) 
    # is_equalize_classes = True 
    df_eq = df[df["is_equalize_classes"]=="True"] 
    Ks = list(np.unique (df_eq["K"]))
    icl_fraction = [] 
    for K in Ks: 
        icl_fraction.append(np.mean(df_eq[df_eq["K"]==K]["icl_val_top1"])) 
    axis.semilogx(Ks, icl_fraction, 'C0.', mfc = 'w',mew = 2, label="Equalize 0s and 1s")

    # is_equalize_classes = False 
    df_eq = df[df["is_equalize_classes"]=="False"]
    Ks = list(np.unique (df_eq["K"]))
    icl_fraction = []
    for K in Ks: 
        icl_fraction.append(np.mean(df_eq[df_eq["K"]==K]["icl_val_top1"]))
    axis.semilogx(Ks, icl_fraction, 'C3.', mfc = 'w',mew = 2,label="Standard")
    
    axis.set_ylim(-0.2,1.1)
    axis.set_xlabel(r"$K$")
    axis.set_ylabel(r"Fraction of runs that acquire ICL")
    axis.legend(loc="upper left", prop={'size': 10},frameon=False)
    axis.set_xticks([1e4, 0.5e5])
    axis.set_title(r"Fig 6E: Fraction of runs with ICL accuracy $>$ 0.75 vs. $K$")
    plt.show()
elif args.experiment_name == "figA8": # vary contextual statistics: equalize 0s and 1s
    api = wandb.Api(timeout=29)
    ENTITY = args.wandb_entity
    PROJECT = args.wandb_project
    # Filter runs by group name
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": args.wandb_group_name})
    
    icl_val_top1_list = []
    epochs_list = []
    iterations_per_epoch=469

    # Download specific data from each run
    for run in runs:
        if "train_loss" not in run.summary:
            continue
        
        run_history = run.history()
        icl_val_top1_list.append(run_history[f"icl_val_top1{run.config['len_context']}"].values)
        epochs_list.append(run_history ["epoch"].values*iterations_per_epoch)
        
    fig, axis = plt.subplots(1,1,figsize=(5,4.5))
    # axis.set_title("ICL accuracy vs. epoch")
    axis.set_xlabel(r"Training iterations $(\times 10^5)$")
    axis.set_ylabel("ICL accuracy")
    for i, (icl_val_top1, epochs) in enumerate(zip(icl_val_top1_list, epochs_list)):
        axis.plot(epochs/1e5 , icl_val_top1, label=f"Run {i+1}", alpha=0.8)
    axis.set_title(r"Fig A8: ICL accuracy vs. iterations, Equalize 0s and 1s, $K=31336$")
        