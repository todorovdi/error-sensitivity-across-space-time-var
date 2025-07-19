import os
from os.path import join as pjoin
from datetime import datetime
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from pingouin import ttest
import scipy.stats as stats

from joblib import Parallel, delayed
from itertools import product
import traceback
import gc
import pingouin as pg
import argparse

from bmp_base import get_lmm_pseudo_r_squared, run_linear_mixed_model 
from bmp_config import path_data, envcode2env
from bmp_behav_proc import *

all_suffixes = 'mav,std,invstd,mavsq,mav_d_std,mav_d_var,Tan,invmavsq,invmav,std_d_mav,invTan'.split(',')

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Run mixed linear models with various parameters')
parser.add_argument('--n_jobs', type=int, default=128, help='Number of parallel jobs to run')
parser.add_argument('--n_jobs_inside', type=int, default=1, help='Number of jobs inside each process')
parser.add_argument('--backend', type=str, default='multiprocessing', choices=['multiprocessing', 'loky'], help='Parallel backend')
parser.add_argument('--debug', type=int, default=0, help='Run in debug mode with fewer iterations')
parser.add_argument('--N', type=int, default=25, help='Transition point from step 1 to step 3')
parser.add_argument('--min_range', type=int, default=3, help='Minimum range value')
parser.add_argument('--max_range', type=int, default=40, help='Maximum range value')
parser.add_argument('--transforms', nargs='+', default=['id','log'], help='transforms to apply to the data')
# one can give space separated params
parser.add_argument('--cocols', nargs='+', default=[ 'env', 'ps2_'], help='Columns to use as covariates')
#'None',
parser.add_argument('--varn0s', nargs='+', default=['error_pscadj', 'error_pscadj_abs'], help='Variable name prefixes')
parser.add_argument('--varn_suffixes', nargs='+', default=all_suffixes, help='Variable name suffixes')
parser.add_argument('--save', type=str, default='results', help='Directory to save results')

args = parser.parse_args()

# Set parameters from command line arguments
n_jobs_inside = args.n_jobs_inside
n_jobs = args.n_jobs
backend = args.backend
debug = args.debug
N = args.N
cocols = args.cocols
# Set variable names
varn0s = args.varn0s
varn_suffixes = args.varn_suffixes
transforms = args.transforms

# Create output directory
prl_path_data = pjoin(path_data, '../NIH_behav_intermed_data', 'mixedlm_prl')
os.makedirs(prl_path_data, exist_ok=True)

####################

# Define range based on N value
std_mavsz_range = list(range(args.min_range, min(N, args.max_range)) )
if N > args.max_range:
    std_mavsz_range += list(range(N, args.max_range, 3))

if debug:
    std_mavsz_range = [3, 5]  # Shorter range for debugging
    n_jobs = 1  # Use single job for debugging
    n_jobs_inside = 1  # Use single job inside for debugging
    
    std_mavsz_range = [4]
    cocols = ['env']
    errn0s = ['error_pscadj_abs']
    varn_suffixes = ['mavsq']

#varn0s = ['error'] #, 'error_pscadj', 'error_pscadj_abs']
#varn_suffixes = ['std', 'invstd', 'Tan']
#N_ = 10
#std_mavsz_range = list(range(3,N_)) + list( range(N_, 32, 3) )

# #std_mavsz_range = range(2, 15)
# std_mavsz_range = range(2, 30)
# varn0s = ['error_pscadj_abs']
# #varn0s = ['error_pscadj', 'error_pscadj_abs']
# #cocols = [ 'env', 'ps2_']
# #cocols = [ 'ps2_']
# cocols = [ 'env']

####################


load = 1
if load:
    fnf_fhl = pjoin(path_data,'dfcs_fixhistlen.pkl')
    print(fnf_fhl)
    print( str(datetime.fromtimestamp(os.stat(fnf_fhl).st_mtime)))
    dfcs_fixhistlen = pd.read_pickle(fnf_fhl )
else:
    fnf = pjoin(path_data,'df_all_multi_tsz__.pkl.zip')
    print(fnf)
    print( str(datetime.fromtimestamp(os.stat(fnf).st_mtime)))
    df_all_multi_tsz = pd.read_pickle(fnf)
    df = df_all_multi_tsz.query('trial_shift_size == 1 and trial_group_col_calc == "trialwe" '
                            ' and retention_factor_s == "0.924"').copy().sort_values(['subject','trials'])
    df,dfall,ES_thr,envv,pert = addBehavCols2(df);
    dfc = df.copy()
    dfcs,dfcs_fixhistlen,dfcs_fixhistlen_untrunc,histlens  = addWindowStatCols(dfc, ES_thr, 
                                        varn0s = ['error','error_pscadj','error_pscadj_abs'])
    dfcs_fixhistlen, me_pct_excl = truncLargeStats(dfcs_fixhistlen_untrunc, histlens, 5.)

dfcs_fixhistlen['environment'] = dfcs_fixhistlen['environment'].astype(int)
df_ = dfcs_fixhistlen[['environment','subject','trials','error_pscadj_abs_Tan29']]
assert not df_.duplicated(['subject','trials']).any()

# make sure length n  is inside
varn0 = 'error_pscadj'
n = 3
for suffix in all_suffixes:
    s = f'{varn0}_{suffix}{n}'
    if s not in dfcs_fixhistlen.columns:
        print(s, 'not in dfcs_fixhistlen.columns')
    assert s in dfcs_fixhistlen.columns

print('all_suffixes = ', all_suffixes )

####################################
prl = []
if varn_suffixes[0] == 'all_suffixes':
    varn_suffixes = all_suffixes
#elif ',' in varn_suffixes:
#    varn_suffixes = varn_suffixes.split(',')
#else:
#    varn_suffixes = [varn_suffixes]
assert set(varn_suffixes).issubset(set(all_suffixes)), \
    f'varn_suffixes {varn_suffixes} not in all_suffixes {all_suffixes}'

print(varn_suffixes, 'varn_suffixes')


args = list(product(cocols, std_mavsz_range, varn0s, varn_suffixes,transforms))
print('Len args = ',len(args))

ind = 0
if n_jobs != 1:
    num_processors = os.cpu_count()

    if num_processors is not None:
        print(f"Number of available logical processors (CPUs): {num_processors}")
    else:
        print("Could not determine the number of processors.")

    # Execute in parallel
    n_jobs_eff = min(len(args), min(n_jobs, num_processors) )
    print(f"Starting {n_jobs_eff} parallel jobs")
    prl = Parallel(n_jobs=n_jobs_eff, backend = backend)\
        (delayed(run_linear_mixed_model)( (dfcs_fixhistlen,*arg), n_jobs_inside=n_jobs_inside ) for arg in args)
    if not debug:
        s_ = str(datetime.now())[:-7].replace(' ','_')
        np.savez( pjoin(prl_path_data, f'prl_alltogether_{s_}'), prl )
else:
    for arg in args:
        prl += [run_linear_mixed_model((dfcs_fixhistlen,*arg), n_jobs_inside=n_jobs_inside)]
#     for arg in args[69:69+1]:
#         prl += [run_model((dfcs_fixhistlen,*arg),ret_res=True)]
        
        if len(prl) >= 100:            
            s_ = str(datetime.now())[:-7].replace(' ','_')
            np.savez( pjoin(prl_path_data, f'prl_{ind}_{s_}'), prl )
            ind += 1
            del prl
            gc.collect()
            prl = []
    # to save the last
    s_ = str(datetime.now())[:-7].replace(' ','_')
    if not debug:
        fnf = pjoin(prl_path_data, f'prl_{ind+1}_{s_}.npz')
        print('saved to ', fnf)
        np.savez( fnf, prl )

if debug:
    print('!!!!!!!!!!!!! it was TTTTTTTTTTTTESTST')  # TEST

print(len(args))