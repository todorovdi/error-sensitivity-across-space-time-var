# %%
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning    
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.stattools import jarque_bera
from numpy.linalg import LinAlgError
import traceback
import gc
import pingouin as pg
import argparse

from bmp_base import get_lmm_pseudo_r_squared
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

# Define the function to be executed in parallel
def run_model(args, ret_res = False, inc_prev = True):
    '''
    ret_res: if True, return the results of all models (but it does not work well for multiprocessing)
    '''
    dfcs_fixhistlen, cocoln, std_mavsz_, varn0, varn_suffix = args
    varn = f'{varn0}_{varn_suffix}{std_mavsz_}'
    subset = [varn, 'err_sens']
    if inc_prev:
        subset += ['prev_error_pscadj_abs']
    df_ = dfcs_fixhistlen.dropna(subset=subset)
    df_ = df_[~np.isinf(df_[varn])]
    df_ = df_[~np.isinf(-df_[varn])]
    df_ = df_[~np.isinf(df_['err_sens'])]
    df_ = df_[~np.isinf(-df_['err_sens'])]

    assert len(df_) > 0, f'No data for {varn} and {cocoln}'

    excfmt = None
    nstarts = 1
    result = None
    if cocoln == 'None':
        s,s2 = f"err_sens ~ {varn}","1"
        model = smf.mixedlm(s, df_, 
                    groups=df_["subject"])
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('ignore',category=ConvergenceWarning)
            result = model.fit()
            wmess = []
            for warning in w:
                wmess += [warning.message]
            result.converged2 = result.converged and \
                ( not (result.params.isna().any() | result.pvalues.isna().any()) )
            #print(f'Converged2: {result.converged2} for {s} and {s2}')
            result.wmess = wmess
        results = {(s,s2): result}
    else:
        flas = []
        s,s2 = f"err_sens ~ C({cocoln}) + {varn} + C({cocoln}) * {varn} + {varn} * prev_error_pscadj_abs",\
            f"~C({cocoln})"; flas += [(s,s2)]
        s,s2 = f"err_sens ~ C({cocoln}) + {varn} + C({cocoln}) * {varn} + {varn} * prev_error_pscadj_abs",\
            f"1"; flas += [(s,s2)]
        s,s2 = f"err_sens ~ C({cocoln}) + {varn} + C({cocoln}) * {varn}", f"~C({cocoln})"; flas += [(s,s2)]
        s,s2 = f"err_sens ~ C({cocoln}) + {varn} + C({cocoln}) * {varn}","1";  flas += [(s,s2)]
        
        s,s2 = f"err_sens ~ C({cocoln}) + {varn}",f"~C({cocoln})"; flas += [(s,s2)]
        s,s2 = f"err_sens ~ C({cocoln}) + {varn}","1"; flas += [(s,s2)]        
     
        results = {}
        for s,s2 in flas:
            try:                
                model = smf.mixedlm(s, df_.copy(), 
                            groups=df_["subject"], re_formula=s2)
                with warnings.catch_warnings(record=True) as w:
                    ###warnings.filterwarnings('ignore',category=ConvergenceWarning)     
                    # n_jobs argument does not really work :(
                    result = model.fit(n_jobs =n_jobs_inside)
                    wmess = []
                    for warning in w:
                        wmess += [warning.message]
                    result.converged2 = result.converged and \
                        ( not (result.params.isna().any() | result.pvalues.isna().any()) )
                    #print(f'Converged2: {result.converged2} for {s} and {s2}')
                    result.wmess = wmess

            #except (LinAlgError,ValueError) as le:
            except LinAlgError as le:
                excfmt = traceback.format_exc()
                result = None
            except ValueError as le:
                print(f'ValueError: {le} for {s} and {s2}')
                raise le
            results[(s,s2)] = result

    #print(len(results), 'models computed for', varn, cocoln, std_mavsz_)
        
    s2summary = {}
    for stpl,result in results.items():        
        if (result is not None) and result.converged:
            #result.remove_data()
            #from pprint import pprint

            summary = result.summary()
            if debug:
                print('result',summary)
            summary.tables[0].loc[5,2] = 'Converged2:'
            summary.tables[0].loc[5,3] = 'Yes' if result.converged2 else 'No'
            summary.wmess = result.wmess
            summary.params = result.params
            summary.pvalues = result.pvalues
            #summary.cov_re = result.cov_re
            #summary.cov_params = result.cov_params()
            r = get_lmm_pseudo_r_squared(result)
            summary.pseudo_r2 = r

            #print(f'Pseudo R2 for {stpl} = {r}')
            try:
                summary.resid = result.resid
                ks_stat, p_value = lilliefors(result.resid, dist='norm')
                jb_stat, jb_p_value, skew, kurtosis = jarque_bera(result.resid)
                summary.lilliefors_test_st = ks_stat
                summary.lilliefors_test_pv = p_value
                summary.jarque_bera_test_st = jb_stat
                summary.jarque_bera_test_pv = jb_p_value
                summary.jarque_bera_test_skew = skew
                summary.jarque_bera_test_kurtosis = kurtosis
                
                summary.fitted_values = result.fittedvalues
            except (ValueError,LinAlgError) as le:
                print(f'resid exception: {le} for {stpl}')
                summary.resid = None
                summary.fitted_values = None
                summary.lilliefors_test_st = np.nan
                summary.lilliefors_test_pv = np.nan
                summary.jarque_bera_test_st = np.nan
                summary.jarque_bera_test_pv = np.nan
                summary.jarque_bera_test_skew = np.nan
                summary.jarque_bera_test_kurtosis = np.nan
                #raise le

        else:
            summary = None
        s2summary[stpl] = summary
    print(args[1:])
    r = {'cocoln': cocoln, 'histlen': std_mavsz_,
            'varn': varn, 'varn0':varn0, 'varn_suffix':varn_suffix,             
             'excfmt':excfmt,
            's2summary': s2summary, 'retention_factor':df_.iloc[0]['retention_factor_s']}
            #'res': result}
            #'nstarts':nstarts,
    if ret_res:
        r['s2res'] = results
    return r

args = list(product(cocols, std_mavsz_range, varn0s, varn_suffixes))
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
        (delayed(run_model)( (dfcs_fixhistlen,*arg) ) for arg in args)
    if not debug:
        s_ = str(datetime.now())[:-7].replace(' ','_')
        np.savez( pjoin(prl_path_data, f'prl_alltogether_{s_}'), prl )
else:
    for arg in args:
        prl += [run_model((dfcs_fixhistlen,*arg))]
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