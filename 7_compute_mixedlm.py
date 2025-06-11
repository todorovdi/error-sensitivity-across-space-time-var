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

from joblib import Parallel, delayed
from itertools import product
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning    
from numpy.linalg import LinAlgError
import traceback
import gc

from bmp_base import get_lmm_pseudo_r_squared
from bmp_config import path_data, envcode2env
from bmp_behav_proc import *


# run long calc
n_jobs_inside = 1
# Number of processes
#n_jobs = 10  # Use all available CPUs even with one job
#n_jobs = 5
#n_jobs = 1
#n_jobs = 50
n_jobs = 128

backend = 'multiprocessing' # 'loky'
#backend = 'loky' 

prl_path_data = pjoin(path_data, '../NIH_behav_intermed_data', 'mixedlm_prl')

####################

N = 25; debug = False  # moment of transition from step 1 to step 3
# Create args array using product
cocols = ['None', 'env', 'ps2_']
#cocols = [ 'env'] #, 'ps2_']
#std_mavsz_range = range(2, 30)
std_mavsz_range = list(range(3,N)) + list( range(N, 40, 3) )
#std_mavsz_range = range(2, 20)
#std_mavsz_range = range(2, 12)
#std_mavsz_range = range(2, 3)
varn0s = ['error_pscadj', 'error_pscadj_abs']
varn_suffixes = 'all_suffixes'
#varn_suffixes = ['std', 'invstd', 'mavsq', 'mav_d_std', 'mav_d_var', 'Tan']

# N = 40; debug = True
# std_mavsz_range = [3,5]

# shorter
cocols = [ 'env', 'ps2_']
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
import pingouin as pg
dfcs_fixhistlen['environment'] = dfcs_fixhistlen['environment'].astype(int)
df_ = dfcs_fixhistlen[['environment','subject','trials','error_pscadj_abs_Tan29']]
assert not df_.duplicated(['subject','trials']).any()
all_suffixes = 'mav,std,invstd,mavsq,mav_d_std,mav_d_var,Tan,invmavsq,invmav,std_d_mav,invTan'.split(',')

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
if varn_suffixes == 'all_suffixes':
    varn_suffixes = all_suffixes

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

    excfmt = None
    nstarts = 1
    result = None
    if cocoln == 'None':
        s,s2 = f"err_sens ~ {varn}","1"
        model = smf.mixedlm(s, df_, 
                    groups=df_["subject"])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=ConvergenceWarning)
            result = model.fit()
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
                    result.wmess = wmess

            except LinAlgError as le:
                excfmt = traceback.format_exc()
                result = None
            results[(s,s2)] = result
        
    s2summary = {}
    for stpl,result in results.items():        
        if (result is not None) and result.converged:
            #result.remove_data()
            summary = result.summary()
            summary.tables[0].loc[5,2] = 'Converged2:'
            summary.tables[0].loc[5,3] = 'Yes' if result.converged2 else 'No'
            summary.wmess = result.wmess
            summary.params = result.params
            summary.pvalues = result.pvalues
            #summary.cov_re = result.cov_re
            #summary.cov_params = result.cov_params()
            r = get_lmm_pseudo_r_squared(result)
            summary.pseudo_r2 = r
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
        fnf = pjoin(prl_path_data, f'prl_{ind+1}_{s_}'
        print('saved to ', fnf))
        np.savez( fnf, prl )

if debug:
    print('!!!!!!!!!!!!! it was TTTTTTTTTTTTESTST')  # TEST

print(len(args))


