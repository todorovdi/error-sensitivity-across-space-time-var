import os
from os.path import join as pjoin
import pandas as pd
import seaborn as sns
import numpy as np
import argparse
import datetime
from joblib import Parallel, delayed
import subprocess as sp

from bmp_base import (calc_target_coordinates_centered,subAngles)
from bmp_config import path_fig
from bmp_behav_proc import *
from figure.mystatann import plotSigAll

data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
scripts_dir = '.'

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs',  default = 20, type=int )
parser.add_argument('--save_suffix',  default='_test', type=str )
parser.add_argument('--use_sub_angles',  default=0, type=int )
parser.add_argument('--n_subjects',  default=20, type=int )
parser.add_argument('--coln_error',  default='error', type=str )
parser.add_argument('--coln_correction_calc',  default=None, type=str )
 
# script flow params
parser.add_argument('--do_read',  default=1, type=int )
parser.add_argument('--do_collect',  default=1, type=int )
parser.add_argument('--do_add_cols',  default=1, type=int )
parser.add_argument('--do_calc_ES',  default=1, type=int )
parser.add_argument('--do_plot',  default=1, type=int )
parser.add_argument('--do_save',  default=1, type=int )
parser.add_argument('--save_owncloud',  default=0, type=int )
parser.add_argument('--perturbation_random_recalc',  default=1, type=int )
parser.add_argument('--long_shift_numerator',  default=0, type=int )
 
# ES calc params
parser.add_argument('--trial_shift_size_max',  default=1, type=int )
parser.add_argument('--do_per_tgt',  default=0, type=int )
parser.add_argument('--do_per_env',  default=0, type=int )
parser.add_argument('--retention_factor',  default='1', type=str )
parser.add_argument('--reref_target_locs',  default=0, type=int )
 
args = parser.parse_args()

args.long_shift_numerator  = bool(args.long_shift_numerator)

retention_factor = None
if ',' in args.retention_factor:
    retention_factor = args.retention_factor.split(',')
else:
    retention_factor = [args.retention_factor]

print(data_dir_input, scripts_dir)
 
use_sub_angles = args.use_sub_angles

subjects = [f for f in os.listdir(data_dir_input) if f.startswith('sub') ]
subjects = list(sorted(subjects))
print(subjects)


###########################################################

if args.do_read:
    print('Start reading raw .csv files')
    #for subject in subjects:
    n_jobs = args.n_jobs
    def f(subject):
        script_name = pjoin(scripts_dir,'bmp_read_behav.py')
        p = sp.Popen((f"python {script_name} --subject {subject} "
            f"--use_sub_angles {use_sub_angles} --save_suffix {args.save_suffix} "
            f"--perturbation_random_recalc {args.perturbation_random_recalc}").split() )
        p.wait()

    r = Parallel(n_jobs=n_jobs,
         backend='multiprocessing')( (delayed(f)\
            ( subject) for subject in subjects[:args.n_subjects]) )

###########################################################
target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)     

if args.do_collect:
    behav_df_all = []
    #or subj in subjects[:4]:#[si]
    for subj in subjects[:args.n_subjects]:
        behav_data_dir = pjoin(data_dir_input,'behavdata')
        #behavdata
        task = 'VisuoMotor'
        updstr = '_upd'
        fname = pjoin(path_data, subj, 'behavdata',
                    f'behav_{task}_df{updstr}{args.save_suffix}.pkl' )
        behav_df_full = pd.read_pickle(fname)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname))

        behav_df = pd.read_pickle(fname)
        behav_df['subject'] = subj
        behav_df['mtime'] = mtime
        
        behav_df_all += [behav_df]
        
    behav_df_all = pd.concat(behav_df_all)
    bc = ['index','level_0']
    bc = list( set(bc) & set( behav_df_all.columns) )
    df_all = behav_df_all.drop(columns=bc).sort_values(['subject','trials']).reset_index(drop=True)

    assert len( df_all['subject'].unique() ) == args.n_subjects

    # save data without err sens computed
    fn = f'df_all{args.save_suffix}.pkl.zip'
    fnf = pjoin(path_data,fn)
    if args.do_save:
        behav_df_all.to_pickle(fnf , compression='zip')
        print(fnf)
        if args.save_owncloud:
            tstr = str( datetime.datetime.now() )[:10] 
            behav_df_all.to_pickle(pjoin('/home/demitau/current/merr_data',fn + '_' + tstr) , compression='zip')

badcols =  checkErrBounds(df_all)
#assert len(badcols) == 0

if args.do_add_cols:
    addBehavCols(df_all)

df_all['vals_for_corr'] = subAngles(df_all['target_locs'], df_all['org_feedback']) # movement 
vars_to_pscadj = ['vals_for_corr']
for varn in vars_to_pscadj:
    df_all[f'{varn}_pscadj'] = df_all[varn]
    df_all.loc[df_all['pert_seq_code'] == 1, f'{varn}_pscadj']= -df_all[varn]

envs = ['stable','random','all']
tgt_inds_all =  [None]
if args.do_per_tgt:
    tgt_inds_all += list(df_all['target_inds'].unique() )

envs_cur = [ 'all']
if args.do_per_env:
    envs_cur += ['stable', 'random']
block_names_cur = ['all']
pertvals_cur = [None]
gseqcs_cur = [ (0,1) ]
tgt_inds_cur = tgt_inds_all
dists_rad_from_prevtgt_cur = [None]
dists_trial_from_prevtgt_cur = [None]
error_type = 'MPE'  # observed - goal, motor performance error

if args.do_calc_ES:
    df_all_multi_tsz, ndf2vn = computeErrSensVersions(df_all, envs_cur,
        block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
        dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur,
        coln_nh = 'non_hit_not_adj',
        coln_nh_out = 'non_hit_shifted',
        computation_ver='computeErrSens3',
        subj_list = subjects[:args.n_subjects], error_type=error_type,
        trial_shift_sizes = np.arange(1, args.trial_shift_size_max + 1),
        addvars=[], use_sub_angles = use_sub_angles, 
        retention_factor = retention_factor,
        reref_target_locs = args.reref_target_locs, 
        coln_error=args.coln_error, 
        coln_correction_calc = args.coln_correction_calc,
        long_shift_numerator=args.long_shift_numerator )

    assert not df_all_multi_tsz.duplicated(['subject','trials','trial_group_col_calc','trial_shift_size','retention_factor_s']).any()

    # dirty hack
    df_all_multi_tsz['err_sens'] = -df_all_multi_tsz['err_sens']
    df_all_multi_tsz['prev_err_sens'] = -df_all_multi_tsz['prev_err_sens']

    fn = f'df_all_multi_tsz_{args.save_suffix}.pkl.zip'
    fnf = pjoin(path_data,fn)
    print(fnf)
    if args.do_save:
        df_all_multi_tsz.to_pickle(fnf, compression='zip')

        df_all_multi_tsz.query('subject == @subjects[0]').\
            to_pickle(pjoin(path_data,'df_ext_onesubj.pkl.zip'),
                  compression='zip')

##############################

df_ = df_all_multi_tsz.query('trial_shift_size == 1 and trial_group_col_calc == "trials" and retention_factor_s == "1.000"')
assert not df_.duplicated(['subject','trials']).any()


##############################

if args.do_plot and len(df_) > 0:
    df_ = truncateDf(df_, q=0,infnan_handling='discard',coln='err_sens' )
    me = df_.groupby(['subject','environment'], observed=True).\
        mean(numeric_only=1).reset_index()

    sns.set(font_scale=1.3)
    fg = sns.catplot(data = me, kind='violin', y='err_sens', 
        hue = 'environment', x='environment',  palette = ['tab:orange', 'tab:grey'], legend=None)
    for ax in fg.axes.flatten():
        ax.axhline(y=0, c='r', ls=':'); #ax.set_ylim(-5,5)


    plotSigAll(ax, 0.8, 0.05, ticklen=0.02,
           df=me, coln='err_sens', colpair = 'environment')

    ttrssig, ttrs = comparePairs(df_, 'err_sens', 'environment')
    assert ttrssig.query('ttstr == "0.0 > 1.0" and not pooled')['pval'].iloc[0] < 0.05

    ##############################

    fnfig = pjoin(path_fig, f'test_ES_mean.pdf')
    plt.savefig(fnfig)