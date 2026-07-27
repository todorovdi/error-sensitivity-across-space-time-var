from itertools import product
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

#data_dir_input = os.path.expandvars('$DATA_MEMORY_ERRORS_STAB_AND_STOCH')
scripts_dir = '.'

parser = argparse.ArgumentParser()
parser.add_argument('--n_jobs',  default = 20, type=int )
parser.add_argument('--save_suffix',  default='_test', type=str )
parser.add_argument('--read_suffix',  default=None, type=str )
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

parser.add_argument('--session_id',  default=1, type=int, required=False )
parser.add_argument('--data_subkind',  default='stabrand', type=str, required=True )
parser.add_argument('--task',  default='visuomotor', type=str, required=False )

parser.add_argument('--shuffle_seed',  default=None, type=int, required=False )
parser.add_argument('--prep_for_interactive',  default=0, type=int, required=False )
 
args = parser.parse_args()

if args.read_suffix is None:
    args.read_suffix = args.save_suffix

if args.data_subkind == 'stabrand':
    from bmp_config import path_data_stabrand as data_dir_input
    from bmp_config import subjects as subjects
elif args.data_subkind == 'passive':
    from bmp_config import path_data_passive as data_dir_input
    from bmp_config import subjects_passive as subjects


args.long_shift_numerator  = bool(args.long_shift_numerator)

retention_factor = None
if ',' in args.retention_factor:
    retention_factor = args.retention_factor.split(',')
else:
    retention_factor = [args.retention_factor]

print(data_dir_input, scripts_dir)
 
use_sub_angles = args.use_sub_angles

#subjects = [f for f in os.listdir(data_dir_input) if f.startswith('sub') ]
#subjects = list(sorted(subjects))
#print(subjects)


###########################################################

if args.do_read:
    print('Start reading raw .csv files')
    #for subject in subjects:
    n_jobs = args.n_jobs
    if args.data_subkind == 'stabrand':
        def f(subject):
            script_name = pjoin(scripts_dir,'bmp_read_behav.py')
            p = sp.Popen((f"python {script_name} --subject {subject} --data_subkind stabrand "
                f"--use_sub_angles {use_sub_angles} --save_suffix {args.save_suffix} "
                f"--perturbation_random_recalc {args.perturbation_random_recalc}").split() )
            p.wait()

        r = Parallel(n_jobs=n_jobs,
            backend='multiprocessing')( (delayed(f)\
                ( subject) for subject in subjects[:args.n_subjects]) )
    elif args.data_subkind == 'passive':
        def f(subject, session_id):
            script_name = pjoin(scripts_dir,'bmp_read_behav.py')
            p = sp.Popen((f"python {script_name} --subject {subject} "
                f"--data_subkind passive --session_id {session_id} --task {args.task} " 
                f"--use_sub_angles {use_sub_angles} --save_suffix {args.save_suffix} "
                f"--perturbation_random_recalc {args.perturbation_random_recalc}").split() )
            p.wait()

        if args.session_id < 1:
            sids = [1,2]
        else:
            sids = [args.session_id]
        r = Parallel(n_jobs=n_jobs,
            backend='multiprocessing')( (delayed(f)\
                ( subject,session_id) for (subject,session_id) in \
                    product(subjects[:args.n_subjects], sids) ) )


###########################################################
target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)
target_coords = calc_target_coordinates_centered(target_angs)     

if args.do_collect:
    behav_df_all = []
    #or subj in subjects[:4]:#[si]
    if args.session_id < 1:
        sids = [1,2]
    else:
        sids = [args.session_id]

    for session_id in sids:
        for subj in subjects[:args.n_subjects]:
            if args.data_subkind == 'stabrand':
                behav_data_dir = pjoin(data_dir_input,subj,'behavdata')
            else:
                behav_data_dir = pjoin(data_dir_input, subj, 
                                    f'session{session_id}', 'behavdata')

            if args.data_subkind != 'stabrand' and not os.path.exists(behav_data_dir):
                print(f'Skipping subject {subj} session {session_id} as no behavdata dir')
                continue
            #behavdata
            task = 'VisuoMotor'
            updstr = '_upd'
            fname = pjoin(behav_data_dir,
                        f'behav_{task}_df{updstr}{args.read_suffix}.pkl' )
            behav_df_full = pd.read_pickle(fname)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fname))

            behav_df = pd.read_pickle(fname)
            behav_df['subject'] = subj
            behav_df['mtime'] = mtime
            behav_df['session_id'] = session_id
            
            behav_df_all += [behav_df]
        
    behav_df_all = pd.concat(behav_df_all)
    bc = ['index','level_0']
    bc = list( set(bc) & set( behav_df_all.columns) )
    df_all = behav_df_all.drop(columns=bc).sort_values(['subject','trials']).reset_index(drop=True)

    assert len( df_all['subject'].unique() ) == args.n_subjects

    # save data without err sens computed
    fn = f'df_all{args.save_suffix}.pkl.zip'
    fnf = pjoin(data_dir_input,fn)
    if args.do_save:
        behav_df_all.to_pickle(fnf , compression='zip')
        print(fnf)
        if args.save_owncloud:
            tstr = str( datetime.datetime.now() )[:10] 
            behav_df_all.to_pickle(pjoin('/home/demitau/current/merr_data',fn + '_' + tstr) , compression='zip')

badcols =  checkErrBounds(df_all)
#assert len(badcols) == 0

if args.do_add_cols:
    if args.data_subkind == 'stabrand':
        ##########################
        dset = 'Romain_Exp2_Cohen'
        addBehavCols(df_all, dset = dset)
    elif args.data_subkind == 'passive':
        dset = 'Romain_Exp1_Cohen'
        dfs = []
        for session_id in sids:
            for subj in subjects[:args.n_subjects]:
                df_ = df_all.query('subject == @subj and session_id == @session_id').copy()
                if len(df_) == 0:
                    continue
                df_ = addBehavCols(df_, dset = dset)
                dfs += [df_]
        df_all = pd.concat(dfs, ignore_index=True)

df_all['vals_for_corr'] = subAngles(df_all['target_locs'], df_all['org_feedback']) # movement 
vars_to_pscadj = ['vals_for_corr']
for varn in vars_to_pscadj:
    df_all[f'{varn}_pscadj'] = df_all[varn]
    df_all.loc[df_all['pert_seq_code'] == 1, f'{varn}_pscadj']= -df_all[varn]


if args.shuffle_seed is not None:
    np.random.seed(args.shuffle_seed)
    perm = np.random.permutation(192*2)
    #print('Permutation of trialwe:', perm[:10])  
    # Apply the same permutation to every group
    # Using group_keys=False keeps the original DataFrame structure
    #['value'].transform(lambda x: x.values[perm])
    #display(df_all[['trials','error']].head())
    df_all['trials'] = df_all.groupby(['subject','env'], group_keys=False)['trials'].transform(lambda x: x.values[perm])
    df_all['trial_index'] = df_all.groupby(['subject','env'], group_keys=False)['trial_index'].transform(lambda x: x.values[perm])
    df_all = df_all.sort_values(['subject','trials']).reset_index(drop=True)
    #display(df_all[['trials','error']].head())
    print('Shuffled trials with seed', args.shuffle_seed)



envs = ['stable','random','all']
tgt_inds_all =  [None]
if args.do_per_tgt:
    tgt_inds_all += list(df_all['target_inds'].unique() )

envs_cur = [ 'all']
if args.do_per_env:
    envs_cur += ['stable', 'random']
# if args.shuffle_seed is not None:
#     envs_cur = ['random']

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
        long_shift_numerator=args.long_shift_numerator, verbose=1 )

    colns_nodup = ['subject','trials','trial_group_col_calc','trial_shift_size','retention_factor_s']
    if 'session_id' in df_all:
        colns_nodup += ['session_id']
    assert not df_all_multi_tsz.duplicated(colns_nodup).any()

    # dirty hack
    df_all_multi_tsz['err_sens'] = -df_all_multi_tsz['err_sens']
    df_all_multi_tsz['prev_err_sens'] = -df_all_multi_tsz['prev_err_sens']

    fn = f'df_all_multi_tsz_{args.save_suffix}.pkl.zip'
    fnf = pjoin(data_dir_input,fn)
    print(fnf)
    if args.do_save:
        df_all_multi_tsz.to_pickle(fnf, compression='zip')

        df_all_multi_tsz.query('subject == @subjects[0]').\
            to_pickle(pjoin(data_dir_input,'df_ext_onesubj.pkl.zip'),
                  compression='zip')

##############################

if args.do_plot or args.prep_for_interactive:
    df_ = df_all_multi_tsz.query('trial_shift_size == 1 and trial_group_col_calc == "trials" and retention_factor_s == "1.000"')
    if 'session_id' in df_all:
        if df_.session_id.nunique() == 2:
            df_2sess = df_.copy()
            df_ = df_2sess.query('session_id == 1')
    assert not df_.duplicated(['subject','trials']).any()


    ##############################
    df_ = truncateDf(df_, q=0,infnan_handling='discard',coln='err_sens' )
    me = df_.groupby(['subject','environment'], observed=True).\
        mean(numeric_only=1).reset_index()

if args.do_plot and len(df_) > 0:

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