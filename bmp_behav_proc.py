# import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
from bmp_config import *
from bmp_base import subAngles

#lh2 = time_lh + nframes of offest 

if hasattr(np, 'atan2'):
    atan2 = np.atan2
else:
    atan2 = np.arctan2

trial_group_cols_all = ['trialwb',
 'trialwe',
 'trialwpert_wb',
 'trialwpert_we',
 'trialwtgt',
 'trialwpert',
 'trialwtgt_we',
 'trialwtgt_wb',
 'trialwtgt_wpert_wb',
 'trialwtgt_wpert_we',
 'trialwpertstage_wb',
 'trialwtgt_wpertstage_wb',
 'trialwtgt_wpertstage_we' ]

# when to discard err_sens
thr_lh_ang_deg = 2
thr_lh_ang_rad = thr_lh_ang_deg * ( np.pi / 180 )

coln2pubname = {'error_lh2_ang':'Starting angle',
                'error_lh2_ang_valid':'Starting angle',
                'error_area2_signed_nn':'Signed area' ,
                'error_area2_signed_nn_valid':'Signed area' }

def getSubjPertSeqCode(subj, task = 'VisuoMotor'):
    fname = op.join(path_data, subj, 'behavdata',
                f'behav_{task}_df.pkl' )
    behav_df_full = pd.read_pickle(fname)
    dfc = behav_df_full

    test_triali = pert_seq_code_test_trial
    r = dfc.loc[dfc['trials'] == test_triali,'perturbation']
    assert len(r) == 1
    if r.values[0] > 5.:
        pert_seq_code = 0
    else:
        pert_seq_code = 1

    return pert_seq_code

def correctPertCol_NIH(df_all, use_sub_angles = 1):
    # by default perturbation in NIH data is == 0 for random, which is confusing
    assert 'environment' in df_all.columns

    c = df_all['environment'] == 0
    df_all['perturbation_'] = -100000.
    df_all.loc[c, 'perturbation_'] = df_all.loc[c,'perturbation']#.copy()
    if use_sub_angles:
        df_all.loc[~c, 'perturbation_'] = subAngles(df_all.loc[~c,'feedback'], 
                df_all.loc[~c,'org_feedback'])  / np.pi * 180
    else:
        df_all.loc[~c, 'perturbation_'] = (df_all.loc[~c,'feedback'] - df_all.loc[~c,'org_feedback']) / np.pi * 180
    df_all['perturbation'] = df_all['perturbation_']
    df_all = df_all.drop(columns=['perturbation_'])

def addBehavCols(df_all, inplace=True, skip_existing = False,
                 dset = 'Romain_Exp2_Cohen', fn_events_full = None, trial_col0 = 'trials'):
    '''
    This is for NIH experiment (and for Bonaiuto data as well)
    inplace, does not change database lengths (num of rows)

    defines trialwb, trialwe, trialwtgt, trialwpert_wb, trialwpert_we
    '''
    assert df_all.index.is_unique
    if 'target_inds' not in df_all.columns:
        df_all['target_inds'] = df_all['tgti_to_show']
        assert 'tgti_to_show' in df_all.columns
        print('addBehavCols: target_inds not found, setting target_inds to tgti_to_show')
    assert trial_col0 in df_all.columns

    if not inplace:
        df_all = df_all.copy()
    subjects     = df_all['subject'].unique()
    tgt_inds_all = df_all['target_inds'].unique()
    pertvals     = df_all['perturbation'].unique()
    if len(pertvals) > 10:
        print(f'WARNING: too many pertvals! len={len(pertvals)}')
        if dset == 'Romain_Exp2_Cohen':
            # maybe we extended perturbations to nonzero vals in random. Then we only take stable
            pertvals_eff = df_all.query('environment == 0')['perturbation'].unique() 
    else:
        pertvals_eff = pertvals

    subj = subjects[0]

    # by default perturbation in NIH data is == 0 for random, which is confusing
    if dset == 'Romain_Exp2_Cohen':
        correctPertCol_NIH(df_all)

    if fn_events_full is not None:
        import mne
        from meg_proc import addTrigPresentCol_NIH
        #fn_events = f'{task}_{hpass}_{ICAstr}_eve.txt'
        #fn_events_full = op.join(path_data, subject, fn_events )
        # these are all events, including non-target triggers
        events0 = mne.read_events(fn_events_full)
        event_ids_all_for_EC = stage2evn2event_ids['target']['all']
        mask   = np.isin(events[:,2], event_ids_all_for_EC)
        events = events0[mask]
        df_all = addTrigPresentCol_NIH(df_all, events)

    df_all['dist_trial_from_prevtgt'] = np.nan
    for subj in subjects:
        for tgti in tgt_inds_all:
            if tgti is None:
                continue
            dfc = df_all[(df_all['subject'] == subj) & (df_all['target_inds'] == tgti)]
            df_all.loc[dfc.index,'dist_trial_from_prevtgt'] =\
                df_all.loc[dfc.index, trial_col0].diff()

    #dist_deg_from_prevtgt
    #dist_trial_from_prevtgt
    # better use strings otherwise its difficult to group later
    lbd = lambda x : f'{x:.2f}'
    df_all['dist_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'dist_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff().abs()
    df_all['dist_rad_from_prevtgt'] = df_all['dist_rad_from_prevtgt'].apply(lbd)

    # signed distance
    df_all['distsgn_rad_from_prevtgt'] = None
    for subj in subjects:
        dfc = df_all[df_all['subject'] == subj]
        df_all.loc[dfc.index,'distsgn_rad_from_prevtgt'] =\
            df_all.loc[dfc.index, 'target_locs'].diff()#.apply(lbd,1)
    df_all['distsgn_rad_from_prevtgt'] = df_all['distsgn_rad_from_prevtgt'].apply(lbd)


    dts = np.arange(1,6)
    for subj in subjects:
        for dt in dts:
            dfc = df_all[df_all['subject'] == subj]
            df_all.loc[dfc.index,f'dist_rad_from_tgt-{dt}'] =\
                df_all.loc[dfc.index, 'target_locs'].diff(periods=dt).abs()
    for dt in dts:
        df_all[f'dist_rad_from_tgt-{dt}'] = df_all[f'dist_rad_from_tgt-{dt}'].apply(lbd)


    if dset == 'Romain_Exp2_Cohen':
        df_all['subject_ind'] = df_all['subject'].str[3:5].astype(int)

        test_triali = pert_seq_code_test_trial
        subj2pert_seq_code = {}
        for subj in subjects:
            mask = df_all['subject'] == subj
            dfc = df_all[mask]
            r = dfc.loc[dfc[trial_col0] == test_triali,'perturbation']
            assert len(r) == 1
            if r.values[0] > 5.:
                pert_seq_code = 0
            else:
                pert_seq_code = 1
            subj2pert_seq_code[subj] = pert_seq_code

        def f(row):
            return subj2pert_seq_code[row['subject']]

        df_all['pert_seq_code'] = df_all.apply(f,1)

        #########################   index within block (second block same numbers)

        if not (skip_existing and ('block_name' not in df_all.columns) ):
            def f(row):
                env = envcode2env[ row['environment']]
                triali = row[trial_col0]
                if env == 'stable' and triali < 200:
                    block_name = env + '1'
                elif env == 'stable' and triali > 300:
                    block_name = env + '2'
                elif env == 'random' and triali < 450:
                    block_name = env + '1'
                elif env == 'random' and triali > 500:
                    block_name = env + '2'
                else:
                    print(row)
                    raise ValueError(f'wrong combin {env}, {triali}')
                return block_name
            df_all['block_name'] = df_all.apply(f,1)
        assert 'block_name' in df_all.columns


    from collections import OrderedDict
    dfc = df_all[df_all['subject'] == subj]
    block_names = list(OrderedDict.fromkeys(dfc['block_name'] ))

    #block_names = list(sorted( df_all['block_name'].unique() ))


    #df_all['trialwb'] = None  # within respective block
    #df_all['trialwe'] = None  # within respective env (inc both blocks)
    # important to do it for a fixed subject
    df_all['trialwb'] = -1
    if dset == 'Romain_Exp2_Cohen':
        df_all['trialwe'] = -1

    mask = df_all['subject'] == subj
    dfc = df_all[mask]

    assert np.min( np.diff( dfc[trial_col0] ) ) > 0

    trials_starts = {}
    for bn in block_names:
        fvi = dfc[dfc['block_name'] == bn].first_valid_index()
        assert fvi is not None
        trials_starts[bn] = dfc.loc[fvi,trial_col0]
    
    mts = np.max( list(trials_starts.values() ) )
    print('Max of indices of trials starting a block = ',mts)
    if dset == 'Romain_Exp2_Cohen':
        assert mts <= 767, mts

    def f(row):
        bn = row['block_name']
        start = trials_starts[bn]
        return row[trial_col0] - start

    df_all['trialwb'] = -1
    for subj in subjects:
        mask = df_all['subject'] == subj
        df_all.loc[mask, 'trialwb'] = df_all[mask].apply(f,1)
    assert np.all( df_all['trialwb'] >= 0)

    ########################   index within env (second block -- diff numbers)

    # within single subject
    if dset == 'Romain_Exp2_Cohen':
        envchanges  = dfc.loc[dfc['environment'].diff() != 0,trial_col0].values
        envchanges = list(envchanges) + [len(dfc)]

        envinterval = []
        for envi,env in enumerate(block_names):
            envinterval += [ (env, (envchanges[envi], envchanges[envi+1])) ]
        block_trial_bounds = dict( ( envinterval ) )

        #block_trial_bounds = {'stable1': [0,192],
        #'random1': [192,384],
        #'stable2': [384,576],
        #'random2': [576,768]}
        def f(row):
            bn = row['block_name']
            tbs = block_trial_bounds[bn]
            start_cur = tbs[0]
            bnbase = bn[:-1]
            tbs0 = block_trial_bounds[bnbase + '1']
            end_first_rel = tbs0[-1] - tbs0[0]
            add = 0
            if bn.endswith('2'):
                add = end_first_rel
            #return row[trial_col0] - start_cur + add
            r = row[trial_col0] - start_cur + add
            if r < 0:
                raise ValueError('r < 0 ')
        #     if bn == 'random2':
        #         import pdb; pdb.set_trace()
            return r

        df_all['trialwe'] = df_all.apply(f,1)
        assert np.all( df_all['trialwe'] >= 0)


        ##########################   index within pertrubation (within block)

        bn2trial_st = {}  # block name 2 trial start
        for bn in ['stable1', 'stable2']:
            dfc_oneb = dfc[dfc['block_name'] == bn]
            df_starts = dfc_oneb.loc[dfc_oneb['perturbation'].diff() != 0]
            trial_st = df_starts[trial_col0].values

            last =  dfc_oneb.loc[dfc_oneb.last_valid_index(), trial_col0]
            trial_st = list(trial_st) +  [last + 1]

            bn2trial_st[bn] = trial_st
            assert len(trial_st) == 6, len(trial_st) # - 1

        #bn2trial_st = df_all.groupby('block_name')[trial_col0].min().to_dict()
        print(bn2trial_st)

        def f(row):
            t = row[trial_col0]
            bn = row['block_name']
            if bn not in bn2trial_st:
                return None
            trial_st = bn2trial_st[bn]
            for tsi,ts in enumerate(trial_st[:-1]):
                ts_next = trial_st[tsi+1]
                if t >= ts and t < ts_next:
                    r = tsi
                    break
            return r

        df_all['pert_stage_wb'] = df_all.apply(f,1)

        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if np.isnan(ps):
                return None
            if bn.endswith('2'):
                ps = int(ps) + 5

            return ps
        df_all['pert_stage'] = df_all.apply(f,1)

        ############################

        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if bn not in bn2trial_st:
                return None
            start = bn2trial_st[bn][int(ps)]
            return row[trial_col0] - start

        # does not distinguish two randoms
        df_all['trialwpertstage_wb'] = df_all.apply(f,1)

        #############################################

        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if np.isnan(ps):
                return None
            else:
                ps = int(ps)
            if bn not in bn2trial_st:
                return None
            ps = int(ps)
            start = bn2trial_st[bn][ps]
            #2,4
            r =  row[trial_col0] - start

            bnrebase = bn[:-1] + '1'
            l0 = bn2trial_st[bnrebase][1] - bn2trial_st[bnrebase][0]
            l1 = bn2trial_st[bnrebase][3] - bn2trial_st[bnrebase][2]
            l15 = bn2trial_st[bnrebase][2] - bn2trial_st[bnrebase][1]
            if ps == 2:
                r += l0
            elif ps == 4:
                r += l0 + l1
            elif ps == 3:
                r += l15

            return int(r)

        df_all['trialwpert_wb'] = df_all.apply(f,1)

        ######################## index within pert within env

        # we use the same but add the end of last trial of stable1.
        # note that this way we distinguish (kind of ) zero pert in the end of
        # first part and zero pert in the beg of second part
        def f(row):
            bn = row['block_name']
            ps = row['pert_stage_wb']
            if bn not in bn2trial_st:
                return None
            start = bn2trial_st[bn][int(ps)]
            add = 0
            if bn == 'stable2':
                add = bn2trial_st['stable1'][int(ps) + 1]
            elif bn == 'random2':
                add = bn2trial_st['random1'][int(ps) + 1]
            #start0 = bn2trial_st[bn][int(ps)]
            return row[trial_col0] - start + add

        df_all['trialwpert_we'] = df_all.apply(f,1)

        ############################# index within target (assuming sorted over trials)

    df_all['trialwtgt'] = -1
    for subj in subjects:
        for tgti in tgt_inds_all:
            mask = (df_all['target_inds'] == tgti) & (df_all['subject'] == subj)
            trials = df_all.loc[mask, trial_col0]
            assert np.all(np.diff(trials.values) > 0)
            df_all.loc[mask, 'trialwtgt'] = np.arange(len(trials) )
    #df_all['trialwtgt'] = df_all['trialwtgt'].astype(int)

    ########################### (assuming sorted over trials)

    if dset == 'Romain_Exp2_Cohen':
        df_all['trialwtgt_wpert_wb'] = -1
        df_all['trialwtgt_wpertstage_wb'] = -1
        df_all['trialwtgt_wpertstage_we'] = -1
        df_all['trialwtgt_wpert_we'] = -1
        df_all['trialwtgt_we'] = -1
        df_all['trialwtgt_wb'] = -1
    df_all['trialwpert']   = -1
    df_all['trialwtgt_wpert']   = -1
    for subj in subjects:
        mask0 = (df_all['subject'] == subj)
        if len(pertvals) > 10:
            mask_pert0 = mask0 & (df_all['environment'] == 0)
        else:
            mask_pert0 = mask0
                  
        for pertv in pertvals_eff:
            mask_pert = mask_pert0 & (df_all['perturbation'] == pertv)
            df_all.loc[mask_pert, 'trialwpert'] = np.arange(sum(mask_pert ) )

        for tgti in tgt_inds_all:
            #for pertv in df_all['perturbation'].unique()
            mask = (df_all['target_inds'] == tgti) & mask0
            for pertv in pertvals_eff:
                mask_pert = mask_pert0 & mask & (df_all['perturbation'] == pertv)
                df_all.loc[mask_pert, 'trialwtgt_wpert'] = np.arange(sum(mask_pert) )

                if dset == 'Romain_Exp2_Cohen':
                    for bn in block_names:
                        mask_bn = mask_pert & (df_all['block_name'] == bn)
                        trials = df_all.loc[mask_bn, trial_col0]
                        df_all.loc[mask_bn, 'trialwtgt_wpert_wb'] = np.arange(len(trials) )
                    for envc in envcode2env:
                        mask_env = mask_pert & (df_all['environment'] == envc)
                        trials = df_all.loc[mask_env, trial_col0]
                        df_all.loc[mask_env, 'trialwtgt_wpert_we'] = np.arange(len(trials) )

            if dset == 'Romain_Exp2_Cohen':
                for pert_stage in range(5):
                    for bn in block_names:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                ( df_all['block_name'] == bn )
                        trials = df_all.loc[mask_ps, trial_col0]
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_wb'] = np.arange( len(trials) )
                    for envc in envcode2env:
                        mask_ps = mask & (df_all['pert_stage_wb'] == float(pert_stage) ) &\
                                (df_all['environment'] == envc)
                        trials = df_all.loc[mask_ps, trial_col0]
                        df_all.loc[mask_ps, 'trialwtgt_wpertstage_we'] = np.arange( len(trials) )



            if dset == 'Romain_Exp2_Cohen':
                for bn in block_names:
                    mask_bn = mask & (df_all['block_name'] == bn)
                    trials = df_all.loc[mask_bn, trial_col0]
                    df_all.loc[mask_bn, 'trialwtgt_wb'] = np.arange(len(trials) )
                for envc in envcode2env:
                    mask_env = mask & (df_all['environment'] == envc)
                    trials = df_all.loc[mask_env, trial_col0]
                    df_all.loc[mask_env, 'trialwtgt_we'] = np.arange(len(trials) )
    #df_all['trialwtgt_wpert_wb'] = df_all['trialwtgt_wpert_wb'].astype(int)

    # trial_group_cols_all = [s for s in df_all.columns if s.find('trial') >= 0]
    tmax = df_all[trial_col0].max()
    for tcn in trial_group_cols_all:
        if dset == 'Romain_Exp2_Cohen':
            assert df_all[tcn].max() <= tmax, tcn
            if ('wpert' in tcn) and (len(pertvals) > 10):
                mx = df_all.query('environment == 0')[tcn].max()
                assert  mx >= 0,    (tcn, mx)
            else:
                assert df_all[tcn].max() >= 0,    tcn
        else:
            if tcn not in df_all:
                continue
            tcnm = df_all[tcn].max()
            if (tcnm >= tmax) or (tcnm <= 0):
                print(f'problem with {tcn}: max of trial={tmax} max of {tcn}={tcnm}')

    if dset == 'Romain_Exp2_Cohen':
        #pscAdj_NIH(df_all, ['error',  ] ) 
        #pscAdj_NIH(df_all, [ 'org_feedback', 'feedback' ], subpi = np.pi ) 
        #def f(x):    
        #    if x > np.pi / 2.:
        #        x -= 2*np.pi
        #    elif x < -np.pi / 2.:
        #        x += 2*np.pi
        #    return x
        #df_all['error'] = df_all['error'].apply(f)
        adjustErrBoundsPi(df_all, ['error'])

        print('ddd')
        badcols =  checkErrBounds(df_all)
        if len(badcols):
            print('bad cols 1 ', badcols)

        df_all['error_deg'] = (df_all['error'] / np.pi) * 180 


        df_all['vals_for_corr'] = df_all['target_locs'] - df_all['org_feedback'] # movement 

        vars_to_pscadj = [ 'error', 'perturbation', 'vals_for_corr']
        # 'prev_error' ?
        for varn in vars_to_pscadj:
            df_all[f'{varn}_pscadj'] = df_all[varn]
            df_all.loc[df_all['pert_seq_code'] == 1, f'{varn}_pscadj']= -df_all[varn]

        vars_to_pscadj = [ 'org_feedback']
        for varn in vars_to_pscadj:
            df_all[f'{varn}_pscadj'] = df_all[varn] - np.pi
            cond = df_all['pert_seq_code'] == 1
            df_all.loc[cond, f'{varn}_pscadj']=  - ( df_all.loc[cond,varn]  -np.pi)

        df_all['error_pscadj_pertstageadj'] = df_all['error_pscadj']
        c = (df_all['pert_stage_wb'] == 3) & (df_all['block_name'] == 'stable1')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']
        c = (df_all['pert_stage_wb'] == 1) & (df_all['block_name'] == 'stable2')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']

        c = (df_all['pert_stage_wb'] == 4) & (df_all['block_name'] == 'stable1')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']
        c = (df_all['pert_stage_wb'] == 2) & (df_all['block_name'] == 'stable2')
        df_all.loc[c, 'error_pscadj_pertstageadj'] = -df_all.loc[c, 'error_pscadj']

        addNonHitCol(df_all)


    badcols =  checkErrBounds(df_all)
    if len(badcols):
        print('bad cols ', badcols)

    return df_all

def addBehavCols2(df):
    '''this is not a replacement of addBehavCols, it just adds more stuff

        args: 
            df -- one row one trial
    '''
    mx = df.groupby(['subject'])['trial_index'].diff().max()
    assert mx == 1, mx
    assert not df.duplicated(['subject','trials']).any()
    #del df_all_multi_tsz
    #df['env'] = df['environment'].apply(lambda x: envcode2env[x])
    assert 'env' in df
    df['feedback_deg'] = df['feedback'] / np.pi * 180
    df['error_deg'] = df['error'] / np.pi * 180 


    checkErrBounds(df,['error','prev_error','error_deg'])


    df['env'] = df['env'].astype('category')
    df['subject'] = df['subject'].astype('category')
    df['trial_index'] = df['trials']

    df['error_abs'] = df['error'].abs()
    df['prev_error_abs'] = df['prev_error'].abs()

    df['prev_error_pscadj'] = df.groupby(['subject','block_name'],
            observed=True)['error_pscadj'].shift(1, fill_value=0)
    df['prev_error_pscadj_abs'] = df['prev_error_pscadj'].abs()


    # remove NaN for random
    df['pert_stage_wb'] = df['pert_stage_wb'].where(
        df['env'] =="stable", -1 )
    df['pert_stage_wb'] = df['pert_stage_wb'].astype(int)

    df['pert_stage'] = df['pert_stage'].where(
        df['env'] =="stable", -1 )
    df['pert_stage'] = df['pert_stage'].astype(int)

    print( df['pert_stage'].unique(),  df['pert_stage_wb'].unique() )

    df['ps_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1,3])
    df.loc[c,'ps_'] = 'pert' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps_'] = 'washout' 
    c = df['pert_stage_wb'].isin([0])
    df.loc[c,'ps_'] = 'pre' 
    print( df['ps_'].unique() )

    ##     ps2_    --  ps_ but perception-delay-aware
    # solve problem with first trial in pert having high ES because it was not preceived yet
    # so I shift everything
    df['ps2_'] = None
    #c = df['trialwpertstage_wb'] == 0
    #df.loc[c,'ps2_']  = df['ps_'].shift(1)

    df['ps2_']  = df.groupby(['subject','block_name'], observed=True).shift(1)['ps_']
    df.loc[ (df['env'] == 'stable') & (df['trialwb'] == 0), 'ps2_' ]  = 'pre'   # otherwise it is None
    df.loc[ (df['env'] == 'random') & (df['trialwb'] == 0), 'ps2_' ]  = 'rnd'   # otherwise it is None
    #print( df['ps2_'].unique() )
    #print( df.loc[df['ps2_'].isnull(),['trials','ps2_']])
    nu = df['ps2_'].isnull()
    #display(df[nu])
    assert not nu.any(), np.sum(nu)
    dfneq =  (df['ps_'] != df['ps2_'])
    #print( df.loc[dfneq, ['subject','trials','trialwb','ps2_','ps_','pert_stage']].iloc[:20])
    _mx  = df.loc[dfneq].groupby(['subject','pert_stage'], observed=True).size().max() 
    assert _mx == 1, _mx

    #print( df['ps2_'].unique() )
    #dfneq =  (df['ps_'] != df['ps2_'])
    #assert df.loc[dfneq].groupby(['subject','pert_stage'], observed=True).size().max() == 1

    # add contra (not perception-delay-aware)
    df['ps3_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1])
    df.loc[c,'ps3_'] = 'pert_pro' 
    c = df['pert_stage_wb'].isin([3])
    df.loc[c,'ps3_'] = 'pert_contra' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps3_'] = 'washout' 
    c = df['pert_stage_wb'].isin([0])
    df.loc[c,'ps3_'] = 'pre' 
    print( df['ps3_'].unique() )

    # separate randoms (not perception-delay-aware)
    df['ps4_'] = 'rnd'
    c = df['pert_stage_wb'].isin([1,3])
    df.loc[c,'ps4_'] = 'pert' 
    c = df['pert_stage_wb'].isin([2,4])
    df.loc[c,'ps4_'] = 'washout' 
    c = df['block_name'] == 'random1'
    df.loc[c,'ps4_'] = 'rnd1' 
    c = df['block_name'] == 'random2'
    df.loc[c,'ps4_'] = 'rnd2' 
    print( df['ps4_'].unique() )

    # subdivide randoms (perception-delay-aware)
    df['ps5_'] = df['ps2_']
    df.loc[ (df['ps2_'] == 'rnd') & (df['trialwb'] < 192. / 3.) ,'ps5_'] = 'subrnd_1'
    df.loc[ (df['ps2_'] == 'rnd') & (df['trialwb'] >= 192. / 3.) & (df['trialwb'] < 192. * 2. / 3.) ,'ps5_'] = 'subrnd_2'
    df.loc[ (df['ps2_'] == 'rnd') & (df['trialwb'] >= 192. * 2. / 3.) ,'ps5_'] = 'subrnd_3'

    df['trialwpertstage_wb'] = df['trialwpertstage_wb'].where(df['env'] =="stable", 
                                        df['trialwb'])
    df['trialwpertstage_wb'] = df['trialwpertstage_wb'].astype(int)

    assert not df.duplicated(['subject','trials']).any()

    df['thr'] = "mestd*0" # for compat
    df = addErrorThr(df)

    ###################################

    #dfc = df_wthr # NOT COPY here, we really want to add it to 
    dfc = df # NOT COPY here, we really want to add it to 
    # df_wthr (to filter TAN by consistency later)
    dfc['err_sens_change'] = dfc['err_sens'] - dfc['prev_err_sens']

    # without throwing away small errors
    dfc['subj'] = dfc['subject'].str[3:5]

    dfc['prevprev_error'] = dfc.groupby(['subject'],
                    observed=True)['prev_error'].shift(1, fill_value=0)
    dfc['err_sign_same'] =  np.sign( dfc['prev_error'] ) *\
        np.sign( dfc['prevprev_error'] )  

    dfc['err_sens_change'] = dfc['err_sens'] - dfc['prev_err_sens']

    dfc['dist_rad_from_prevprevtgt'] = \
        dfc.groupby('subject', observed=True)['target_locs'].diff(2).abs()

    # if one of the errors is small
    m = (dfc['prevprev_error'].abs() * 180 / np.pi < dfc['error_deg_initstd'] ) | \
        (dfc['prev_error'].abs() * 180 / np.pi <  dfc['error_deg_initstd']   ) 
    # if both of the errors are small
    m2 = (dfc['prevprev_error'].abs() * 180 / np.pi < dfc['error_deg_initstd'] ) & \
        (dfc['prev_error'].abs() * 180 / np.pi <  dfc['error_deg_initstd']   ) 

    # set sign to one when unclear
    dfc['err_sign_same2'] = dfc['err_sign_same']
    dfc['err_sign_same2'] = dfc['err_sign_same2'].where( m, 1)#.astype(int)

    # set sign to zero when unclear
    dfc['err_sign_same3'] = dfc['err_sign_same']
    dfc['err_sign_same3'] = dfc['err_sign_same3'].where( m, 0)#.astype(int)

    # set sign to minus one when unclear
    dfc['err_sign_same4'] = dfc['err_sign_same']
    dfc['err_sign_same4'] = dfc['err_sign_same4'].where( m, -1)#.astype(int)

    # set sign to zero when unclear, more strict
    dfc['err_sign_same5'] = dfc['err_sign_same']
    dfc['err_sign_same5'] = dfc['err_sign_same5'].where( m2, 0)#.astype(int)

    # set NaNs to 0
    mnan = ~(dfc['prevprev_error'].isna() | dfc['prev_error'].isna())
    for coln_suffi in range(1,6):
        if coln_suffi == 1:
            s = ''
        else:
            s = str(coln_suffi)
        coln_cur = 'err_sign_same' + s
        dfc[coln_cur] = dfc[coln_cur].where( mnan, 0)
        dfc[coln_cur] = dfc[coln_cur].astype(int)

    dfc['dist_rad_from_prevprevtgt'] = dfc['dist_rad_from_prevprevtgt'].\
        apply(lambda x: f'{x:.2f}' )
    dfc['dist_rad_from_prevprevtgt'] 
    #assert not dfc['err_sign_same'].isna().any()
    #dfc['err_sign_same'] = dfc['err_sign_same'].astype(int)

    dfc['err_sign_pattern'] = np.sign( dfc['prevprev_error'] ).apply(str) + \
        np.sign( dfc['prev_error'] ).apply(str)    
    print('N bads =',sum(dfc['err_sign_pattern'].str.contains('nan')))
    dfc.loc[dfc['err_sign_pattern'].str.contains('nan'),'err_sign_pattern'] = ''
    dfc['err_sign_pattern'] = dfc['err_sign_pattern'].astype(str)
    dfc['err_sign_pattern'] = dfc['err_sign_pattern'].str.replace('1.0','1')

    # for stats
    dfc['error_pscadj_abs'] = dfc['error_pscadj'].abs()
    dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].\
        where(dfc['environment'] == 0, dfc['trialwb'])
    dfc['trialwpertstage_wb'] = dfc['trialwpertstage_wb'].astype(int)

    dfc['error_change'] = dfc['error'] - dfc['error'].shift(1)
    dfc['error_pscadj_change'] = dfc['error_pscadj'] - dfc['error_pscadj'].shift(1)
    def f(x):    
        if x > np.pi:
            x -= 2*np.pi
        elif x < -np.pi:
            x += 2*np.pi
        return x
    dfc['error_change'] = dfc['error_change'].apply(f)
    dfc['error_pscadj_change'] = dfc['error_pscadj_change'].apply(f)
    dfc.loc[dfc['trialwb'] == 0, 'err_sens'] = np.nan

    ##########################

    dfni = df[~np.isinf(df['err_sens'])]
    ES_thr = calcESthr(dfni, 5.)
    dfall = truncateNIHDfFromES(df, mult=5., ES_thr=ES_thr)


    ###################################
    # just get perturabtion and env scheme from one subject
    # will be needed for plotting time resovled
    #dfall = dfall.reset_index()
    dfc_p = df.query(f'subject == "{subjects[0]}"')
    dfc_p = dfc_p.sort_values('trials')
    pert = dfc_p['perturbation'].values[:192*4]
    tr   = dfc_p['trials'].values[:192*4]
    envv = dfc_p['environment'].values[:192*4].astype(float)
    envv[envv == 0] = np.nan
    pert[envv == 1] = np.nan

    ##############################

    return df,dfall,ES_thr,envv,pert

def addWindowStatCols(dfc, ES_thr, varn0s = ['error_pscadj', 'error_pscadj_abs'],
                     histlens_min = 3, histlens_max = 40,
                      mav_d__make_abs = False, min_periods = 2, cleanTan = True   ):
    from pandas.errors import PerformanceWarning
    import warnings
    print( 'dfc.trial_group_col_calc.nunique() = ', dfc.trial_group_col_calc.nunique() )

    dfcs = dfc.sort_values(
        ['pert_seq_code', 'subject', 'trial_group_col_calc','trials']).copy()

    assert dfcs.trials.diff().max() == 1
    # good to add block name because we make a pause between so supposedly we loose memory about last errors
    grp = dfcs.\
        groupby(['pert_seq_code', 'subject', 'trial_group_col_calc','block_name'],
               observed=True)

    #varn0s = ['err_sens','error', 'org_feedback']
    #varn0s = ['err_sens','error_pscadj', 'error_change','error_pscadj_abs'] #, 'org_feedback_pscadj']
    #varn0s = ['error_pscadj', 'error_pscadj_abs'] #, 'org_feedback_pscadj']

    
    histlens = np.arange(histlens_min, histlens_max)
    ddof = 1 # pandas uses 1 by def for std calc
    for std_mavsz_ in histlens:
        for varn in varn0s:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=PerformanceWarning)

                for g,gi in grp.groups.items():
                    dfcs.loc[gi,f'{varn}_std{std_mavsz_}'] = dfcs.loc[gi,varn].shift(1).\
                        rolling(std_mavsz_, min_periods = min_periods).std(ddof=ddof)   
                    dfcs.loc[gi,f'{varn}_mav{std_mavsz_}'] = dfcs.loc[gi,varn].shift(1).\
                        rolling(std_mavsz_, min_periods = min_periods).mean()   

                dfcs[f'{varn}_invstd{std_mavsz_}'] = 1/dfcs[f'{varn}_std{std_mavsz_}']
                dfcs[f'{varn}_var{std_mavsz_}']    = dfcs[f'{varn}_std{std_mavsz_}'] ** 2
                dfcs[f'{varn}_mavsq{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] ** 2
                # shoud I change? so far I took abs of mav for mav d std and mav d var
                if mav_d__make_abs:
                    dfcs[f'{varn}_mav_d_std{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'].abs() / dfcs[f'{varn}_std{std_mavsz_}']
                    dfcs[f'{varn}_mav_d_var{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'].abs() / dfcs[f'{varn}_var{std_mavsz_}']
                else:
                    dfcs[f'{varn}_mav_d_std{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] / dfcs[f'{varn}_std{std_mavsz_}']
                    dfcs[f'{varn}_mav_d_var{std_mavsz_}']  = dfcs[f'{varn}_mav{std_mavsz_}'] / dfcs[f'{varn}_var{std_mavsz_}']
                dfcs[f'{varn}_Tan{std_mavsz_}']    = dfcs[f'{varn}_mavsq{std_mavsz_}'] / dfcs[f'{varn}_var{std_mavsz_}']
                dfcs[f'{varn}_invmavsq{std_mavsz_}'] = 1 / dfcs[f'{varn}_mavsq{std_mavsz_}']
                dfcs[f'{varn}_invmav{std_mavsz_}']   = 1 / dfcs[f'{varn}_mav{std_mavsz_}']
                dfcs[f'{varn}_std_d_mav{std_mavsz_}']   = dfcs[f'{varn}_std{std_mavsz_}'] / dfcs[f'{varn}_mav{std_mavsz_}']
                dfcs[f'{varn}_invTan{std_mavsz_}']   = dfcs[f'{varn}_std{std_mavsz_}']**2 / dfcs[f'{varn}_mav{std_mavsz_}']**2

    if cleanTan:
        print('Cleaning Tan')
        for std_mavsz_ in histlens:
            for varn in varn0s:
                c = dfcs['trialwb'] < std_mavsz_
                dfcs.loc[c,f'{varn}_Tan{std_mavsz_}'] = np.nan

    dfcs_fixhistlen_untrunc = dfcs.copy()
                
    # remove too big ES
    dfcs1 = dfcs.query('err_sens.abs() <= @ES_thr')
    dfcs_fixhistlen  = truncateDf(dfcs1, 'err_sens', q=0.0, infnan_handling='discard',  cols_uniqify = ['subject'],
                                  verbose=True) #,'env'
    dfcs_fixhistlen['environment'] = dfcs_fixhistlen['environment'].astype(int)
    #dfcs_fixhistlen_untrunc = dfcs_fixhistlen.copy()
    import gc; gc.collect()
    print('addWindowStatCols: Finished')
            
    return dfcs, dfcs_fixhistlen, dfcs_fixhistlen_untrunc, histlens
    #dfall_notclean_ = pd.concat(dfs)
    # ttrs = pd.concat(ttrs)
    # ttrs = ttrs.rename(columns={'p-val':'pval'})

def getQueryPct(df,qs,verbose=True):
    szprop = df.query(qs).groupby(['subject'],observed=True).size() / df.groupby(['subject'],observed=True).size()
    szprop *= 100
    me,std = szprop.mean(), szprop.std()
    if verbose:
        print(f'{qs} prpopration mean = {me:.3f} %, std = {std:.3f} %')
    return me,std

def truncLargeStats(dfcs_fixhistlen_untrunc, histlens, std_mult, 
    varns0 = [ 'error_pscadj_abs', 'error_pscadj'], suffixes = None):
    '''
    it NaNifies outliers but does not remove them (because if I were to remove rows for all, the dataset will become super small)
    '''
    # remove too large entries
    maxhl = np.max(histlens) ; print(maxhl)

    # NaN-ify too big stat values
    std_mult = 5.
    
    if suffixes is None:
        suffixes = 'mav,std,invstd,mavsq,mav_d_std,mav_d_var,Tan,invmavsq,invmav,std_d_mav,invTan'.split(',')

    varnames_all = []
    for varn0 in varns0: #'error_change']:
        for std_mavsz_ in histlens:#[1::10]:
            #varnames_toshow0_ = []
            for suffix in suffixes:
                varn = f'{varn0}_{suffix}{std_mavsz_}'
            #    varn = 
            #for varn in  ['{varn0}_std{std_mavsz_}',
            #              '{varn0}_invstd{std_mavsz_}',                     
            #             '{varn0}_mavsq{std_mavsz_}','{varn0}_invmavsq{std_mavsz_}',
            #              '{varn0}_mav_d_std{std_mavsz_}','{varn0}_std_d_mav{std_mavsz_}',
            #              '{varn0}_mav_d_var{std_mavsz_}',
            #             '{varn0}_Tan{std_mavsz_}','{varn0}_invTan{std_mavsz_}',
            #             '{varn0}_std{std_mavsz_}',
            #             '{varn0}_mav{std_mavsz_}','{varn0}_invmav{std_mavsz_}']:        
                varnames_all += [varn.format(varn0=varn0,std_mavsz_=std_mavsz_)]
    varnames_all            

    # here untrunc meaning without removing big stat vals (but big ES vals were removed already)
    dfcs_fixhistlen_ = dfcs_fixhistlen_untrunc.copy()
    #dfcs_fixhistlen_ = dfcs_fixhistlen
    #cs = np.ones(len(dfcs_fixhistlen), dtype=bool)
    #for varnames_toshow in varnames_toshow0:
    me_pct_excl = []
    for varn in varnames_all:
        std = dfcs_fixhistlen_untrunc[varn].std()
        c = dfcs_fixhistlen_untrunc[varn].abs() > std*  std_mult
        me_,std_ = getQueryPct(dfcs_fixhistlen_untrunc, f'{varn} > {std* std_mult}', False)
        print('Num excl: {:30}, mean={:.3} %, std={:.3f} len={:7},  stdthr={:.4f}'.format(varn,me_, std_, len(c), std*std_mult) )
        dfcs_fixhistlen_.loc[c,varn] = np.nan
        me_pct_excl += [{'varn':varn, 'mean_excl':me_, 'std_excl':std_,
                         'std_thr':std*std_mult}]
    me_pct_excl = pd.DataFrame(me_pct_excl)

    kill_Tan_2nd = False
    if kill_Tan_2nd:
        varnames_all_Tanlike = []
        for varn0 in varns0: #'error_change']:
            for std_mavsz_ in range(2,maxhl+1):#[1::10]:
                #varnames_toshow0_ = []
                for varn in  ['{varn0}_mav_d_std{std_mavsz_}',
                              '{varn0}_mav_d_var{std_mavsz_}',
                             '{varn0}_Tan{std_mavsz_}']:        
                    varnames_all_Tanlike += [varn.format(varn0=varn0,std_mavsz_=std_mavsz_)]

        for varn in varnames_all_Tanlike:
            dfcs_fixhistlen_.loc[dfcs_fixhistlen_['trialwb'] == 2, varn] = np.nan
    return dfcs_fixhistlen_, me_pct_excl

def _addErrorThr(df, stds):
    # estimate error at second halfs of init stage
    df_wthr = df.merge(stds, on='subject')

    df_wthr['error_initstd'] = df_wthr.error_deg_initstd /  180 * np.pi 
    #df_wthr
    return df_wthr

def _calcStds(df):
    qs_initstage = 'pert_stage_wb.abs() < 1e-10'
    df_init = df.query(qs_initstage + ' and trialwb >= 10')
    grp = df_init.groupby(['subject','pert_stage'],observed=True)
    #display(grp.size())
    #df_init['feedback_deg'].min(), df_init['feedback_deg'].max()

    #grp['error_deg'].std()

    stds = df_init.groupby(['subject'],observed=True)['error_deg'].std()#.std()
    return stds

def addErrorThr(df):
    # def df_thr thing
    stds = _calcStds(df)
    mestd = stds.mean()

    print('mestd = {:.4f}, stds.std() = {:.4f} '.format(mestd, stds.std() ) )

    stds = stds.to_frame().reset_index().rename(columns={'error_deg':'error_deg_initstd'})


    df_wthr = _addErrorThr(df, stds)
    return df_wthr 


def pscAdj_NIH(df_all, cols, subpi = False, inplace=True):
    if not inplace:
        df_all = df_all.copy()
    if subpi:
        sub = np.pi
    else:
        sub = 0
    vars_to_pscadj = cols
    for varn in vars_to_pscadj:
        df_all[f'{varn}_pscadj'] = df_all[varn] - np.pi
        cond = df_all['pert_seq_code'] == 1
        df_all.loc[cond, f'{varn}_pscadj']=  - ( df_all.loc[cond,varn]  - sub)
    return df_all

def addNonHitCol(df):
    # in place
    from bmp_base import point_in_circle_single, radius_cursor, radius_target
    from bmp_error_sensitivity import target_coords

    def f(row):
        #print(row.keys())
        target_ind = row['target_inds']
        feedbackX  = row['feedbackX']
        feedbackY  = row['feedbackY']
        nh = point_in_circle_single(target_ind, target_coords, feedbackX,
                        feedbackY, radius_target + radius_cursor)
        return nh

    df['non_hit_not_adj'] = df.apply(f,1)


def getSubDf(df, subj, pertv, tgti, env, block_name=None, pert_seq_code=None,
        dist_rad_from_prevtgt=None, dist_trial_from_prevtgt=None,
        non_hit=False, verbose=0, nonenan=False ):
    '''
    if nonenan is True, then NaN in numeric columns are treated as None
    inputs should be NOT lists
    '''
    assert not isinstance(pertv,list)
    assert not isinstance(tgti,list)
    assert not isinstance(block_name,list)
    assert not isinstance(subj,list)
    # and so on

    assert env in ['stable','random','all'], env
    if pertv is not None:
        assert isinstance(pertv, float) or isinstance(pertv,int), pertv
        pvm = np.abs( np.array(df['perturbation'], dtype=float) - pertv ) < 1e-10
        df = df[pvm]
    elif nonenan:
        pvm = df['perturbation'].isna()
        df = df[pvm]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after perturbation')
        else:
            print('after pert len = ',len(df ))

    if tgti is not None:
        # this is a bit ugly but since int != np.int64 it should be easier
        if str(int(tgti) ) in ['0','1','2','3']:
            df = df[df['target_inds'] == float(tgti) ]
        elif isinstance(tgti, str) and tgti == 'all':
            pvm = ~df['target_inds'].isna()
            df  = df[pvm]
    elif nonenan:
        pvm = df['target_inds'].isna()
        df = df[pvm]
    if verbose:
        if len(df) == 0:
            print('empty after target_ind')
        else:
            print('after target_ind len = ',len(df ))



    if (subj is not None):
        if isinstance(subj,list):
            df = df[df['subject'].isin(subj) ]
        elif subj != 'mean':
            df = df[df['subject'] == subj]
    if len(df) == 0 and verbose:
        print('empty after subject')
        raise ValueError(f'Nothing for subject {subj}')

    # not env == 'all'
    if (env is not None) and ('environment' in df.columns):
        if env in env2envcode:  # make trial inds (within subject)
            # could be str or int codes
            if isinstance(df['environment'].values[0], str):
                df = df[df['environment'] == env]
            else:
                envcode = env2envcode[env]
                #db_inds = np.where(df['environment'] == envcode)[0]
                df = df[df['environment'] == envcode]
        elif isinstance(env,str) and env == 'all':
            if isinstance(df['environment'].values[0], str):
                df = df[df['environment'] == env]
        else:
            raise ValueError(f'wrong env = {env}')
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after env')
        else:
            print('after env len = ',len(df ))

    if non_hit:
        df = df[df['non_hit'] ]

    # this is a subject parameter
    if pert_seq_code is not None:
        df = df[df['pert_seq_code'] == pert_seq_code]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after pert_seq_code')
        else:
            print('after pert_seq_code len = ',len(df ))

    if (block_name is not None):
        if block_name not in ['all', 'only_all']:
            df = df[df['block_name'] == block_name]
        elif block_name == 'only_all':
            df = df[df['block_name'] == 'all']
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after block_name')
        else:
            print('after block_name len = ',len(df ))

    if dist_rad_from_prevtgt is not None:
        assert type(df['dist_rad_from_prevtgt']._values[0]) == type(dist_rad_from_prevtgt)
        df = df[df['dist_rad_from_prevtgt'] == dist_rad_from_prevtgt]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after dist_rad_from_prevtgt')
        else:
            print('after dist_rad_from_prevtgt len = ',len(df ))

    if dist_trial_from_prevtgt is not None:
        df = df[df['dist_trial_from_prevtgt'] == dist_trial_from_prevtgt]
    if verbose:
        if len(df) == 0 and verbose:
            print('empty after dist_trial_from_prevtgt')
        else:
            print('after dist_trial_from_prevtgt len = ',len(df ))

    if verbose:
        print('final len(df) == ',len(df) )
    return df

def getMaskNotNanInf(vals, axis = None):
    if (vals.ndim > 1) and (axis is not None):
        #r = ~np.any( np.isinf(y), axis=1)
        r = ~ np.any ( np.isnan(vals) | np.isinf(vals), axis=axis )
    else:
        r = ~ ( np.isnan(vals) | np.isinf(vals) )
    return r

def truncateDf(df, coln, q=0.05, infnan_handling='keepnan', inplace=False,
    return_mask = False, trialcol = 'trials',
    cols_uniqify = ['trial_shift_size',
                                'trial_group_col_calc', 'retention_factor_s'] , 
    verbose=False ,
    hi = None, low=None,
    trunc_hi = True, trunc_low = True, abs=False, retloc=False):

    assert len(df), 'df is empty'

    if not inplace:
        df = df.copy()

    ntrials_per_subject = df[trialcol].nunique()

    grp0 = df.groupby([trialcol] + cols_uniqify, observed=True)
    mx=  grp0.size().max()
    assert mx <= df['subject'].nunique(), mx

    #df = df.set_index( ['subject', trialcol] + cols_uniqify )
    #mask = np.ones(len(df), dtype=bool)

    # good
    #mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    mask = getMaskNotNanInf(df[coln] )

    #print('fff')
    #if clean_infnan:
    #    # mask of good
    #    mask =  ~ ( df[coln].isna() | np.isinf( df[coln] ) )
    #    df =  df[ mask ]
    #    mask = np.array(mask)

    if  ( (q is not None) and (q > 1e-10) ) or (hi is not None) or (low is not None):
        # before some of them could be inf, not nan
        df.loc[~mask, coln] = np.nan
        if abs:
            df_ = df[['subject'] + cols_uniqify + [coln] ].copy()
            df_[coln] = df_[coln].abs()
            grp = df_.groupby(['subject'] + cols_uniqify)
            #grp = grp[coln].apply(lambda x, gk: x.abs(), group_keys=True )
            #display(grp)
        else:
            grp = df.groupby(['subject'] + cols_uniqify)

        mgsz = np.max(grp.size() )
        print(  f' np.max(grp.size() )  == {mgsz}')
        assert mgsz <= ntrials_per_subject

        if low is None:
            qlow = grp[coln].quantile(q=q)
        if hi is None:
            qhi  = grp[coln].quantile(q=1-q)

            assert not qhi.reset_index()[coln].isna().any(), qhi

        dfs = []
        # for each group separately
        for gk,ginds in grp.groups.items():

            dftmp = df.loc[ginds]

            if abs:
                dftmp_ = df_.loc[ginds]
                assert dftmp_[coln].min() >= 0
            else:
                dftmp_ = dftmp

            assert len(ginds) == len(dftmp) , 'perhaps index was not reset after concat'
            #print(len(ginds) )
            # DEBUG
            # if len(dftmp) > ntrials_per_subject:
            #    return gk, ginds, dftmp, grp
            assert len(dftmp) <= ntrials_per_subject,(len(dftmp), ntrials_per_subject)

            if low is None:
                lowc = qlow[gk]
            else:
                lowc = low
            if hi is None:
                hic  = qhi[gk]
            else:
                hic = hi

            if verbose:
                print( 'gk={}, lowc={:.6f}, hic={:.6f}'.format( gk, lowc,hic))

            mask_good  =  ~ ( dftmp[coln].isna() | np.isinf( dftmp[coln] ) )
            #mask_trunc =
            #mask_trunc = pd.DataFrame(index=dftmp.index, columns=[0], dtype=bool).fillna(True)
            mask_trunc = pd.Series(index=dftmp.index, dtype=bool)
            mask_trunc[:] = True

            # what to keep
            if trunc_hi:
                mask_trunc &= (dftmp_[coln] < hic)
            if trunc_low:
                mask_trunc &= (dftmp_[coln] > lowc)

            # what to remove
            # if keepnan, then mark as nan, but don't remove
            if infnan_handling in ['keepnan', 'discard']:
                mask_bad = (~mask_good) | (~mask_trunc)
            elif infnan_handling == 'do_nothing':
                mask_bad = (~mask_trunc)
            else:
                raise ValueError(f'Wrong {infnan_handling}')
            #display(mask_good)
            #return
            #display(mask_bad)
            #display(dftmp)
            dftmp.loc[mask_bad  , coln ] = np.nan

            mask = mask & (~mask_bad)

            if np.all( (np.isnan(dftmp[coln]) | np.isinf(dftmp[coln]))  ):
                display(dftmp[ ['subject','trials'] + cols_uniqify + ['err_sens']] )
                print(gk,len(ginds), sum(mask_bad), sum(mask_good), sum(mask_trunc) )
                raise ValueError(gk)

            # if keepnan, then mark as nan, but don't remove
            if infnan_handling == 'discard':
                dftmp = dftmp[~mask_bad]

            dfs += [dftmp]
        df = pd.concat(dfs, ignore_index = 0)  #TODO should I really ignore index here??
        print('dubplicate check ')
        assert not df.duplicated().any()
    elif infnan_handling == 'discard':
        if verbose:
            sz = df[~mask].groupby('subject', observed=True).size() / df.groupby('subject', observed=True).size() * 100 
            print( f'Discarded percentage {sz.mean():.3f}, (std={sz.std():.3f} )' )
        df = df[mask]

    r = df
    if return_mask:
        r = df, mask
    if retloc:
        r = df, locals()
    return r


def getTrialGroupName(pertv, tgti, env, block):
    assert isinstance(env,str)
    coln = 'trial'
    if tgti is not None:
        coln += 'wtgt'
    if not (pertv is None or (isinstance(pertv,str) and pertv == 'all' ) ):
        if coln != 'trial':
            coln += '_'
        coln += 'wpert'
    if env in env2envcode and ( (block is None) or (block == 'all') ) :
        if coln != 'trial':
            coln += '_'
        coln += 'we'
    if block is not None and (block in block_names):
        if coln != 'trial':
            coln += '_'
        coln += 'wb'
    assert not ( ( coln.find('wb') >= 0) and (coln.find('we') >= 0  ) ), coln

    if coln == 'trial': # if use all
        coln = 'trials' # just historically so

    return coln

def computeErrSensVersions(df_all, envs_cur,block_names_cur,
        pertvals_cur,gseqcs_cur,tgt_inds_cur,
        dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur,
        subj_list=None, error_type='MPE',
        coln_nh = 'non_hit',
        coln_nh_out = 'non_hit_shifted',
        trial_shift_sizes = [1],
        DEBUG=0, allow_duplicating=True, time_locked = 'target',
        addvars = None, target_info_type = 'inds',
        coln_correction_calc = None, coln_error = 'error',
        computation_ver = 'computeErrSens2',
        df_fulltraj = None,  trajPair2corr = None,
        drop_feedback = False,
        verbose=0, use_sub_angles = 0, retention_factor = 1.,
          reref_target_locs = True,
          long_shift_numerator = False, err_sens_coln = 'err_sens'  ):
    '''
        if allow_duplicating is False we don't allow creating copies
        of subsets of indices within subject (this can be useful for decoding)
    '''
    from bmp_config import block_names

    assert computation_ver ==  'computeErrSens3' 

    if not isinstance(retention_factor, list):
        retention_factor = [retention_factor]
    assert isinstance(dists_trial_from_prevtgt_cur, list)
    assert isinstance(dists_rad_from_prevtgt_cur, list)

    assert ('index', 'level_0') not in df_all.columns

    if not allow_duplicating:
        assert not ( (None and tgt_inds_cur) and\
                np.isin(tgt_inds_cur,np.arange(4,dtype=int) ).any() )
        assert not ( ( (None and envs_cur) or ('all' in envs_cur) ) and\
                np.isin(envs_cur, env2envcode.keys() ).any() )
        assert not ( ( (None and block_names_cur) or ('all' in block_names_cur) ) and\
                np.isin(block_names_cur, block_names ).any() )
        k = 'perturbation'
        pv = df_all.loc[~df_all[k].isnull(), k].unique()
        assert not ( ( (None and pertvals_cur) or ('all' in pertvals_cur) ) and\
                np.isin(pertvals_cur, pv).any() )
        assert not ( ( (None and pertvals_cur) or ('all' in pertvals_cur) ) and\
                np.isin(pertvals_cur, pv).any() )
        assert len(trial_shift_sizes) == 1

    dfme = []
    from itertools import product as itprod
    from bmp_error_sensitivity import computeErrSens3 as computeES
    #addargs = None # will be added later per subj
    addargs0 = {'use_sub_angles': use_sub_angles}


    p = itprod(envs_cur,block_names_cur,pertvals_cur,gseqcs_cur,tgt_inds_cur,
               dists_rad_from_prevtgt_cur,dists_trial_from_prevtgt_cur)
    p = list(p)
    print('len(prod ) = ',len(p))
    print('prod = ',p)

    if subj_list is None:
        subj_list = df_all['subject'].unique()

    #colns_set  = []; colns_skip = [];
    debug_break = 0
    dfs = []; #df_inds = []
    for subj in subj_list: #[:1]:
        for tpl in p:
            #print(len(tpl), tpl)
            (env,block_name,pertv,gseqc,tgti,drptgt,dtptgt) = tpl

            #print(tpl)
            tpl = env,block_name,pertv,gseqc,tgti,\
                drptgt,dtptgt,\
                None,None
            print(f'subj = {subj}, prod tuple contents = ', sprintf_tpl_statcalc(tpl) )
            #df = df_all
            # if we take only non-hit then, since we'll compute err sens sequentially
            # we'll get wrong
            #if trial_group_col in ['trialwb']:
            #    raise ValueError('not implemented')
            df = getSubDf(df_all, subj, pertv,tgti,env,block_name,
                          non_hit = False, verbose=verbose)
            db_inds = df.index
            #df_inds += [db_inds]

            tgn = getTrialGroupName(pertv, tgti, env, block_name)
            print(f'  selected {len(df)} inds out of {len(df_all) }; tgn = {tgn}')
            #coln = getColn(pertv, tgti, env, block_name, None) #, trial_group_col)
            #print('  ',tgn, coln,len(df))
            if (len(df) == 0) or (len(db_inds) == 0):
                #rowi += 1
                print('skip',tgn,subj)
                #colns_skip += [coln]
                if DEBUG:
                    debug_break = 1
                    break
                continue

            if computation_ver == 'computeErrSens3':
                if df_fulltraj is not None:
                    addargs = {'df_fulltraj': \
                       df_fulltraj.query(f'subject == "{subj}"'),
                       'trajPair2corr':trajPair2corr }
                else:
                    addargs = addargs0
                    addargs['reref_target_locs'] = reref_target_locs
                addargs['long_shift_numerator'] = long_shift_numerator
            else:
                addargs = addargs0


            
            for tsz in trial_shift_sizes:
                for rf in retention_factor:
                    #if tsz == 0:
                    #    escoln = 'err_sens':
                    #else:
                    #    escoln = 'err_sens_-{tsz}t':
                    # resetting index is important
                    if 'level_0' in df.columns:
                        df = df.drop(columns=['level_0'])
                    if 'index' in df.columns:
                        df = df.drop(columns=['index'])
                    dfri = df.reset_index()
                    r = computeES(dfri, df_inds=None,
                        error_type=error_type,
                        colname_nh = coln_nh,
                        correct_hit = 'inf', shiftsz = tsz,
                        err_sens_coln=err_sens_coln,
                        coln_nh_out = coln_nh_out,
                        time_locked = time_locked, addvars=addvars,
                        target_info_type = target_info_type,
                        coln_correction_calc = coln_correction_calc,
                        coln_error = coln_error,
                        recalc_non_hit = False, 
                        retention_factor = float(rf),
                        **addargs)

                    print(f'computation_ver = {computation_ver}. retention_factor={rf} tsz = {tsz}')
                    if computation_ver == 'computeErrSens2':
                        nhna, df_esv, ndf2vn = r
                    elif computation_ver == 'computeErrSens3':
                        ndf2vn = None
                        nhna, df_esv = r

                    # if I don't convert to array then there is an indexing problem
                    # even though I try to work wtih db_inds it assigns elsewhere
                    # (or does not assigne at all)
                    es_vals = np.array( df_esv[err_sens_coln] )
                    assert np.any(~np.isnan(es_vals)), tgn  # at least one is not None
                    assert np.any(~np.isinf(es_vals)), tgn  # at least one is not None

                    #colns_set += [coln]

                    dfcur = df.copy()
                    dfcur['trial_shift_size'] = tsz  # NOT _nh, otherwise different number
                    dfcur['time_locked'] = time_locked  # NOT _nh, otherwise different number
                    dfcur[err_sens_coln] = es_vals  # NOT _nh, otherwise different number
                    dfcur['trial_group_col_calc'] = tgn
                    dfcur['error_type'] = error_type
                    # here it means shfited by 1 within subset
                    dfcur[coln_nh_out] = np.array( df_esv[coln_nh_out] )

                    dfcur['correction'] = np.array( df_esv['correction'] )
                    if 'belief_' in df_esv.columns:
                        dfcur['belief_'] = np.array( df_esv['belief_'] )

                    # copy columns including prev_err_sens
                    for cn in ['trial_inds_glob_prevlike_error', 'trial_inds_glob_nextlike_error',
                               f'prev_{err_sens_coln}', 'prev_error', 
                               'retention_factor', 'retention_factor_s',
                               'vals_for_corr1','vals_for_corr2',
                              'prevlike_error', 'prev_time' ]:
                        if cn not in df_esv.columns:
                            print(f'WARNING: {cn} not in df_esv.columns')
                        else:
                            dfcur[cn] = df_esv[cn].to_numpy()

                    if computation_ver == 'computeErrSens2':
                        errn = ndf2vn['prev_error']
                        dfcur['dist_rad_from_prevtgt2'] = dfcur['target_locs'].values -\
                            df_esv['prev_target'].values
                        dfcur[errn] = np.array( df_esv[errn] )
                    else:
                        dfcur['dist_rad_from_prevtgt2'] =\
                            df_esv['target_loc'].values -\
                            df_esv['prev_target_loc'].values
                        dfcur['dist_rad_from_prevtgt_shiftrespect'] =\
                            df_esv['target_loc'].values -\
                            df_esv['prev_target_loc_shiftrespect'].values

                    for avn in addvars:
                        if avn in dfcur.columns:
                            continue
                        dfcur[avn] = np.array(df_esv[avn])

                    #lbd(0.5)
                    #print(dfcur['target_locs'].values, df_esv['prev_target'].values )
                    #raise ValueError('f')


                    dfs += [ dfcur.reset_index(drop=True)  ]

                if DEBUG and tgn == 'trialwtgt_we':
                    debug_break = 1; print('brk')
                    db_inds_save = db_inds
                    df_save = df
                    break
            if debug_break:
                break
        if debug_break:
            break
        print('Subj = ',subj, 'computation finished successfully')
    print('computeErrSensVersions: Main calc finished successfully')


    df_all2 = pd.concat(dfs)
    df_all2.reset_index(inplace=True, drop=True)
    if drop_feedback and ( 'feedbackX' in df_all2.columns ):
        df_all2.drop(['feedbackX','feedbackY'],axis=1,inplace=True)

    if 'trajectoryX' in df_all2.columns:
        df_all2.drop(['trajectoryX','trajectoryY'],axis=1,inplace=True)

    # convert to string
    lbd = lambda x : f'{x:.2f}'
    df_all2['dist_rad_from_prevtgt2'] =\
        df_all2['dist_rad_from_prevtgt2'].abs().apply(lbd)

    lbd = lambda x : f'{x:.2f}'
    df_all2['dist_rad_from_prevtgt_shiftrespect'] =\
        df_all2['dist_rad_from_prevtgt_shiftrespect'].abs().apply(lbd)


    df_all2.loc[df_all2['trials'] == 0, coln_nh_out] = False
    #df_all2.loc[df_all2['trials'] == 0, 'non_hit_not_adj'] = False
    df_all2.loc[df_all2['trials'] == 0, 'err_sens'] = np.inf
    return df_all2, ndf2vn


def sprintf_tpl_statcalc(tpl):
    ''' used to pring tuples of specific format, for debug'''
    env,bn,pertv,gseqc,tgti,\
        dist_rad_from_prevtgt,dist_trial_from_prevtgt,\
        trial_group_col_calc,trial_group_col_av = tpl
    locs = locals().copy()
    s = ''
    for ln,lv in locs.items():
        if ln == 'tpl':
            continue
        lneff = ln
        st = 'trial_group_col_'
        st2 = 'dist_rad_from_prevtgt'
        st3 = 'dist_trial_from_prevtgt'
        if ln.startswith(st):
            lneff = 'tgc' + ln[len(st):]
        elif ln.startswith(st2):
            lneff = 'drpt' + ln[len(st2):]
        elif ln.startswith(st3):
            lneff = 'dtpt' + ln[len(st3):]
        s += f'{lneff}={lv}; '
    return s[:-2]

def aggRows(df, coln_time, operation, grp = None, coltake='corresp',
            colgrp = 'trial_index' ):
    '''
    Take row with highest/lowest value of coln_time
    coln_time is the column on value of which one wants to aggregate
    '''
    assert coln_time in df.columns
    if grp is None:
        assert colgrp in df.columns
    # coln_time = 'time'
    assert operation in ['min','max']
    from datetime import timedelta
    if coln_time == 'time':
        diffmin = df[coln_time].diff().min()
        if isinstance(diffmin, timedelta):
            assert diffmin.total_seconds() >= 0, diffmin  # need monotnicity
        else:
            assert diffmin >= 0, diffmin  # need monotnicity
    assert coltake is not None

    if grp is None:
        grp = df.groupby(colgrp)
    else:
        if colgrp is not None:
            print('aggRows WARNING: Column ', colgrp, ' is not used due to grp != None')

    if coltake != 'corresp':
        cns = [cn for cn in df.columns if cn != coln_time]
        if operation == 'min':
            coltake = 'first'
        elif operation == 'max':
            coltake = 'last'
        agg_d = dict(zip(cns, len(cns) * [coltake]))
        agg_d[coln_time] = operation
        dfr = grp.agg(agg_d)
    else:
        if operation == 'min':
            idx = grp[coln_time].idxmin()
        elif operation == 'max':
            idx = grp[coln_time].idxmax()
        dfr = df.loc[idx]
    return dfr.sort_values([coln_time])

# make polynomial fits
def plotPolys(ax, dftmp, fitcol, degs=range(2,6), mean=1):
    if mean:
        me = dftmp.groupby(fitcol).median().reset_index()
        dftmp = me
    dftmp[fitcol] = pd.to_numeric(dftmp[fitcol] )
    esv, dv = dftmp[['err_sens',fitcol]]._values.T
    print(np.min(dv),dv,dv-np.min(dv),esv)
    #pr = np.polyfit(esv,dv,2)
    from numpy.linalg import LinAlgError
    dvu = np.unique(dv)
    dvu = np.array( list(sorted(dvu)) )
    print(dvu)
    for deg in degs:
        try:
            pr = np.polyfit(dv-np.min(dv),esv-np.min(esv),deg)
        except (SystemError,LinAlgError):
            print(f'Failed deg={deg}')
            print(dv,esv, np.std(dv))
            continue

        poly = np.poly1d(pr)
        #if len(degs) > 1:
        if mean:
            lbl = f'polynomial fit of means deg={deg}'
        else:
            lbl = f'polynomial fit deg={deg}'
        #else:
        #    lbl = None
        esv2 = poly(dvu-np.min(dvu)) + np.min(esv)
        print(dvu-np.min(dvu), esv)
        ax.plot(range(len(dvu)) , poly(dvu-np.min(dvu)) + np.min(esv),
                label=lbl, c='grey', lw=0.85 )
    return pr
    #ax.legend(loc='lower right')

def getPvals(dftmp, fitcol, pairs, alternative='greater'):
    from scipy.stats import ttest_ind
    pvalues = []
    pair2pv = {}
    for drp in pairs:
        if isinstance(drp[0],str):
            vs1 = dftmp.query(f'{fitcol} == "{drp[0]}"')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == "{drp[1]}"')['err_sens']
        else:
            vs1 = dftmp.query(f'{fitcol} == {drp[0]}')['err_sens']
            vs2 = dftmp.query(f'{fitcol} == {drp[1]}')['err_sens']
        #ttr = ttest_ind(vs1,vs2)

        assert not np.any( np.isnan(vs1) )
        assert not np.any( np.isnan(vs2) )

        ttr = ttest_ind(vs1,vs2, alternative=alternative)
        pvalues += [ttr.pvalue]

        pair2pv[drp] = ttr.pvalue

        print(drp, len(vs1),len(vs2), ttr.pvalue)
    # pvalues = [
    #     sci_stats.mannwhitneyu(robots, flight, alternative="two-sided").pvalue,
    #     sci_stats.mannwhitneyu(flight, sound, alternative="two-sided").pvalue,
    #     sci_stats.mannwhitneyu(robots, sound, alternative="two-sided").pvalue
    # ]

    # pvalues
    # [0.00013485140468088997, 0.2557331102364572, 0.00022985464929005115]

    # Transform each p-value to "p=" in scientific notation
    formatted_pvalues = [f'p={pvalue:.2e}' for pvalue in pvalues]
    return pvalues, formatted_pvalues, pair2pv

def shiftTrials(dfcc_all_sub, dfcc_all, shift=1):
    if isinstance(shift, int):
        shift = [shift]

    rows = []
    for i,row in dfcc_all_sub.iterrows():
        tind = row['trial_index']
        subj = row['subject']

        row0 = dfcc_all.query('subject == @subj and trial_index == @tind')
        
        #row0['_sh'] = 0
        #rows += [row0]
        for sh in shift:
            tind1 = tind + sh
            df_ = dfcc_all.query('subject == @subj and trial_index == @tind1')
            if len(df_):
                row1 = dict( df_.iloc[0] )
            else:
                continue

            row1['_sh'] = sh
            rows += [row1]
            # only mvt phase
            #dfpretraj = dfc_all.query('subject == @subj')
            #grp_perti = dfpretraj.groupby(['trial_index'])
    return pd.DataFrame(rows).sort_values(['trial_index','_sh'])

def myttest(df_, qs1, qs2, varn, alt = ['two-sided','greater','less'], paired=False,
            cols_checkdup = []):
    # cols_checkdup = ['subject','trials'])
    from pingouin import ttest
    ttrs = []
    if isinstance(alt,str):
        alt = [alt]

    try:
        df1 = df_.query(qs1)
    except Exception as e:
        print(f'myttest: Exception {e} for {qs1}')
        raise ValueError(f'bad qs {qs1}')

    try:
        df2 = df_.query(qs2)
    except Exception as e:
        print(f'myttest: Exception {e} for {qs2}')
        raise ValueError(f'bad qs {qs2}')

    assert ( (len(df1) > 0) and (len(df2) > 0) )

    if len(cols_checkdup):
        assert not df1.duplicated(cols_checkdup).any()
        assert not df2.duplicated(cols_checkdup).any()
    for alt_ in alt:
        ttr = ttest(df1[varn].values, 
                    df2[varn].values, alternative=alt_, paired=paired)
        ttrs += [ttr]
    ttrs = pd.concat(ttrs)
    ttrs['paired'] = paired
    ttrs = ttrs.rename(columns={'p-val':'pval'})
    ttrs['varn'] = varn
    ttrs['qs1'] = qs1
    ttrs['qs2'] = qs2
    ttrs['N1'] = len(df1)
    ttrs['N2'] = len(df2)
    return ttrs

def multi_comp_corr(ttrs, method='bonf'):
    from pingouin import multicomp 
    ttrs['pval_uncorr'] = ttrs['pval']
    ttrs['mc_corr_method'] = method
    ttrs['mc_corr_N'] = -1
    alt = ttrs['alternative'].unique()
    if method != 'none':
        assert ttrs.index.is_unique
        for alt_ in alt:
            dfc = ttrs.query('alternative == @alt_')
            pvs = dfc['pval'].values
            reject, pvs2 = multicomp(pvs, method = method)
            #print(pvs, pvs2, len(dfc), ttrs.loc[dfc.index,'pval'] )
            ttrs.loc[dfc.index,'pval'] = pvs2
            ttrs.loc[dfc.index,'mc_corr_N'] = len(pvs)
    return ttrs

def compare0(df, varn, alt=['greater','less'], cols_addstat = []):
    from pingouin import ttest
    '''
        returns ttrs (not only sig)
    '''
    if isinstance(alt,str):
        alt = [alt]
    ttrs = []
    for alt_ in alt: #, 'two-sided']:
        ttr = ttest( df[varn], 0, alternative = alt_ )
        ttr['alt'] = alt_
        ttr['val1'] = varn
        ttr = ttr.rename(columns ={'p-val':'pval'})

        for coln in cols_addstat:
            ttr[coln + '_mean'] = df[coln].mean()
            ttr[coln + '_std' ] = df[coln].std()

        ttrs += [ttr]
    ttrs = pd.concat(ttrs, ignore_index = 1)
    ttrs['N1'] = len(df)
    decorateTtestRest(ttrs)
    return ttrs

def decorateTtestRest(ttrs):
    ttrs['ttstr']  = ''
    def f(row):
        alt = row['alternative']
        v1 = row['val1']
        v2 = row.get('val2', 0)
        if alt == 'greater':
            s = f'{v1} > {v2}'
        elif alt == 'less':
            s = f'{v1} < {v2}'
        elif alt == 'two-sided':
            s = f'{v1} != {v2}'
        return s

    ttrs['ttstr']  = ttrs.apply(f,axis=1)

def pval2starcode(pval):
    c = None
    if pval > 0.05:
        c = 'ns'
    elif pval <= 0.0001:
        c = '****'
    elif pval <= 0.001:
        c = '***'
    elif pval <= 0.01:
        c = '**'
    elif pval <= 0.05:
        c = '*'
    return c

def comparePairs(df_, varn, col, 
                 alt = ['two-sided','greater','less'], paired=False,
                 pooled = 2, updiag = True, qspairs = None, multi_comp_corr_method = 'bonf') -> (pd.DataFrame,pd.DataFrame):
    '''
    returns sig,all
    runs t-tests on all pairs of queries defined in qspairs
    qspairs: when None, just take all unique values of col
    pooled = 2 means do pooled and non-pooled

    bonf,sidak,holm,fdr_bh,fdr_by,none
    '''
    assert isinstance(paired, bool)
    assert len(df_), 'Given empty dataset'
    assert varn is not None, 'varn cannot be None'
    ttrs = []
    if int(pooled) == 1:
        ttrs = comparePairs_(df_,varn,col, pooled=True, 
                             alt=alt, paired=paired, qspairs = qspairs)
        ttrs += [ttrs]
    if (int(pooled) == 0) or (pooled == 2):
        ttrs_np = comparePairs_(df_,varn,col, pooled=False, 
            alt=alt, paired=paired, updiag = updiag, qspairs = qspairs)
        ttrs += [ttrs_np]
    ttrs = pd.concat(ttrs, ignore_index=1)
     
    ttrs = multi_comp_corr(ttrs, multi_comp_corr_method)
    #ttrs['pval_uncorr'] = ttrs['pval']
    #ttrs['mc_corr_method'] = multi_comp_corr_method
    #if multi_comp_corr_method != 'none':
    #    for alt_ in alt:
    #        dfc = ttrs.query('alternative == @alt_')
    #        pvs = dfc['pval'].values
    #        reject, pvs2 = multicomp(pvs, method = multi_comp_corr_method)
    #        ttrs.loc[dfc.index,'pval'] = pvs2
        

    ttrs = addStarcodes(ttrs)
    ttrssig = ttrs.query('pval <= 0.05').copy()

    if len(ttrssig) == 0:
        return None,ttrs

    decorateTtestRest(ttrs)
    decorateTtestRest(ttrssig)
    return  ttrssig, ttrs

def addStarcodes(ttrs):
    ttrs.loc[ ttrs['pval'] > 0.05   , 'starcode'] = 'ns'
    ttrs.loc[ ttrs['pval'] <= 0.05  , 'starcode'] = '*'
    ttrs.loc[ ttrs['pval'] <= 0.01  , 'starcode'] = '**'
    ttrs.loc[ ttrs['pval'] <= 0.001 , 'starcode'] = '***'
    ttrs.loc[ ttrs['pval'] <= 0.0001, 'starcode'] = '****'
    return ttrs

def comparePairs_(df_, varn, col, pooled=True , alt=  ['two-sided','greater','less'], paired=False, updiag = True, qspairs = None):
    '''
    all upper diag pairs of col values
    runs t-tests on all pairs of queries defined in qspairs
    '''
    from bmp_behav_proc import myttest
    assert len(df_)
    assert varn is not None, 'varn cannot be None'

    ttrs = []

    if isinstance(col, (list,np.ndarray) ):
        cols = col
    else:
        cols = [col]

    if not pooled:
        if col is not None:
            s1 = ['subject'] + cols +  [varn]
            s2 = ['subject'] +  cols
        else:
            s1 = ['subject',  varn]
            s2 = ['subject' ]
        try:
            df_ = df_[s1].groupby(s2,observed=True).mean(numeric_only=1).reset_index()
        except KeyError as e:
            print(f'KeyError: {e} for {s1} and {s2}')
            raise ValueError(f'Bad columns {s1} or {s2} in df_')

    #print(df_.groupby()

    #colvals = colvals[~np.isnan(colvals)]
    if qspairs is None:
        colvals = df_[col].unique()
        for cvi,cv in enumerate(colvals):
            #vals1 = df.query('@col == @cv')
            if updiag:
                loop2 = enumerate(colvals[cvi+1:])
            else:
                loop2 = enumerate(colvals)
            for cvj,cv2 in loop2:
                # need if not updiag
                if cv == cv2:
                    continue

                #vals2 = df.query('@col == @cv2')
                cv_ = cv
                cv2_ = cv2
                if isinstance(cv,str):
                    cv_ = '"' + cv + '"'
                    cv2_ = '"' + cv2 + '"'
                qs1 = f'{col} == {cv_}'
                qs2 = f'{col} == {cv2_}'
                ttrs_ = myttest(df_,qs1, qs2, varn, alt=alt, paired=paired)
                ttrs_['val1'] = cv
                ttrs_['val2'] = cv2
                ttrs += [ttrs_]
    else:
        for qs1,qs2 in qspairs:
            ttrs_ = myttest(df_,qs1, qs2, varn, alt=alt, paired=paired)
            ttrs_['val1'] = qs1
            ttrs_['val2'] = qs2
            ttrs_['val1_parsed'] = qs1.split('=')[1].strip()
            ttrs_['val2_parsed'] = qs2.split('=')[1].strip()
            ttrs += [ttrs_]

    ttrs = pd.concat(ttrs, ignore_index=1)
    ttrs['pooled'] = pooled
    return ttrs
 
def calcESthr(df, mult):
    assert not np.isinf(df['err_sens']).any()
    dfni = df                                           
    dfni_d = dfni.groupby(['subject'],observed=True)\
        ['err_sens'].describe().reset_index()
    ES_thr = dfni_d[dfni_d.columns[1:]].mean().to_dict()['std'] * mult
    #ES_thr_single = ES_thr
    return ES_thr

def truncateNIHDfFromES(df_wthr, mult, ES_thr=None):
    # remove trials with error > std_mult * std of error
    std_mult = mult

    dfni = df_wthr[~np.isinf(df_wthr['err_sens'])]
    if ES_thr is None:
        ES_thr = calcESthr(dfni, mult)
        print(f'ES_thr (recalced) = {ES_thr}')

    dfni_g = dfni.query('err_sens.abs() <= @ES_thr')
    nremoved_pooled = len(dfni) - len(dfni_g)

    sz = dfni.groupby(['subject'],observed=True).size()
    sz_g = dfni_g.groupby(['subject'],observed=True).size()
    mpct = ((sz - sz_g) / sz).mean() * 100
    print(f'Mean percentage of removed trials = {mpct:.3f}%, '
          f'pooled = {nremoved_pooled / len(dfni) * 100:.3f}%')

    dfall = dfni_g.copy()
    dfall['thr'] = "mestd*0" # just for compat
    return dfall

def checkErrBounds(df, cols=['error','prev_error','correction',
                             'error_deg','prev_error_deg','belief','perturbation' ]):
    # ,'target_locs' can be > Pi because they are 90 + smth, with smth \in [0,pi]
    # not feedback and org_feedback or target_locs because they are referenced not to 0 in NIH data
    bd = np.pi 
    bd_deg = 180
    badcols = []
    badcols_tuples = []
    for col in cols:
        if col in df.columns:
            mx = df[col].abs().max()
            if col.endswith('deg') or col == 'perturbation':                
                if  mx > bd_deg:
                    badcols += [col]
                    badcols_tuples += [(col,mx, np.sum(df[col].abs() > bd_deg) )]

            else:
                if mx > bd:
                    badcols += [col]
                    #badcols_tuples += [(col,mx)]
                    badcols_tuples += [(col,mx, np.sum(df[col].abs() > bd) )]
    print('checkErrBounds: Bad columns found: (colname, max abs val, count)', badcols_tuples)
    return badcols

def adjustErrBoundsPi(df, cols):
    bd = np.pi 
    bd_deg = 180
    for col in cols:
        mx = df[col].abs().max()
        if  mx > bd_deg:
            bd_eff =  bd_deg
        else:
            bd_eff =  bd
        print('adjustErrBoundsPi', col, bd_eff)
            
        def f(x):    
            if x > bd_eff:
                x -= bd_eff * 2
            elif x < -bd_eff:
                x += bd_eff * 2
            return x
        df[col] = df[col].apply(f)


#####################

def corrMean(dfallst, coltocorr = 'trialwpertstage_wb', 
             stagecol = 'pert_stage_wb', 
             coln = 'err_sens', method = 'pearson', covar = None):
    '''
    compute correlation with p-value within subject and also mean across
    does it within condition defined by stagecol

    returns:
        tuple of correlation dataframes
        first is mean within across subjects, second with separate subjects
        
    '''
    # corr or partial correlation within participant, averaged across participants
    import pingouin as pg
    assert coltocorr in dfallst.columns
    assert stagecol in dfallst.columns
    assert coln in dfallst.columns

    def f(df_):
        try:
            if covar is None:
                r = pg.corr( df_[coltocorr], df_[coln],  method=method)
            else:
                r = pg.partial_corr( df_, coltocorr, coln, covar, method=method)
            r['method'] = method
            r['mean_x'] = df_[coltocorr].mean()
            r['mean_y'] = df_[coln].mean()
            r['std_x'] = df_[coltocorr].std()
            r['std_y'] = df_[coln].std()
        except ValueError as e:
            return None
        return r

    groupcols0 = []
    if 'thr' in dfallst.columns:
        groupcols0 = ['thr'] 
    groupcols = groupcols0 + [stagecol]
    groupcols2 = groupcols0 + ['subject', stagecol]

    # separate subjects
    corrs_per_subj_me0 = dfallst.groupby(groupcols2, observed=True).apply(f)
    corrs_per_subj_me0['method'] = method
    

    # mean over subjects
    corrs_per_subj_me = corrs_per_subj_me0.rename(columns={'p-val':'pval'})
    corrs_per_subj_me = corrs_per_subj_me.\
        groupby(groupcols, observed=True)[['r','pval',
            'mean_x','mean_y','std_x','std_y']].mean(numeric_only = 1)
    corrs_per_subj_me['method'] = method
    corrs_per_subj_me['varn'] = covar

    return corrs_per_subj_me, corrs_per_subj_me0


def formatRecentStatVarnames(isec, histlen_str=' (histlen='):
    '''
    takes list of varnames, outputs list of nice varnames
    '''
    isec_nice = []
    for s in isec:
        s2 = s.replace('error_pscadj_abs','Error magnitude')\
            .replace('error_pscadj','Signed error')\
            .replace('error','Signed error')\
            .replace('_Tan',' mean^2/var' + histlen_str)\
            .replace('_mavsq',' mean^2' + histlen_str)\
            .replace('_invstd',' 1/std' + histlen_str)\
            .replace('_invmav',' 1/mean' + histlen_str)\
            .replace('_mav_d_std',' mean/std' + histlen_str)\
            .replace('_std_d_mav',' std/mean' + histlen_str)\
            .replace('_std',' std' + histlen_str)\
            .replace('_mav_d_var',' mean/std' + histlen_str) 
        if len(histlen_str):
            s2 += ')' 
        isec_nice.append(s2 )
    return isec_nice

def checkSavingsNIH(dfall, method = 'spearman' ):
    s1,s2 = set(['pert_stage','err_sens','trial_index','pert_stage']), set(dfall.columns) 
    assert s1 < s2, ( s1 - s2 )
    cols_ttrs = ['qs1','qs2','alternative','T','pval','mc_corr_N','dof','paired']

    print('Correlation computation method = ',method)
    corrs_per_subj_me_,corrs_per_subj  = corrMean(dfall, 
                stagecol = 'pert_stage', coln='err_sens' ,method=method)

    # show stat signif
    stage_pairs = [(1,6),(3,8)]
    ttrs2 = []

    lst1 = stage_pairs[0]    
    lst2 = stage_pairs[1]    

    df_ = corrs_per_subj.reset_index().query('pert_stage.isin(@lst1) or pert_stage.isin(@lst2)')
    ttrs_sig, ttrs2 = comparePairs(df_, 'r', 'pert_stage', 
        qspairs=[(f'pert_stage == {stage_pairs[0][0]}', f'pert_stage == {stage_pairs[0][1]}'),
                  (f'pert_stage == {stage_pairs[1][0]}', f'pert_stage == {stage_pairs[1][1]}') ], 
        paired=True)

    print(f'Significant statistical difference between ES slopes (comparing pairs of stages {stage_pairs}):')
    ttrs2_sig = ttrs2.query('pval <= 5e-2')
    if len(ttrs2_sig):
        display( ttrs2_sig[cols_ttrs] )
    else:
        print('No significant differences found for savings pairs')

    ###########################

    stage_pairs_nice = {"1-6":'first and last', "3-8":'second and third'}
    
    some = False
    for irow,row in ttrs2.query('alternative == "two-sided"').iterrows():
        #sp = row['stage_pair']
        qs1,qs2 = row['qs1'], row['qs2']
        sp = f'{qs1.split("==")[1].strip()}-{qs2.split("==")[1].strip()}'

        pv=row['pval']
        T=row['T']

        s = ''
        if pv > 0.05:
            s = 'not '
        else:
            some |= True
        #print(sp,pv)
        print('ES during {} perturbations are {}significantly different, t={:.2f}, p-value = {:.2e}.'.\
                  format(stage_pairs_nice[sp],s,T,pv) )
    if not some:
        print(f'\n\nNo savings (we have used {method}) !')

    ##################   let's check for other two pairs as well (not related to savings)
    stage_pairs = [(1,3),(6,8)]
    print(stage_pairs)
    lst1 = stage_pairs[0]    
    lst2 = stage_pairs[1]    
    df_ = corrs_per_subj.reset_index().query('pert_stage.isin(@lst1) or pert_stage.isin(@lst2)')
    #ttrs2 = []
    ttrs_sig, ttrs2 = comparePairs(df_, 'r', 'pert_stage', 
        qspairs=[(f'pert_stage == {stage_pairs[0][0]}', f'pert_stage == {stage_pairs[0][1]}'),
                  (f'pert_stage == {stage_pairs[1][0]}', f'pert_stage == {stage_pairs[1][1]}') ], 
        paired=True)
    print(f'Significant statistical difference between other ES slopes (comparing pairs of stages {stage_pairs}):')
    display( ttrs2.query('pval <= 5e-2')[cols_ttrs] )#