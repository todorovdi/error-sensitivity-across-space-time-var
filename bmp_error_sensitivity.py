import numpy as np
import pandas as pd
import warnings
from bmp_base import (calc_target_coordinates_centered, subAngles)
from bmp_config import env2envcode, env2subtr
from bmp_config import target_angs, stage2evn2event_ids

target_coords = calc_target_coordinates_centered(target_angs)

def getPrev(vals, defval = np.nan, shiftsz=1):
    ins = [ defval ] * shiftsz
    outvals = np.insert(vals, 0, ins)[:-shiftsz]
    return outvals

def getIndShifts(trial_inds, time_locked='target', shiftsz=1):
    ''' can be both global inds and local inds'''
    insind = [ -1000000 ] * shiftsz

    valid_inds_cur  = np.ones( len(trial_inds), dtype=bool )
    valid_inds_next = np.ones( len(trial_inds), dtype=bool )
    if time_locked in ['target', 'trial_end']:
        # insert in the beginning, i.e. shift right, i.e.
        # if trial_inds_cur[i] = trial_inds[i-1]
        trial_inds_cur = np.insert(trial_inds, 0, insind)[:-shiftsz]
        trial_inds_next      = trial_inds
    elif time_locked == 'feedback':
        trial_inds_cur       = trial_inds
        # insert in the end
        trial_inds_next      = np.insert(trial_inds, len(trial_inds), insind)[shiftsz:]
        # if trial_inds_next[i] = trial_inds[i+1]
    else:
        raise ValueError(time_locked)

    valid_inds_cur[trial_inds_cur < 0] = False
    valid_inds_next[trial_inds_next < 0] = False

    return  trial_inds_cur, valid_inds_cur, trial_inds_next, valid_inds_next

def shiftVals(vals, trial_inds_cur, valid_inds_cur,
              invalid_val = np.nan):
    '''
    only applies indices (only valid ones)
    '''
    assert len(vals) == len(trial_inds_cur)
    vals_cur  = np.ones(len(vals) ) * invalid_val
    vals_cur[valid_inds_cur] = vals[trial_inds_cur[valid_inds_cur] ]

    return vals_cur

def computeErrSens3(behav_df, df_inds=None, epochs=None,
                    do_delete_trials=1,
                    time_locked='target',
                    correct_hit = 'inf', error_type = 'MPE',
                    colname_nh = 'non_hit_not_adj',
                    coln_nh_out = None,
                    shiftsz=1,
                    err_sens_coln = 'err_sens',
                    addvars = [], recalc_non_hit = True, target_info_type = 'inds',
                    coln_correction_calc = None,
                    coln_error = 'error',
                    df_fulltraj = None,
                    trajPair2corr = None,
                    verbose = 0, use_sub_angles = 0, retention_factor = 1.,
                    reref_target_locs = True,
                   long_shift_numerator = False ):
    '''
    computes error sensitiviy across dataset. So indexing is very important here.
    '''

    assert shiftsz >= 1
    # in read_beahav we compute
    # errors.append(feedback[trial] - target_locs[trial])
    # target_locs being target angles

    assert correct_hit in ['prev_valid', 'zero', 'inf', 'nan' ]
    # modifies behav_df in place

    if df_inds is None:
        df_inds = behav_df.index

    dis = np.diff(behav_df.index.values)

    if error_type == 'MPE':
        errors0       = np.array(behav_df.loc[df_inds, coln_error])
    else:
        raise ValueError(f'error_type={error_type} not implemnted')

    if recalc_non_hit or (colname_nh not in behav_df.columns):
        raise ValueError(f'{colname_nh} is not present and recalc_non_hit not implemneted')

    nonhit = behav_df[colname_nh].to_numpy()

    # not that non-hit has different effect on error sens calc depending on
    # which time_locked is used
    #############################
    tind_coln = 'trials'

    trial_inds_glob = np.array( behav_df.loc[df_inds, tind_coln])

    # replicating getIndShifts here because it's easier than change there
    if (df_fulltraj is not None) and shiftsz == 1:

        prev_trial_inds_glob = np.array( behav_df.loc[df_inds, 'prev_trial_index_valid'])

        if time_locked in ['target', 'trial_end']:
            print('Using prev_trial_index_valid')
            trial_inds_next      = trial_inds_glob
            valid_inds_cur  = np.ones( len(trial_inds_glob), dtype=bool )
            valid_inds_next = np.ones( len(trial_inds_glob), dtype=bool )
            valid_inds_next[trial_inds_next < 0] = False

            trial_inds_cur = prev_trial_inds_glob
            trial_inds_cur[np.isnan(prev_trial_inds_glob)  ] = -1e6
            trial_inds_cur = trial_inds_cur.astype(int)

            valid_inds_next[trial_inds_cur < 0] = False
            valid_inds_next[trial_inds_next < 0] = False

            trial_inds_glob1, valid_inds1, trial_inds_glob2, valid_inds2 = \
                trial_inds_cur, valid_inds_cur, trial_inds_next, valid_inds_next
        else:
            raise ValueError('not impl')
    else:
        trial_inds_glob1, valid_inds1, trial_inds_glob2, valid_inds2 = \
                getIndShifts(trial_inds_glob, time_locked=time_locked, shiftsz=shiftsz)

    trial_inds_loc0 = np.arange(len(trial_inds_glob ) )

    # suffix 1 is for those in the relative past and 2 for those in the relative future
    trial_inds_loc1, valid_inds1, trial_inds_loc2, valid_inds2 = \
        getIndShifts(trial_inds_loc0, time_locked=time_locked, shiftsz=shiftsz)
    trial_inds_loc1_s1, valid_inds1_s1, _, _ = \
        getIndShifts(trial_inds_loc0, time_locked=time_locked, shiftsz=1)

    time = np.array(behav_df.loc[df_inds,'time']).copy()
    target_locs  = np.array(behav_df.loc[df_inds,'target_locs']).copy()
    if reref_target_locs:
        print('Reref target locs')
        target_locs -= np.pi
    movement      = np.array(behav_df.loc[df_inds, 'org_feedback'])

    # which values will be in the numerator of the ES  
    if coln_correction_calc is None:
        #correction = (target_angs2 - next_movement) - (target_angs - movement)
        # -belief
        if use_sub_angles:
            vals_for_corr = subAngles(target_locs, movement)
        else:
            vals_for_corr = target_locs - movement
    else:
        vals_for_corr = behav_df.loc[df_inds, coln_correction_calc].to_numpy()

    # if true then we compute how error changes between now and far past changes wrt error in the far past, which is strange
    # if false, then we compute how error changes between now and PREVIOUS trials wrt error in the far past 
    #print(f'{long_shift_numerator=}, {shiftsz=}')
    if long_shift_numerator:
        vals_for_corr1  = shiftVals(vals_for_corr,
                                   trial_inds_loc1, valid_inds1)
    else:
        vals_for_corr1  = shiftVals(vals_for_corr,
                                   trial_inds_loc1_s1, valid_inds1_s1)

    vals_for_corr2  = shiftVals(vals_for_corr,
                        trial_inds_loc2, valid_inds2)


    errors1  = shiftVals(errors0, trial_inds_loc1, valid_inds1)
    # this should not NOT _s1
    errors2  = shiftVals(errors0, trial_inds_loc2, valid_inds2)

    nonhit_err1_compat = shiftVals(nonhit,
                            trial_inds_loc1, valid_inds1,
                                 invalid_val = False)
    nonhit_err2_compat = shiftVals(nonhit,
                            trial_inds_loc2, valid_inds2,
                                 invalid_val = False)

    targets_locs1  = shiftVals(target_locs,
                            trial_inds_loc1, valid_inds1)
    targets_locs2 = shiftVals(target_locs,
                                   trial_inds_loc2, valid_inds2)

    # it HAS to be target-centered here,
    # otherwise multiplying by retention factor is bad
    # so assuming it is ofb - target
    if use_sub_angles:
        correction = subAngles(retention_factor * vals_for_corr1, vals_for_corr2)
    else:
        correction =  retention_factor * vals_for_corr1 -  vals_for_corr2

    df_esv = {}
    df_esv['trial_inds_glob'] = trial_inds_glob
    df_esv['trial_inds_glob_prevlike_error'] = trial_inds_glob1
    df_esv['trial_inds_glob_nextlike_error'] = trial_inds_glob2
    df_esv = pd.DataFrame( df_esv )

    if len(addvars):
        for vn in addvars:
            if vn.startswith('prev_'):
                valn = vn[5:]
                vals = behav_df.loc[df_inds, valn ]
                vals = shiftVals(vals, trial_inds_loc1, valid_inds1)
            elif vn.startswith('next_'):
                valn = vn[5:]
                vals = behav_df.loc[df_inds, valn ]
                vals = shiftVals(vals, trial_inds_loc2, valid_inds2)
            else:
                valn = vn
                print(f'computeErrSens3: boring add variable {vn}')
                vals = behav_df.loc[df_inds, valn ]

            df_esv[vn] = vals

    df_esv[colname_nh] = nonhit.astype(bool)
    if coln_nh_out is None:
        coln_nh_out = colname_nh + '_shifted'

    # trial end means that we use all the info from the current trial that is
    # available at the end
    if time_locked == 'trial_end':
        nonhit_err_compat = nonhit_err2_compat  # current trial error
    elif time_locked == 'target':
        nonhit_err_compat = nonhit_err1_compat  # prev error
    elif time_locked == 'feedback':
        nonhit_err_compat = nonhit_err1_compat  # current trial error

    df_esv[coln_nh_out] = nonhit_err_compat.astype(bool)

    df_esv[err_sens_coln] = np.nan
    c = df_esv[coln_nh_out]

    if time_locked == 'trial_end':
        errors_denom = errors2
    else:
        errors_denom = errors1

    df_esv.loc[c,err_sens_coln]  = (correction[c] / errors_denom[c])

    df_esv['retention_factor'] = retention_factor
    df_esv['retention_factor_s'] = df_esv['retention_factor'].apply(lambda x: f'{x:.3f}')

    nh = np.sum( ~df_esv[coln_nh_out] )
    if correct_hit == 'prev_valid':
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.inf
        hit_inds = np.where(~df_esv[coln_nh_out] )[0]
        for hiti in hit_inds:
            prev = df_esv.loc[ :hiti, err_sens_coln ]
            good = np.where( ~ (np.isinf( prev ) | np.isnan(prev) ) )[0]
            if len(good):
                lastgood = good[-1]
                df_esv.loc[ hiti, err_sens_coln ] = df_esv.loc[ lastgood, err_sens_coln ]
        #df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  =
    elif correct_hit == 'zero':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = 0
    elif correct_hit == 'inf':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.inf
    elif correct_hit == 'nan':
        if verbose:
            print(f'correct_hit == {correct_hit}: setting {nh} out of {len(df_esv)}')
        df_esv.loc[~df_esv[coln_nh_out],err_sens_coln]  = np.nan

    #raise ValueError('debug')
    df_esv['correction'] = correction
    df_esv['error_type'] = error_type

    # '_like' means that it is not necessarily stricly prev/next and depends on
    # time_locked
    df_esv['prevlike_error'] = errors1
    df_esv['nextlike_error'] = errors2

    df_esv['prevlike_target_loc'] = targets_locs1
    df_esv['nextlike_target_loc'] = targets_locs2

    # for decoding later
    df_esv['belief_']      = -vals_for_corr
    # corr = 1-2
    df_esv['vals_for_corr1']      = vals_for_corr1
    df_esv['vals_for_corr2']      = vals_for_corr2
    # this should be set here (not in behav_proc)
    # because it depends on the coln_corr_calc
    df_esv['prev_belief_'] = -getPrev(vals_for_corr.astype(float) )
    df_esv['prev_movement'] = getPrev(movement.astype(float))
    #####

    df_esv['environment']  = np.array( behav_df.loc[df_inds, 'environment'] )
    df_esv['perturbation'] = np.array( behav_df.loc[df_inds, 'perturbation'])
    df_esv['prev_error']      = getPrev(errors0) # errors0 is unshifted
    df_esv['prev_error_shiftrespect']      = getPrev(errors0, shiftsz = shiftsz) # errors0 is unshifted
    df_esv['target_loc']      = target_locs.astype(float)
    df_esv['prev_target_loc']              = getPrev(target_locs.astype(float) )
    df_esv['prev_target_loc_shiftrespect'] = getPrev(target_locs.astype(float), shiftsz=shiftsz )

    df_esv['prev_time']              = getPrev(time.astype(float) )

    # always ES from immediately preceding trial (even for larger shifts)
    df_esv[f'prev_{err_sens_coln}'] = getPrev( df_esv[err_sens_coln].to_numpy() )

    #raise ValueError('debug')

    return nonhit, df_esv
    # return a mask
    # some adjustment of non_hit, based on time lock and whether we are in
    # stable or random environment
    non_hit = non_hit.copy()
    from bmp_config import n_trials_in_block as N
    if env == 'all':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[N*4 - 1] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[0] = False  # Removing first trial of each block
    elif env == 'stable':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    elif env == 'random':
        if time_locked == 'feedback':
            # remove trials preceding hit (because no next error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, len(non_hit), 1)[1:])
            non_hit[[N-1, N*2-1]] = False  # Removing last trial of each block
        elif time_locked == 'target':
            # remove trials following hit (because no previous error)
            non_hit = ~(~non_hit | ~np.insert(non_hit, 0, 1)[:-1])
            non_hit[[0, N]] = False  # Removing first trial of each block
    return non_hit