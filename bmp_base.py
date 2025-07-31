import numpy as np
import os.path as op
import math

# for NIH data
width = 800  # need to match the screen size during the task
height = 800
# this is radius on which NIH targets appear
radius = int(round(height*0.5*0.8))
radius_target = 12
radius_cursor = 8

from bmp_config import target_angs

def int_to_unicode(array):
    return ''.join([str(chr(int(ii))) for ii in array])


def bincount(a):
    """Count the number of each different values in a."""
    y = np.bincount(a)
    ii = np.nonzero(y)[0]
    return np.vstack((ii, y[ii])).T

def point_in_circle_single(target_ind, target_coords, feedbackX,
                    feedbackY, circle_radius):
    non_hit = list()
    d = math.sqrt(math.pow(target_coords[target_ind][0]-feedbackX, 2) +
                  math.pow(target_coords[target_ind][1]-feedbackY, 2))
    if d > circle_radius:
        non_hit = True
    else:
        non_hit = False
    return non_hit

# for each dim tell whether we hit target or not
def point_in_circle(targets, target_coords, feedbackX,
                    feedbackY, circle_radius, target_info_type = 'inds'):
    non_hit = list()
    for ii in range(len(targets)):
        if target_info_type == 'inds':
            tgtloc = target_coords[targets[ii]]
        elif target_info_type == 'locs':
            tgtloc = targets[ii]
        else:
            raise ValueError('Wrong target info type')
        d = math.sqrt(math.pow(tgtloc[0]-feedbackX[ii], 2) +
                      math.pow(tgtloc[1]-feedbackY[ii], 2))
        if d > circle_radius:
            non_hit.append(True)
        else:
            non_hit.append(False)
    return non_hit

def init_target_positions():
    # height and width of the screen = 600
    # radius of the invisible boundary = 240
    targetAngs = [22.5+180, 67.5+180, 112.5+180, 157.5+180]
    target_types = []
    for x in range(0, len(targetAngs)):
        current = targetAngs[x]*(np.pi/180)
        target_types.append((int(round(600/2.0 +
                                       np.cos(current) * 240)),
                             int(round(600/2.0 +
                                       np.sin(current) * 240))))
    return target_types

def calc_target_coordinates_centered(target_angs):
    target_coords = list()
    for x in range(0, len(target_angs)):
        rad_ang = (target_angs[x]-(90*np.pi/180))
        target_coords.append([int(round(np.cos(rad_ang) * radius)),
                              int(round(np.sin(rad_ang) * radius))])
    return target_coords

def calc_rad_angle_from_coordinates(X, Y, radius_ = None):
    '''
    angle counting from bottom direction CCW (i.e. right)
    so 1,0 gives 90
    '''
    if radius_ is None:
        radius_cur = radius  # global var defined in the beg, distance home to target
    else:
        radius_cur = radius_

    angles = np.arctan2(Y/float(radius_cur),
                        X/float(radius_cur)) # [-pi,pi]
    # change the 0 angle (0 is now bottom vertical in the circle)
    angles = angles + np.pi/2. 
    # make the angle between 0 and np.pi

    c = angles < 0
    angles[c] = angles[c] + 2*np.pi
    c = angles > np.pi
    angles[c] = angles[c] - 2*np.pi

    #for i in np.where(angles < 0):
    #    angles[i] = angles[i] + 2*np.pi
    #for i in np.where(angles > np.pi):
    #    angles[i] = angles[i] - 2*np.pi
    return angles

def rot(xs,ys, ang=20. * np.pi / 180., startpt =(0.,0.) ):
    # ang is in radians
    xs = np.array(xs, dtype = float) - startpt[0]
    ys = np.array(ys, dtype = float) - startpt[1]
    assert ang < np.pi + 1e-5, ang
    xs2 = xs * np.cos(ang) - ys * np.sin(ang)
    ys2 = xs * np.sin(ang) + ys * np.cos(ang)

    xs2 += startpt[0]
    ys2 += startpt[1]
    return np.array( [xs2, ys2])

def subAngles(ang1, ang2):
    # angles should be in radians
    import pandas as pd
    if isinstance(ang1, pd.Series):
        ang1 = ang1.values
    if isinstance(ang2, pd.Series):
        ang2 = ang2.values
    r = np.exp(ang1 * 1j) * np.exp(-ang2 * 1j)
    return np.log(r).imag

def assert_len_equal(a1,a2):
    assert len(a1) == len(a2), (len(a1),len(a2))

def get_lmm_pseudo_r_squared(model_results):
    """
    Calculates Nakagawa & Schielzeth's R-squared for LMMs.
    R²m: Marginal R-squared (variance explained by fixed effects)
    R²c: Conditional R-squared (variance explained by fixed and random effects)

    Args:
        model_results: A fitted MixedLMResults object from statsmodels.

    Returns:
        A dictionary with 'R2_marginal' and 'R2_conditional'.
    """
    import pandas as pd
    
    # Variance of fixed effects component
    # This is Var(X*beta) where X is the design matrix for fixed effects
    # and beta are the fixed effects coefficients.
    var_f = np.var(np.dot(model_results.model.exog, model_results.fe_params))

    # Variance of random effects components
    # model_results.cov_re gives the variance-covariance matrix of random effects.
    # We sum the variances (diagonal elements).
    # If model_results.cov_re is scalar (e.g. for a single random intercept from older statsmodels or simpler cases)
    if np.isscalar(model_results.cov_re):
        var_r = model_results.cov_re
    elif isinstance(model_results.cov_re, (np.ndarray, pd.Series)) and model_results.cov_re.ndim <= 1:
        # If it's a 1D array (e.g., just variances, no covariances shown directly)
        # or a scalar that became a 0-dim array
        var_r = np.sum(model_results.cov_re) # Sum if multiple variance components, or just the value if one
    else: # It's a 2D matrix (e.g. for random slopes or multiple random effects)
        var_r = np.sum(np.diag(model_results.cov_re))


    # Residual variance (sigma_epsilon^2)
    # In statsmodels, this is referred to as 'scale'
    var_e = model_results.scale

    # Total variance
    var_total = var_f + var_r + var_e
    
    if var_total == 0: # Avoid division by zero
        return {"R2_marginal": 0, "R2_conditional": 0}

    # Marginal R-squared (variance explained by fixed effects)
    r2_marginal = var_f / var_total

    # Conditional R-squared (variance explained by fixed and random effects)
    r2_conditional = (var_f + var_r) / var_total

    return {"R2_marginal": r2_marginal, "R2_conditional": r2_conditional}

# Define the function to be executed in parallel
def run_linear_mixed_model(args, ret_res = False, inc_prev = True, n_jobs_inside = 1,
                           all_formulas = True, target_var = 'err_sens'):
    '''
    ret_res: if True, return the results of all models (but it does not work well for multiprocessing)
    '''
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning    
    from statsmodels.stats.diagnostic import lilliefors
    from statsmodels.stats.stattools import jarque_bera
    import traceback
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from numpy.linalg import LinAlgError

    dfcs_fixhistlen, cocoln, std_mavsz_, varn0, varn_suffix, transform = args
    varn = f'{varn0}_{varn_suffix}{std_mavsz_}'
    subset = [varn, target_var]
    if inc_prev:
        subset += ['prev_error_pscadj_abs']
    df_ = dfcs_fixhistlen.dropna(subset=subset)
    df_ = df_[~np.isinf(df_[varn])]
    df_ = df_[~np.isinf(-df_[varn])]
    df_ = df_[~np.isinf(df_[target_var])]
    df_ = df_[~np.isinf(-df_[target_var])]

    varn_eff = varn
    if transform == 'log':
        print(f'Using log transform for {varn}')
        if np.any(df_[varn] <= 0):
            print(f'Warning: {varn} has non-positive values, using log_abs transformation')
            varn_eff = f'log_abs_{varn}'
            df_[varn_eff] = np.log(1e-8 + df_[varn].abs())
        else:
            varn_eff = f'log_{varn}'
            df_[varn_eff] = np.log(df_[varn])
    elif transform == 'BoxCox':
        varn_eff = f'BoxCox_{varn}'
        from scipy import stats
        df_[f'BoxCox_{varn}'], best_lambda_BoxCox = stats.boxcox(0.1 -df_[varn].min() + df_[varn])
    print(varn_eff)

    assert len(df_) > 0, f'No data for {varn} and {cocoln}'

    excfmt = None
    nstarts = 1
    result = None
    if cocoln == 'None':
        s,s2 = f"err_sens ~ {varn_eff}","1"
        model = smf.mixedlm(s, df_, groups=df_["subject"])
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
        if all_formulas:
            s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff} + C({cocoln}) * {varn_eff} + {varn_eff} * prev_error_pscadj_abs",\
                f"~C({cocoln})"; flas += [(s,s2)]
            s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff} + C({cocoln}) * {varn_eff} + {varn_eff} * prev_error_pscadj_abs",\
                f"1"; flas += [(s,s2)]
            s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff} + C({cocoln}) * {varn_eff}", f"~C({cocoln})"; flas += [(s,s2)]
            s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff} + C({cocoln}) * {varn_eff}","1";  flas += [(s,s2)]

            s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff}",f"~C({cocoln})"; flas += [(s,s2)]
        s,s2 = f"{target_var} ~ C({cocoln}) + {varn_eff}","1"; flas += [(s,s2)]

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
            # if debug:
            #     print('result',summary)
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
                summary.lilliefors_test_pv = p_value  # small p-values mean NOT normally distributed
                summary.jarque_bera_test_st = jb_stat
                summary.jarque_bera_test_pv = jb_p_value  # small p-values mean NOT normally distributed
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
            'varn': varn_eff, 'varn0':varn0, 'varn_suffix':varn_suffix,             
             'excfmt':excfmt, 'transform': transform,
            's2summary': s2summary, 'retention_factor':df_.iloc[0]['retention_factor_s']}
    if transform == 'BoxCox':
        r['best_lambda_BoxCox'] = best_lambda_BoxCox
            #'res': result}
            #'nstarts':nstarts,
    if ret_res:
        r['s2res'] = results
    return r

def find_continuous_streaks(numbers):
    #numbers = sorted([int(n) for n in numbers_str])

    if numbers is None:
        return []

    streaks = []
    current_streak_start = numbers[0]
    current_streak_end = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] == current_streak_end + 1:
            current_streak_end = numbers[i]
        else:
            if current_streak_start == current_streak_end:
                streaks.append(str(current_streak_start))
            else:
                streaks.append(f"{current_streak_start}-{current_streak_end}")
            current_streak_start = numbers[i]
            current_streak_end = numbers[i]

    # Add the last streak
    if current_streak_start == current_streak_end:
        streaks.append(str(current_streak_start))
    else:
        streaks.append(f"{current_streak_start}-{current_streak_end}")

    return streaks


def calc_EBM(df_, std_mavsz_, varn0, varn_suffix, n_folds=5, n_jobs=-1):
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    varn = f'{varn0}_{varn_suffix}{std_mavsz_}'
    df_ = df_[~np.isnan(df_[varn])] #NaN
    df_ = df_[~np.isinf(df_[varn])]
    df_ = df_[~np.isinf(df_['err_sens'])]
    
    #dfcs_fixhistlen, cocoln, std_mavsz_, varn0, varn_suffix = args

    X_train = df_[varn].values.reshape(-1, 1) 
    y_train = df_['err_sens'].values#.reshape(-1, 1) 
    print('shapes are ', X_train.shape, y_train.shape)
    ebm = ExplainableBoostingRegressor(feature_names=[varn], random_state=42)
    cv_scores = cross_val_score(ebm, X_train, y_train, cv=n_folds, scoring='r2', n_jobs=n_jobs)
    #print(f"Cross-validation R2 scores: {cv_scores}")
    r2 = np.mean(cv_scores)
    print(f"{varn} mean CV R2 score: {r2:.4f}")

    ebm.fit(X_train, y_train)
    return ebm, cv_scores, r2

# def eboost(X,y):
# from interpret.glassbox import ExplainableBoostingRegressor
# from interpret import show
# from sklearn.model_selection import cross_val_score, train_test_split
# ebm = ExplainableBoostingRegressor(feature_names=feature_names, random_state=42)

# cv_scores = cross_val_score(ebm, X_train, y_train, cv=5, scoring='r2')
# print(f"Cross-validation R2 scores: {cv_scores}")
# print(f"Mean CV R2 score: {np.mean(cv_scores):.4f}")

# from interpret.glassbox import ExplainableBoostingRegressor
# from interpret import show
#     ebm.fit(X_train, y_train)

#     global_explanation = ebm.explain_global(name='EBM Global Explanation')
#     show(global_explanation)