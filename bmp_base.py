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