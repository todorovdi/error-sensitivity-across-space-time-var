import os
import multiprocessing
import numpy as np
import socket
from os.path import join as pjoin
import argparse
import configparser
import io as StringIO
import pprint

os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS']='1'

# these env variables have to be set 
path_data      = os.environ["DATA_MEMORY_ERRORS_STAB_AND_STOCH"]
path_fig       = os.environ["FIG_MEMORY_ERRORS_STAB_AND_STOCH"]
#path_code      = os.environ["$CODE_MEMORY_ERRORS"]

subjects_predef = ['sub01_WGPOZPEE', 'sub02_CLTLNQWL', 'sub03_GPVDQMWB',
            'sub04_XNDMUSRS', 'sub05_ZGPBOAQU', 'sub06_DLLYEPVA',
            'sub07_MJWXBESS', 'sub08_TXVPROYY', 'sub09_VFDOXEVC',
            'sub10_BJJWDKEK', 'sub11_ERHGZFPL', 'sub12_ZWFBQSXR',
            'sub13_EALZKBNL', 'sub14_RPEADEJG', 'sub15_TAMMXQQS',
            'sub16_SJILLGUV', 'sub17_SUMYMRAR', 'sub18_BBPOBFOQ',
            'sub19_MVAQVMEL', 'sub20_YOGCJKKB']

env2envcode = dict(stable=0, random=1)
env2subtr   = dict(stable=20, random=25)

ps_2nice = dict( zip(['pre','pert','washout','rnd'], 
        ['No perturbation','Perturbation','Washout','Random']) )

phase2trigger = {
    'REST_PHASE': 10,
    'REST_PHASE_RANDOM': 15,
    'TARGET_PHASE': 20,
    'TARGET_PHASE_RANDOM': 25,
    'FEEDBACK_PHASE': 30,
    'FEEDBACK_PHASE_RANDOM': 35,
    'ITI_PHASE': 40,
    'ITI_PHASE_RANDOM': 45,
    'BREAK_PHASE': 50,
    'BREAK_PHASE_RANDOM': 50
}
trigger2phase = {
    10: 'REST_PHASE',
    15: 'REST_PHASE',
    20: 'TARGET_PHASE',
    25: 'TARGET_PHASE',
    30: 'FEEDBACK_PHASE',
    35: 'FEEDBACK_PHASE',
    40: 'ITI_PHASE',
    45: 'ITI_PHASE',
    50: 'BREAK_PHASE',
    55: 'BREAK_PHASE'
}

# this is for stable1, for stable 2 it is inverse
pert_seq = {0: (0,30,0,-30,0), 1: (0,-30,0,30,0)}
block_names = ['stable1','random1','stable2','random2'] # order is important!
pert_stages = np.arange(5, dtype = int)
pertvals = [0, 30, -30]

pert_seq_code_test_trial = 40

control_types_all = ['feedback', 'movement' , 'target', 'belief']
time_lockeds_all = ['feedback', 'target']

a,b = list( zip( *list( env2envcode.items() ) ) )
envcode2env = dict( zip( b,a ) )

####################

# they are same
if os.path.exists(path_data):
    subjects = [f for f in os.listdir(path_data) if f.startswith('sub') ]
    subjects = list(sorted(subjects))
else:
    print(f'data dir {path_data} does not exist, setting default subjects')
    subjects = subjects_predef

if os.path.expandvars('$USER') == 'demitau':
    n_jobs = multiprocessing.cpu_count() - 2
else:
    n_jobs = multiprocessing.cpu_count()
XGB_tree_method_def = 'gpu_hist'

##########################

event_ids_tgt_stable = [20, 21, 22, 23]
event_ids_tgt_random = [25, 26, 27, 28]
event_ids_tgt = event_ids_tgt_stable + event_ids_tgt_random
#event_ids_tgt = [20, 21, 22, 23, 25, 26, 27, 28]
event_ids_feedback_stable = [30]
event_ids_feedback_random = [35]
event_ids_feedback = event_ids_feedback_stable + event_ids_feedback_random

stage2event_ids = { 'target':event_ids_tgt, 'feedback':event_ids_feedback }
stage2evn2event_ids = { 'target':
                            {'stable': event_ids_tgt_stable,
                            'random': event_ids_tgt_random,
                            'all':event_ids_tgt },
                        'feedback':
                            {'stable':event_ids_feedback_stable,
                            'random':event_ids_feedback_random,
                            'all':event_ids_feedback} }
stage2evn2event_ids_str = {}
for tl,vs in stage2evn2event_ids.items():
    stage2evn2event_ids_str[tl] = {}
    for env,ids in vs.items():
        stage2evn2event_ids_str[tl][env] = list(map(str,ids))

freq_names = ['broad', 'theta', 'alpha', 'beta', 'gamma']
freqs = [(4, 60), (4, 7), (8, 12), (13, 30), (31, 60)]
freq_name2freq = dict( list(zip(freq_names,freqs) ) )

target_angs = (np.array([157.5, 112.5, 67.5, 22.5]) + 90) * \
              (np.pi/180)

# target onset time is fixed
stage2time = {'home':(-0.5,0) }
stim_channel_name = 'UPPT001'
delay_trig_photodi = 18  # to account for delay between trig. & photodi.
min_event_duration = 0.02
n_trials_in_block = 192

DEBUG_ntrials       = 40
DEBUG_nchannels     = 3

#stage2time_bounds = { 'feedback': (-2,5), 'target':(-5,2) }
stage2time_bounds = { 'feedback': (-2,3), 'target':(-2,1.5) }

analysis_name2var_ord = {'movement_errors_next_errors_belief':['movement', 'errors', 'next_errors', 'belief'],
'prevmovement_preverrors_errors_prevbelief':['prevmovement', 'preverrors', 'errors', 'prevbelief'] }

def parline2par(line):
    tuples = []
    exprs = line.split('; ')
    for expr in exprs:
        if expr.find('=') >= 0:
            lhs,rhs = expr.split('=')
            tuples += [(lhs,rhs)]
    par = dict(tuples)
    return par

class CustomAction(argparse.Action):
    def __init__(self, check_func, *args, **kwargs):
        """
        argparse custom action.
        :param check_func: callable to do the real check.
        """
        self._check_func = check_func
        super(CustomAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string):
        if isinstance(values, list):
            values = [self._check_func(parser, v) for v in values]
        else:
            values = self._check_func(parser, values)
        setattr(namespace, self.dest, values)

class FormatPrinter(pprint.PrettyPrinter):

    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)
