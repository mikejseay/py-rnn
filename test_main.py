''' testing the network module '''

from tqdm import tqdm

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.stats import norm

import network as N

NETWORK_PARAMS = {'n_units': 800,
                 'p_plastic': 0.6,
                 'p_connect': 0.1,
                 'syn_strength': 1.5,
                 'tau_ms': 10,
                 'sigmoid': np.tanh,
                 'noise_amp': 0.001}

TRIAL_PARAMS = {'length_ms': 1000,
                'spacing': 2,
                'time_step': 1,
                'start_train_ms': 250,
                'end_train_ms': 1400,}

INPUT_PARAMS = {'n_units': 1,
                'value': 5,
                'start_ms': 200,
                'duration_ms': 50}

OUTPUT_PARAMS = {'n_units': 1,
                 'value': 1,
                'center_ms': 1250,
                'width_ms': 30,
                'baseline_val': 0.2}

TRAIN_PARAMS = {'n_trials_recurrent': 20,
                'n_trials_readout': 10,
                'n_trials_test': 10}

NET = N.Network(**NETWORK_PARAMS)
TRYAL = N.Trial(**TRIAL_PARAMS)
IN = N.Input(TRYAL, **INPUT_PARAMS)
OUT = N.Output(TRYAL, **OUTPUT_PARAMS)
TRAIN = N.Trainer(NET, IN, OUT, TRYAL, **TRAIN_PARAMS)

WXX_mask = np.random.rand(NET.n_units, NET.n_units)  # uniform distribution!
WXX_mask[WXX_mask <= NET.p_connect] = 1
WXX_mask[WXX_mask < 1] = 0
WXX_vals = np.random.normal(scale=NET.scale_recurr, size=(NET.n_units, NET.n_units))
WXX_nonsparse = WXX_vals * WXX_mask
np.fill_diagonal(WXX_nonsparse, 0)

WXX = csr_matrix(WXX_nonsparse)
WInputX = np.random.normal(scale=1, size=(NET.n_units, IN.n_units))
WXOut = np.random.normal(scale=1/np.sqrt(NET.n_units), size=(OUT.n_units, NET.n_units))


WXX_ini = WXX.copy()
WXOut_ini = WXOut.copy()

X_history = np.zeros((NET.n_units, TRYAL.n_steps))
Out_history = np.zeros((OUT.n_units, TRYAL.n_steps))
WXOut_len = np.zeros((TRYAL.n_steps))
WXX_len = np.zeros((TRYAL.n_steps))
dW_readout_len = np.zeros((TRYAL.n_steps))
dW_recurr_len = np.zeros((TRYAL.n_steps))

Xv = 2 * np.random.rand(NET.n_units, 1) - 1
X = NET.sigmoid(Xv)
O = np.zeros((OUT.n_units,1))

TRAIN_RECURR = False
TRAIN_READOUT = False
train_window = 0

use_noiseamp = 0
time_div = NET.tau_ms / TRYAL.time_step

for i in tqdm(range(TRYAL.n_steps)):

    in_vec = IN.series[:, i]
    noise = use_noiseamp * np.random.normal(scale=np.sqrt(TRYAL.time_step), size=(NET.n_units,1))
    Xv_current = WXX * X + WInputX * in_vec + noise
    Xv += (-Xv + Xv_current) / time_div
    X = NET.sigmoid(Xv)
    O = np.dot(WXOut, X)

    if (i == TRYAL.start_train_n):
        train_window = True
    if (i == TRYAL.end_train_n):
        train_window = False

    if train_window and i % TRYAL.spacing == 0:

        if TRAIN_RECURR:
            error = X - Target_innate_X[:, i]
            for plas in 1:NET.n_plastic

        if TRAIN_READOUT:
            pass
        
    Out_history[:, i] = O
    X_history[:, [i]] = X
    WXOut_len[i] = np.sqrt(np.sum(np.square(WXOut[:])))
    WXX_len[i] = np.sqrt(np.sum(np.square(WXX[:])))

