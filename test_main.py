''' testing the network module '''

import numpy as np
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
                'end_train_ms': 1400, }

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
TRIAL = N.Trial(**TRIAL_PARAMS)
IN = N.Input(TRIAL, **INPUT_PARAMS)
OUT = N.Output(TRIAL, **OUTPUT_PARAMS)
TRAIN = N.Trainer(NET, IN, OUT, TRIAL, **TRAIN_PARAMS)
