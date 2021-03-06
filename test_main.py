""" implements the experimental protocol of Laje & Buonomano (2013).
    sets parameters, performs the experiment, and saves figures into a PDF
    that will be placed into the figs subdirectory of the codebase """

import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import network as N

# setting parameters
GENERATOR_PARAMS = {'n_units': 800,
                    'p_plastic': 0.6,
                    'p_connect': 0.1,
                    'syn_strength': 1.5, }

TRIAL_PARAMS = {'length_ms': 1000,
                'spacing': 2,
                'time_step': 1,
                'start_train_ms': 250,
                'end_train_ms': 1400, }

INPUT_PARAMS = {'n_units': 1,
                'value': 5,
                'start_ms': 200,
                'duration_ms': 50, }

OUTPUT_PARAMS = {'n_units': 1,
                 'value': 1,
                 'center_ms': 1250,
                 'width_ms': 30,
                 'baseline_val': 0.2, }

TRAIN_PARAMS = {'tau_ms': 10,
                'sigmoid': np.tanh,
                'noise_harvest': 0,
                'noise_train': 0.001,
                'n_trials_recurrent': 20,  # 20
                'n_trials_readout': 10,  # 10
                'n_trials_test': 10, }  # 10

# instantiating objects
GEN = N.Generator(**GENERATOR_PARAMS)
TRYAL = N.Trial(**TRIAL_PARAMS)
IN = N.Input(TRYAL, **INPUT_PARAMS)
OUT = N.Output(TRYAL, **OUTPUT_PARAMS)
TRAIN = N.Trainer(GEN, IN, OUT, TRYAL, **TRAIN_PARAMS)

# performing the experimental protocol
TRAIN.initialize_weights()
TRAIN.harvest_innate()
TRAIN.train_recurrent()
x_history, out_history, error_history, wxout_history = TRAIN.train_readout()
f_lst = TRAIN.test()

# saving the test trial figures into a PDF (one trial per page)
cwd = os.getcwd()
fig_dir = os.path.join(cwd, 'figs')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
pdf_path = os.path.join(fig_dir, 'test_result.pdf')

with PdfPages(pdf_path) as pdf:
    for f in f_lst:
        pdf.savefig(f)
