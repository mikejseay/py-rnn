''' stuff related to neural networks '''

# TODO
# pre-extract np stuff
# use in-place arithmetic when possible
# take care with rng seeding, etc.

import numpy as np
from scipy.stats import norm

class Network(object):

    ''' a randomly-connected recurrent neural network '''

    def __init__(self, n_units, p_plastic, p_connect, syn_strength,
                 tau_ms, sigmoid, noise_amp):
        self.n_units = n_units
        self.p_plastic = p_plastic
        self.p_connect = p_connect
        self.syn_strength = syn_strength

        self.tau_ms = tau_ms
        self.sigmoid = sigmoid
        self.noise_amp = noise_amp

        self.n_plastic = int(np.round(n_units * p_plastic))
        self.scale_recurr = syn_strength / np.sqrt(p_connect * n_units)

class Trial(object):

    ''' a single trial and its (timing) characteristics '''
    # defaults not currently designed to be configured
    extra_train_ms = 150
    extra_end_ms = 200
    plot_points = 500

    def __init__(self, length_ms, spacing, time_step, start_train_ms, end_train_ms):
        self.length_ms = length_ms
        self.spacing = spacing
        self.time_step = time_step
        self.start_train_ms = start_train_ms
        self.end_train_ms = end_train_ms

        self.max_ms = self.end_train_ms + self.extra_end_ms
        self.n_steps = int(np.floor(self.max_ms / self.time_step))
        self.time_ms = np.arange(0, self.max_ms, self.time_step)

        # plotting
        self.plot_skip = np.ceil(self.n_steps / self.plot_points)
        if self.plot_skip % 2 == 0:
            self.plot_skip += 1

class Input(object):

    ''' an input to a network '''

    def __init__(self, trial_obj, n_units, value, start_ms, duration_ms):
        self.n_units = n_units
        self.value = value
        self.start_ms = start_ms
        self.duration_ms = duration_ms

        # making input time series
        startpulse_idx = int(np.round(start_ms / trial_obj.time_step))
        pulsedur_samps = int(np.round(duration_ms / trial_obj.time_step))

        input_series = np.zeros((n_units, trial_obj.n_steps))
        input_series[0, startpulse_idx:startpulse_idx + pulsedur_samps - 1] = value
        self.series = input_series

class Output(object):

    ''' a (desired) output from a network '''

    def __init__(self, trial_obj, n_units, value, center_ms, width_ms, baseline_val):
        self.n_units = n_units
        self.value = value
        self.center_ms = center_ms
        self.width_ms = width_ms
        self.baseline_val = baseline_val

        # making output time series
        bell = norm.pdf(trial_obj.time_ms, center_ms, width_ms).reshape(1, -1)
        bell /= np.max(bell)  # by the way, this is a fast way to normalize a vector to 1
        self.series = bell * (value - baseline_val) + baseline_val

class Trainer(object):

    ''' training parameters '''

    def __init__(self, network_obj, input_obj, output_obj, trial_obj,
                 n_trials_recurrent, n_trials_readout, n_trials_test):

        # other objs
        self.network = network_obj
        self.input = input_obj
        self.output = output_obj
        self.trial = trial_obj

        # number of trials
        self.n_trials_recurrent = n_trials_recurrent
        self.n_trials_readout = n_trials_readout
        self.n_trials_test = n_trials_test

    def train(self, stage):
        ''' training kernel '''

        if stage is 'innate':
            use_noiseamp = 0
        else:
            use_noiseamp = self.network.noise_amp

        
