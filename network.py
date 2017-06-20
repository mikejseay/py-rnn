''' stuff related to neural networks '''

class Network(object):

    ''' represents a randomly-connected recurrent neural network '''

    def __init__(s, n_units, p_plastic, p_connect, syn_strength):
        s.n_units = n_units
        s.p_plastic = p_plastic
        s.p_connect = p_connect
        s.syn_strength = syn_strength
