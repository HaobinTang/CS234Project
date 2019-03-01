import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000         # number of training epochs
        self.simu_len = 10             # length of the busy cycle that repeats itself
        self.num_ex = 1                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 2               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action


import tensorflow as tf

class config_cartpole:
    def __init__(self, use_baseline):
        self.env_name="CartPole-v0"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_pendulum:
    def __init__(self, use_baseline):
        self.env_name="InvertedPendulum-v1"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cheetah:
    def __init__(self, use_baseline):
        self.env_name="HalfCheetah-v1"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 3
        self.layer_size             = 32
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

def get_config(env_name, baseline):
    if env_name == 'cartpole':
        return config_cartpole(baseline)
    elif env_name == 'pendulum':
        return config_pendulum(baseline)
    elif env_name == 'cheetah':
        return config_cheetah(baseline)

