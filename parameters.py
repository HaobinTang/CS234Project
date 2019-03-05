import numpy as np
import math
import tensorflow as tf
import job_distribution


class Parameters:
    def __init__(self):
        self.env_name = "simulator"

        # output config
        self.output_path = "results/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"

        # parameters for the policy and baseline models
        # 1st hidden layer size = 520 with relu to match baseline model
        self.n_layers               = 2
        self.layer_size             = 20
        self.activation             = tf.nn.relu

        # model and training config
        self.use_baseline = True
        self.normalize_advantage = True
        # self.num_epochs = 10000        # number of training epochs
        self.simu_len = 200             # length of the busy cycle that repeats itself ？？？
        self.num_ex = 10                # number of sequences
        self.num_batches = 100         # number of batches trained on

        self.summary_freq = 1          # interval for summary output
        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 20    # number of sequences to compute baseline ???
        self.batch_size = 10000        # number of steps used to compute each policy update
        self.episode_max_length = 2000  # enforcing an artificial terminal

        self.num_res = 2               # number of resources in the system
        self.num_nw = 10                # maximum allowed number of work in the queue

        self.time_horizon = 20         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.new_job_rate = 0.3        # lambda in new job arrival Poisson Process

        self.gamma = 1                 # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # # graphical representation
        # assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        # self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        # self.network_input_height = self.time_horizon
        # self.network_input_width = \
        #     (self.res_slot +
        #      self.max_job_size * self.num_nw) * self.num_res + \
        #     self.backlog_width + \
        #     1  # for extra info, 1) time since last new job

        # compact representation
        self.network_input_dim = (self.time_horizon * (self.num_res + 1) +  # current work
                                self.num_nw * (self.num_res + 1) +           # new work
                                1)                                                  # backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.learning_rate = 0.001    # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # # supervised learning mimic policy
        # self.batch_size = 10
        # self.evaluate_policy_name = "SJF"
