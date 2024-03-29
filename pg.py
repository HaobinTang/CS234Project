import numpy as np
# from collections import OrderedDict
import tensorflow as tf
import parameters
import environment
import os
from utils.general import get_logger, Progbar, export_plot

def build_mlp(mlp_input, output_size, scope, n_layers, size, output_activation=None):
    """
    Build a feed forward network (multi-layer perceptron, or mlp)
    with 'n_layers' hidden layers, each of size 'size' units.
    Use tf.nn.relu nonlinearity between layers.
    Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output layer size
          scope: the scope of the neural network
          n_layers: the number of hidden layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
    Returns:
          The tensor output of the network
    """
    #######################################################
    #########   YOUR CODE HERE - 7-20 lines.   ############
    input = mlp_input
    with tf.variable_scope(scope):
        input = tf.layers.dense(input, 500, activation = tf.nn.relu)
        for i in range(n_layers):
          input = tf.layers.dense(input, size, activation = tf.nn.relu)
        output = tf.layers.dense(input, output_size, activation = output_activation)
    return output

class PG(object):
    """
    Abstract Class for implementing a Policy Gradient Based Algorithm
    """
    def __init__(self, env, pa, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: an built simulator environment
                pa: class with hyperparameters
                logger: logger instance from the logging module
        """
        # directory for training outputs
        if not os.path.exists(pa.output_path):
          os.makedirs(pa.output_path)

        # store hyperparameters
        self.pa = pa
        self.logger = logger
        if logger is None:
          self.logger = get_logger(pa.log_path)
        self.env = env

        # state and action space
        self.observation_dim = self.pa.network_input_dim
        self.action_dim = self.pa.network_output_dim

        #training hyperparameters
        self.num_frames = pa.num_frames
        self.lr = self.pa.learning_rate

        # build model
        self.build()

    def add_placeholders_op(self):
        """
        Add placeholders for observation, action, and advantage:
            self.observation_placeholder, type: tf.float32
            self.action_placeholder, type: depends on the self.discrete
            self.advantage_placeholder, type: tf.float32
        """
        self.observation_placeholder = tf.placeholder(tf.float32, shape = [None, self.observation_dim])
        self.action_placeholder = tf.placeholder(tf.int64, shape = [None,])
        self.advantage_placeholder = tf.placeholder(tf.float32, shape = [None,])

    def build_policy_network_op(self, scope = "policy_network"):
        """
        Build the policy network, construct the tensorflow operation to sample
        actions from the policy network outputs, and compute the log probabilities
        of the actions taken (for computing the loss later). These operations are
        stored in self.sampled_action and self.logprob. Must handle both settings
        of self.discrete.

        Args:
              scope: the scope of the neural network
        """
        action_logits = build_mlp(self.observation_placeholder, self.action_dim, scope, self.pa.n_layers, self.pa.layer_size)
        self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), axis = 1)
        self.logprob = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.action_placeholder, logits = action_logits)

    def add_loss_op(self):
        """
        Compute the loss, averaged for a given batch.
        The update for REINFORCE with advantage:
        θ = θ + α ∇_θ log π_θ(a_t|s_t) A_t
        """
        self.loss = - tf.reduce_mean(self.logprob * self.advantage_placeholder)

    def add_optimizer_op(self):
        """
        Set 'self.train_op' using AdamOptimizer
        """
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)

    def add_baseline_op(self, scope = "baseline"):
        """
        Build the baseline network within the scope.
        Use build_mlp with the same parameters as the policy network to
        get the baseline estimate, and setup a target placeholder and
        an update operation so the baseline can be trained.

        Args:
          scope: the scope of the baseline network
        """
        self.baseline = tf.squeeze(build_mlp(self.observation_placeholder, 1, scope, self.pa.n_layers, self.pa.layer_size))
        self.baseline_target_placeholder = tf.placeholder(tf.float32, shape = [None, ])
        loss = tf.losses.mean_squared_error(labels = self.baseline_target_placeholder, predictions = self.baseline)
        self.update_baseline_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(loss)

    def build(self):
        """
        Build the model by adding all necessary variables.
        """
        # add placeholders
        self.add_placeholders_op()
        # create policy net
        self.build_policy_network_op()
        # add square loss
        self.add_loss_op()
        # add optmizer for the main networks
        self.add_optimizer_op()

        # add baseline
        if self.pa.use_baseline:
            self.add_baseline_op()

    def initialize(self):
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()
        # tensorboard stuff
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_summary(self):
        """
        Tensorboard stuff.
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.pa.output_path, self.sess.graph)

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        Args:
          rewards: deque
          scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        """
        Add summary to tensorboard
        """

        fd = {
        self.avg_reward_placeholder: self.avg_reward,
        self.max_reward_placeholder: self.max_reward,
        self.std_reward_placeholder: self.std_reward,
        self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

    def sample_path(self, env, num_episodes = None):
        """
        Sample trajectories from the environment.

        Args:
            env: a built simulator environment
            num_episodes: the number of episodes to be sampled
              if none, sample one batch (size indicated by config file)
        Returns:
          paths: a list of paths. Each path in paths is a dictionary with
              path["observation"] a numpy array of ordered observations in the path
              path["actions"] a numpy array of the corresponding actions in the path
              path["reward"] a numpy array of the corresponding rewards in the path
          total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.pa.batch_size):
            env.reset()
            states, actions, rewards, infos = [], [], [], []
            state = env.observe()
            episode_reward = 0

            for step in range(self.pa.episode_max_length):
                states.append(state)
                action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})[0]
                actions.append(action)
                state, reward, done, info = env.step(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if (done or step == self.pa.episode_max_length-1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.pa.batch_size:
                    break

            path = {"observation" : np.array(states),
                            "reward" : np.array(rewards),
                            "action" : np.array(actions),
                            "info"   : info}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= self.pa.num_seq_per_batch:
              break

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep
        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):
                    G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T
            where T is the last timestep of the episode.
        Args:
              paths: recorded sample paths.  See sample_path() for details.
        Return:
              returns: return G_t for each timestep
        """
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = []
            T = len(rewards)
            for i in range(T):
                gammas = np.logspace(0, T - i, num = T - i, base = self.pa.gamma, endpoint = False)
                r_t = np.dot(rewards[i:], gammas)
                returns.append(r_t)
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def calculate_advantage(self, returns, observations):
        """
        Calculate the advantage
        using baseline adjustment if necessary, and normalizing the advantages if necessary.
        If neither of these options are True, just return returns.
        Args:
              returns: all discounted future returns for each step
              observations: observations
        Returns:
              adv: Advantage
        """
        adv = returns
        if self.pa.use_baseline:
            adv = adv - self.sess.run(self.baseline, feed_dict = {self.observation_placeholder : observations})
        if self.pa.normalize_advantage:
            std = np.std(adv)
            mean = np.mean(adv)
            adv = (adv - mean) / std
        return adv

    def update_baseline(self, returns, observations):
        """
        Update the baseline from given returns and observation
        Args:
              returns: Returns from get_returns
              observations: observations
        """
        self.sess.run(self.update_baseline_op, feed_dict = {self.observation_placeholder : observations, self.baseline_target_placeholder: returns})

    def train(self):
        """
        Performs training
        """
        last_eval = 0
        last_record = 0
        scores_eval = []

        self.init_averages()
        scores_eval = [] # list of scores computed at iteration time

        for t in range(self.pa.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            scores_eval = scores_eval + total_rewards
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.pa.use_baseline:
                self.update_baseline(returns, observations)
            self.sess.run(self.train_op, feed_dict={
                          self.observation_placeholder : observations,
                          self.action_placeholder : actions,
                          self.advantage_placeholder : advantages})

            # tf stuff
            if (t % self.pa.summary_freq == 0):
                self.update_averages(total_rewards, scores_eval)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        self.logger.info("- Training done.")
        export_plot(scores_eval, "Score", self.pa.env_name, self.pa.plot_output)

    def evaluate(self, env = None, num_episodes = 1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None: env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # initialize
        self.initialize()
        # model
        self.train()

if __name__ == '__main__':
    pa = parameters.Parameters()
    env = environment.Env(pa, end = 'no_new_job')
    # train model
    model = PG(env, pa)
    model.run()
