import numpy as np
import math
import tensorflow as tf

import parameters

class Env:
    """
    Abstract Class for implementing environment of simulator
    """
    def __init__(self, pa, nw_len_seqs=None, nw_size_seqs=None,
                 seed=42, end='no_new_job'):
        """
        Initialize Environment Class
        Args:
                pa: hyperparameters from configure.py
                nw_len_seqs: a sequence of duration of all jobs
                nw_size_seqs: a sequence of resources requirement of all jobs
                seed: seed to fix generated sequence
                end: two types of termination - "no_new_job" means all jobs have been assigned. "all_done" means all jobs have been finished.

        """
        # load hyperparameters and termination condition
        self.pa = pa
        self.end = end

        # load the stochastic distribution of jobs
        self.nw_dist = pa.dist.bi_model_dist

        # Initial parameter to record current time
        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None or nw_size_seqs is None:
            # generate job sequences: 1. duration sequences; 2 resource requirement sequences
            self.nw_len_seqs, self.nw_size_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            # Parameter to indicate workload of the generated sequence: sum of (duration * requried resource for each job) / (# jobs) / (# available resources)
            self.workload = np.zeros(self.pa.num_res)
            for i in range(self.pa.num_res):
                self.workload[i] = \
                    np.sum(self.nw_size_seqs[:, i] * self.nw_len_seqs) / \
                    float(pa.res_slot) / \
                    float(len(self.nw_len_seqs))
                print("Load on # " + str(i) + " resource dimension is " + str(self.workload[i]))
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            self.nw_size_seqs = np.reshape(self.nw_size_seqs,
                                           [self.pa.num_ex, self.pa.simu_len, self.pa.num_res])
        else:
            self.nw_len_seqs = nw_len_seqs
            self.nw_size_seqs = nw_size_seqs

        # Index for which example sequence
        self.seq_no = 0
        # Index to record which new job to be schedule next in job sequence
        self.seq_idx = 0

        # initialize simulator
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):
        """
        Generate job sequences under stochastic distribution as job_distribution assumed
        Args:
                simu_len: length of the sequence of jobs
        Return:
                nw_len_seq: job duration sequence tensor
                nw_size_seq: job resource requirement sequence tensor

        """

        nw_len_seq = np.zeros(simu_len, dtype=int)
        nw_size_seq = np.zeros((simu_len, self.pa.num_res), dtype=int)

        for i in range(simu_len):

            if np.random.rand() < self.pa.new_job_rate:  # a new job comes

                nw_len_seq[i], nw_size_seq[i, :] = self.nw_dist()

        return nw_len_seq, nw_size_seq

    def get_new_job_from_seq(self, seq_no, seq_idx):
        """
        Get a new job from the job sequences given the current sequence number and current index
        Args:
                seq_no: current sequence number
                seq_idx: current job index in the job sequence
        Return:
                new_job: a new job from Job class with parameters from job sequences
                            - resource
                            - duration
                            - job id as current length of job record
                            - arrival time
        """
        new_job = Job(res_vec=self.nw_size_seqs[seq_no, seq_idx, :],
                      job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

    def observe(self):

            state = np.zeros(self.pa.time_horizon * (self.pa.num_res + 1) +  # current work
                                    self.pa.num_nw * (self.pa.num_res + 1) + # new work
                                    1,                                       # backlog indicator
                                    dtype = np.float32)

            pt = 0

            # current work reward, after each time step, how many jobs left in the machine
            job_allocated = np.ones(self.pa.time_horizon) * len(self.machine.running_job)
            for j in self.machine.running_job:
                job_allocated[j.finish_time - self.curr_time: ] -= 1

            state[pt: pt + self.pa.time_horizon] = job_allocated
            pt += self.pa.time_horizon

            # current work available slots
            for i in range(self.pa.num_res):
                state[pt: pt + self.pa.time_horizon] = self.machine.avbl_slot[:, i]
                pt += self.pa.time_horizon

            # new work duration and size
            for i in range(self.pa.num_nw):

                if self.job_slot.slot[i] is None:
                    state[pt: pt + self.pa.num_res + 1] = 0
                    pt += self.pa.num_res + 1
                else:
                    state[pt] = self.job_slot.slot[i].len
                    pt += 1

                    for j in range(self.pa.num_res):
                        state[pt] = self.job_slot.slot[i].res_vec[j]
                        pt += 1

            # backlog queue
            state[pt] = self.job_backlog.curr_size
            pt += 1

            assert pt == len(state)  # fill up the compact representation vector

            return state

    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)

        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)

        return reward

    def Move_on(self, done):
        '''
        1. Update state transition of time (proceeding) when no job was allocated.
        2. Get new job from job sequence and put it into job slots or backlog
        3. Update reword after allocation of jobs
        Args:
                done: whether the episode ends
        Return:
                reward: the incremental slowdown reward by time proceeds
        '''
        # Proceeds one time step
        self.curr_time += 1
        self.machine.time_proceed(self.curr_time)
        self.extra_info.time_proceed()

        # add new jobs
        self.seq_idx += 1

        if self.end == "no_new_job":  # Termination type 1: end of new job sequence
            if self.seq_idx >= self.pa.simu_len:
                done = True
        elif self.end == "all_done":  # Termination type 2: everything has to be finished
            if self.seq_idx >= self.pa.simu_len and \
               len(self.machine.running_job) == 0 and \
               all(s is None for s in self.job_slot.slot) and \
               all(s is None for s in self.job_backlog.backlog):
                done = True
            elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                done = True
                print("Run out of maximum allowed time!")

        if not done: # not terminate
            if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                # Get new job from job sequence
                new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)
                # if a new job comes
                if new_job.len > 0:
                    to_backlog = True
                    # Try to put in new visible job slots
                    for i in range(self.pa.num_nw):
                        if self.job_slot.slot[i] is None:
                            self.job_slot.slot[i] = new_job
                            self.job_record.record[new_job.id] = new_job
                            to_backlog = False
                            break
                    # Put to backlog if job slots are full
                    if to_backlog:
                        if self.job_backlog.curr_size < self.pa.backlog_size:
                            self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                            self.job_backlog.curr_size += 1
                            self.job_record.record[new_job.id] = new_job
                        else:  # abort, backlog full
                            print("Backlog is full.")
                            # exit(1)
                    #update new job extra information
                    self.extra_info.new_job_comes()
        # Update reward
        reward = self.get_reward()
        return reward

    def Allocate (self, a):
        '''
        Update parameters for new state after taking a selected action
        Args:
                a: action from the agent like PG_network, SJF
        '''
        # Store the allocated job in job_record with its number in record as id
        self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a]
        self.job_slot.slot[a] = None

        # dequeue backlog to fill the job slot just allocated
        if self.job_backlog.curr_size > 0:
            self.job_slot.slot[a] = self.job_backlog.backlog[0]  # if backlog empty, it will be 0
            self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
            self.job_backlog.backlog[-1] = None
            self.job_backlog.curr_size -= 1


    def step(self, a, repeat=False):
        '''
        Update parameters for new state after taking a selected action
        Args:
                a: action from the agent like PG_network, SJF
        Returns:
                ob:
                reward:
                done:
                info
        '''
        status = None

        done = False
        reward = 0
        info = None
        #
        if a == self.pa.num_nw:  # explicit void action
            status = 'MoveOn'
        elif self.job_slot.slot[a] is None:  # implicit void action
            status = 'MoveOn'
        else:
            allocated = self.machine.allocate_job(self.job_slot.slot[a], self.curr_time)
            if not allocated:  # implicit void action
                status = 'MoveOn'
            else:
                status = 'Allocate'
        if status == 'MoveOn':
            reward= self.Move_on(done)
        elif status == 'Allocate':
            self.Allocate(a)

        state = self.observe()
        info = self.job_record
        # if done, reset sequence idex and move to next sequence
        if done:
            self.seq_idx = 0
            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex
            self.reset()

        return state, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    """
    Abstact class for job
    """
    def __init__(self, res_vec, job_len, job_id, enter_time):
        self.id = job_id
        self.res_vec = res_vec
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    """
    Abstact class for job slots
    """
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    """
    Abstact class for job backlog
    """
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    """
    Abstact class for job record in a dictionary
    """
    def __init__(self):
        self.record = {}


class Machine:
    """
    Abstact class for Machine
    """
    def __init__(self, pa):
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []

    def allocate_job(self, job, curr_time):
        '''
        Determine whether a job chosen by the agent can be allocated in the resources. If allocated, updated the information of job
        slot and running jobs and return "allocated==True". If fail to allocate, return "allocated==False".
        Args:
                job: the job chosen by the agent and ready to allocate
                curr_time: Current time of the state
        Return:
                allocated: the condition whether the job is successfully allocated
        '''
        allocated = False

        for t in range(0, self.time_horizon - job.len):

            new_avbl_res = self.avbl_slot[t: t + job.len, :] - job.res_vec

            if np.all(new_avbl_res[:] >= 0):

                allocated = True

                self.avbl_slot[t: t + job.len, :] = new_avbl_res
                job.start_time = curr_time + t
                job.finish_time = job.start_time + job.len

                self.running_job.append(job)

                break

        return allocated

    def time_proceed(self, curr_time):
        '''
        When time proceeds, resource is doing 1 timestep of the job. Move the remain part of the job to the front and release
        the end of the resource to the maximum number of available resource slots. Also, compare the current time with the
        finish time of a job, if the current time is equal and bigger than the job, it is done and can be removed from the running jobs.
        Args:
                curr_time: Current time of the state
        Return:

        '''
        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)


class ExtraInfo:
    """
    Abstact class for Extra Information
    """
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1

    env = Env(pa)

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print ("New job is backlogged.")

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print ("- Backlog test passed -")


if __name__ == '__main__':
    test_backlog()
