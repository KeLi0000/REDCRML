import numpy as np
import random
from memory.SegmentTree import SumTree, SumSegmentTree, MinSegmentTree


class UniformReplayBuffer(object):
    def __init__(self, state_dim, memory_size=10000, batch_size=32):
        self.s_size = state_dim * 2 + 2
        self.m_size = memory_size
        self.b_size = batch_size
        self.transition_cnt = 0
        self.memory = np.zeros((self.m_size, self.s_size))

    def reset(self):
        self.transition_cnt = 0
        self.memory = np.zeros((self.m_size, self.s_size))

    def store_transition(self, s, a, s_, r):
        transition = np.hstack((s, [a], s_, [r]))
        if self.transition_cnt == self.m_size:
            self.memory[0: (self.transition_cnt - 2), :] = self.memory[1: self.transition_cnt - 1, :]
            self.memory[self.transition_cnt - 1, :] = transition
        else:
            self.memory[self.transition_cnt, :] = transition
            self.transition_cnt += 1

    def sample(self):
        train_set = None
        if self.transition_cnt > self.b_size:
            if self.transition_cnt == self.m_size:
                sample_index = np.random.choice(self.m_size, size=self.b_size)
            else:
                sample_index = np.random.choice(self.transition_cnt, size=self.b_size)
            train_set = self.memory[sample_index, :]
        return train_set


class SimplePrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    abs_error_upper = 1.  # clipped abs error

    def __init__(self, memory_size, batch_size, alpha=0.6, beta0=0.4, beta_inc=0.001):
        self.tree = SumTree(memory_size)
        self.b_size = batch_size
        self.alpha = alpha  # [0~1] convert the importance of TD error to priority
        self.beta = beta0  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = beta_inc

    def store_transition(self, s, a, s_, r):
        transition = np.hstack((s, [a], s_, [r]))
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p < 0.01:
            max_p = self.abs_error_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self):
        b_idx, minibatch = np.empty((self.b_size,), dtype=np.int32), np.empty((self.b_size, self.tree.data[0].size))
        weights_is = np.empty((self.b_size, 1))
        pri_seg = self.tree.total_p / self.b_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(self.b_size):
            rand_prob = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            idx, p, data = self.tree.get_leaf(rand_prob)
            prob = p / self.tree.total_p
            weights_is[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], minibatch[i, :] = idx, data
        return b_idx, minibatch, weights_is

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_error_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


########################################################################################################################


class ReplayBuffer(object):
    # _radius = 0.01

    def __init__(self, memory_size, batch_size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._storage_sampling_cnt = []  # 每个样本被采样次数
        self._storage_td_error = []  # 每个样本的TD-Error
        self._storage_reward = []  # 每个样本的奖励
        self._maxsize = memory_size
        self._b_size = batch_size
        self._next_idx = 0

        self._s_dim = None

    def __len__(self):
        return len(self._storage)

    def stack_transition(self, s, a, s_, r, c) -> np.ndarray:
        if self._s_dim is None:
            self._s_dim = len(s)

        if isinstance(a, list) or isinstance(a, np.ndarray):
            transition = np.hstack((s, a, s_, [r], [c]))
        else:
            transition = np.hstack((s, [a], s_, [r], [c]))
        return transition

    def store_transition(self, s, a, s_, r, c):
        """
        Add transition to memory
        :param s: t时刻状态
        :param a: t时刻动作
        :param s_: t+1时刻状态
        :param r: t时刻回报
        :param c: t+1时刻状态类型，结束或继续
        :return:
        """
        transition = self.stack_transition(s, a, s_, r, c)

        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
            self._storage_sampling_cnt.append(0)
            self._storage_td_error.append(0)
            self._storage_reward.append(r)
        else:
            # 存储样本
            self._storage[self._next_idx] = transition
            self._storage_sampling_cnt[self._next_idx] = 0
            self._storage_td_error[self._next_idx] = 0
            self._storage_reward[self._next_idx] = r
        # 生成下一个样本存储位置
        self._next_idx = (self._next_idx + 1) % self._maxsize

        # idx = self._is_exist(s)
        # if idx == -1:
        #     if self._next_idx >= len(self._storage):
        #         self._storage.append(transition)
        #     else:
        #         self._storage[self._next_idx] = transition
        #     self._next_idx = (self._next_idx + 1) % self._maxsize
        # else:
        #     old_transition = self._storage[idx].copy()
        #     if a != old_transition[self._s_dim]:
        #         if self._next_idx >= len(self._storage):
        #             self._storage.append(transition)
        #         else:
        #             self._storage[self._next_idx] = transition
        #         self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        train_set = np.array([self._storage[i] for i in idxes])
        return train_set

    # def _is_exist(self, s):
    #     if len(self._storage) <= 1:
    #         return -1
    #     else:
    #         transition_s = np.array(self._storage.copy())[:, 0: self._s_dim]
    #         delta_s = np.array(s) - transition_s
    #
    #         if np.linalg.norm(delta_s, axis=1).min() <= self._radius:
    #             return np.linalg.norm(delta_s, axis=1).argmin()
    #     return -1

    def sample(self):
        """Sample a batch of experiences.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if len(self._storage) < self._b_size:
            return None
        else:
            idxes = np.array([random.randint(0, len(self._storage) - 1) for _ in range(self._b_size)])
            return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, memory_size, batch_size, alpha=0.6, beta0=0.4, beta_inc=0.001):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(memory_size, batch_size)
        assert alpha > 0
        self._alpha = alpha

        assert beta0 > 0
        self._beta = beta0
        self._beta_inc = beta_inc

        it_capacity = 1
        while it_capacity < memory_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def store_transition(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().store_transition(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        len_segment = self._it_sum.sum(0, len(self._storage) - 1) / self._b_size
        for i in range(self._b_size):
            mass = random.uniform(len_segment * i, len_segment * (i + 1))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        if len(self._storage) < self._b_size:
            return None, None, None
        else:
            # 获取样本索引
            idxes = self._sample_proportional()

            # 构建训练集
            train_set = self._encode_sample(idxes)

            # 计算 IS-weights
            self._beta = np.min([1., self._beta + self._beta_inc])
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample / p_min) ** (-self._beta)
                weights.append([weight])
            weights = np.array(weights)
            return idxes, train_set, weights

    def batch_update(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            if priority <= 0:
                priority = 0.00001
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class EpisodeMemory(ReplayBuffer):
    def __init__(self, memory_size):
        super().__init__(memory_size, 0)

    def replay(self):
        return self._storage.copy()

    def reset(self):
        self._storage = []
        self._storage_sampling_cnt = []  # 每个样本被采样次数
        self._storage_td_error = []  # 每个样本的TD-Error
        self._storage_reward = []  # 每个样本的奖励
        self._next_idx = 0
