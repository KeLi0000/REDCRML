# _*_ coding : utf-8 _*_
# @Time: 2025/3/6 22:38
# @File: CurriculumMemory.py
# @Project: iatdrl2
# @Description: 实现一个课程化的replay buffer
# @Authorfix: 李珂
import copy
import math
import random
from typing import List, Union
import numpy as np
import torch
from network.Func import CriticLoss

from memory.Memory import ReplayBuffer
from memory.SegmentTree import SumSegmentTree, MinSegmentTree
from network import PoActorNetwork, PoCriticNetwork, device


class CurriculumReplayBuffer(ReplayBuffer):
    def __init__(self, memory_size, batch_size, curriculum_omega: List[float], sp_lambda=0.5, alpha=0.6, beta0=0.4,
                 beta_inc=0.001, s_dim=0, a_dim=0, gamma=0.99, ubatch_size=1024):
        """
        Curriculum replay buffer，课程化回放经验集
        :param memory_size: 经验集大小
        :param batch_size: 训练批大小
        :param curriculum_omega: 课程因子权重向量
        :param sp_lambda: SP因子的lambda参数
        :param alpha: 差异化采样权重，0没有差异化，1全部差异化
        :param beta0: 差异采样化纠偏权重初始值
        :param beta_inc: 差异采样化纠偏权重增量
        :param s_dim: 状态空间维度
        :param a_dim: 动作空间维度
        :param gamma: 期望折扣系数
        :param ubatch_size: 更新时的batch大小
        """
        super(CurriculumReplayBuffer, self).__init__(memory_size, batch_size)
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

        self._q_eval_nets = []
        self._q_tgt_nets = []
        self._a_net = None
        self._update_worker = None
        self._worker_event = torch.multiprocessing.Event()
        self._worker_event.clear()
        self._s_dim = s_dim
        self._a_dim = a_dim
        self._gamma = gamma
        self._sp_omega = curriculum_omega
        self._sp_lambda = sp_lambda
        self._ubatch_size = ubatch_size

    def store_transition(self, s, a, s_, r, c):
        """See ReplayBuffer.store_effect"""
        transition = self.stack_transition(s, a, s_, r, c)
        idx = None
        if len(self._storage) < self._maxsize:
            self._storage.append(transition)
            self._storage_sampling_cnt.append(0)
            self._storage_td_error.append(float(self._sp_lambda))
            self._storage_reward.append(r)
            idx = len(self._storage) - 1
        else:
            # 选择合适的存储位置
            sampling_cnt = np.array(self._storage_sampling_cnt, dtype=np.float64)
            sampling_cnt /= np.max(sampling_cnt)
            sampling_td_error = np.array(self._storage_td_error, dtype=np.float64)
            sampling_td_error = 1 - sampling_td_error / np.max(sampling_td_error)
            idx = np.argmax(sampling_cnt + sampling_td_error)  # 按照采样频次和TD-error来选，采过最多的、损失最小的
            # 存储样本
            self._storage[idx] = transition
            self._storage_sampling_cnt[idx] = 0
            self._storage_td_error[idx] = float(self._sp_lambda)
            self._storage_reward[idx] = r
        self._storage_td_error[idx] = self._max_priority ** self._alpha
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

        # idx = self._next_idx
        # super().store_transition(*args, **kwargs)

    def _sample_proportional(self):
        res = []
        len_segment = self._it_sum.sum(0, len(self._storage) - 1) / self._b_size
        for i in range(self._b_size):
            mass = random.uniform(len_segment * i, len_segment * (i + 1))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
            self._storage_sampling_cnt[idx] += 1
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
            if math.isnan(p_min) or math.isinf(p_min):
                p_min = 1e-6
            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                try:
                    with np.errstate(divide='raise'):
                        weight = (p_sample / p_min) ** (-self._beta)
                except FloatingPointError:
                    weight = (p_sample / 1e-6) ** (-self._beta)
                weights.append([weight])
            weights = np.array(weights)
            return idxes, train_set, weights

    def _calc_td_error(self, transitions: np.ndarray) -> np.ndarray:
        # 预处理样本集
        batch_s = torch.Tensor(transitions[:, :self._s_dim]).to(device)
        batch_a = torch.Tensor(transitions[:, self._s_dim: self._s_dim + self._a_dim]).to(device)
        batch_s_ = torch.Tensor(transitions[:, self._s_dim + self._a_dim: 2 * self._s_dim + self._a_dim]).to(device)
        batch_r = torch.Tensor(transitions[:, -2][np.newaxis, :]).to(device)
        batch_r = batch_r.T
        batch_c = transitions[:, -1][np.newaxis, :]
        if len(self._q_eval_nets) == 2:
            # 计算 Clipped Double Q-Learning Target
            with torch.no_grad():
                tgt_a = self._a_net(batch_s)
                tgt1_c = self._q_tgt_nets[0](torch.concat([batch_s_, tgt_a], dim=1))
                tgt2_c = self._q_tgt_nets[1](torch.concat([batch_s_, tgt_a], dim=1))
                y_target = torch.add(batch_r,
                                     torch.multiply(torch.minimum(tgt1_c, tgt2_c), self._gamma))  # 不考虑终止状态的学习目标
                termination_index = np.where(batch_c > 0)[0].tolist()
                if len(termination_index) > 0:
                    y_target[termination_index] = batch_r[termination_index]  # 若为终止状态，则目标更换为回报值，不考虑后续影响
            # 计算Critic 1的TD-Error
            c1_eval = self._q_eval_nets[0](torch.concat([batch_s, batch_a], dim=1))
            c1_abs_errors = torch.mean(torch.abs(y_target - c1_eval), dim=1)
            # 计算Critic 2的TD-Error
            c2_eval = self._q_eval_nets[1](torch.concat([batch_s, batch_a], dim=1))
            c2_abs_errors = torch.mean(torch.abs(y_target - c2_eval), dim=1)
            abs_errors = torch.minimum(c1_abs_errors, c2_abs_errors).detach().cpu().numpy()
        else:
            with torch.no_grad():
                # 计算当前 critic
                c_eval = self._q_eval_nets[0](torch.concat([batch_s, batch_a], dim=1))
                # 估计未来 critic
                c_target = self._q_tgt_nets[0](torch.concat([batch_s_, self._a_net(batch_s_)], dim=1))
            # 计算 critic 目标
            y_target = torch.add(batch_r, torch.multiply(c_target, self._gamma))  # 不考虑终止状态的学习目标
            termination_index = np.where(batch_c > 0)[0].tolist()
            if len(termination_index) > 0:
                y_target[termination_index] = batch_r[termination_index]  # 若为终止状态，则目标更换为回报值，不考虑后续影响
            # 优化网络并保存 loss ，然后更新 Priorities
            abs_errors = torch.mean(torch.abs(y_target - c_eval), dim=1)
        return abs_errors

    def calc_fsp_list(self, idxes: List[int]) -> Union[List[float], None]:
        if type(idxes) is list:
            delta = np.array(self._storage_td_error)[idxes]
            leq_idxes = np.where(delta <= self._sp_lambda)[0]
            else_idxes = np.where(delta > self._sp_lambda)[0]
            sp_priority = np.zeros_like(delta)
            sp_priority[leq_idxes] = np.exp(np.fabs(delta[leq_idxes]) - self._sp_lambda)
            sp_priority[else_idxes] = np.exp(self._sp_lambda - np.fabs(delta[else_idxes]))
            return sp_priority.tolist()
        else:
            return None

    def calc_fsp(self, start_idx: int, end_idx: int) -> Union[float, List[float]]:
        if start_idx == end_idx:
            delta = self._storage_td_error[start_idx]
            if abs(delta) <= self._sp_lambda:
                sp_priority = math.exp(abs(delta) - self._sp_lambda)
            else:
                sp_priority = math.exp(self._sp_lambda - abs(delta))
            return sp_priority
        else:
            delta = np.array(self._storage_td_error[start_idx:end_idx])
            leq_idxes = np.where(delta <= self._sp_lambda)[0]
            else_idxes = np.where(delta > self._sp_lambda)[0]
            sp_priority = np.zeros_like(delta)
            sp_priority[leq_idxes] = np.exp(np.fabs(delta[leq_idxes]) - self._sp_lambda)
            sp_priority[else_idxes] = np.exp(self._sp_lambda - np.fabs(delta[else_idxes]))
            return sp_priority.tolist()

    def calc_fds_list(self, idxes: List[int]) -> Union[List[float], None]:
        sample_cnt_max = float(np.max(np.array(self._storage_sampling_cnt, dtype=float)))
        if type(idxes) is list:
            diversity_f = 1 - np.array(self._storage_sampling_cnt, dtype=float)[idxes] / sample_cnt_max
            return diversity_f.tolist()
        else:
            return None

    def calc_fds(self, start_idx: int, end_idx: int) -> Union[float, List[float]]:
        sample_cnt_max = float(np.max(np.array(self._storage_sampling_cnt, dtype=float)))
        if start_idx == end_idx:
            idx = start_idx
            diversity_f = 1 - self._storage_sampling_cnt[idx] / sample_cnt_max
            return diversity_f
        else:
            diversity_f = 1 - np.array(self._storage_sampling_cnt[start_idx:end_idx], dtype=float) / sample_cnt_max
            return diversity_f.tolist()

    def calc_frwd_list(self, idxes: List[int]) -> Union[List[float], None]:
        rwd_min = np.min(np.array(self._storage_reward))
        rwd_max = np.max(np.array(self._storage_reward))
        if type(idxes) is list:
            rwd = np.array(self._storage_reward)[idxes]
            reward_f = (rwd - rwd_min) / (rwd_max - rwd_min)
            return reward_f.tolist()
        else:
            return None

    def calc_frwd(self, start_idx: int, end_idx: int) -> Union[float, List[float]]:
        rwd_min = np.min(np.array(self._storage_reward))
        rwd_max = np.max(np.array(self._storage_reward))
        if start_idx == end_idx:
            idx = start_idx
            rwd = self._storage_reward[idx]
            reward_f = (rwd - rwd_min) / (rwd_max - rwd_min)
            return reward_f
        else:
            rwd = np.array(self._storage_reward[start_idx:end_idx])
            reward_f = (rwd - rwd_min) / (rwd_max - rwd_min)
            return reward_f.tolist()

    def calc_priority(self, start_idx: int, end_idx: int) -> Union[float, List[float]]:
        if start_idx == end_idx:
            idx = start_idx
            td_f = self.calc_fsp(idx, idx)
            diversity_f = self.calc_fds(idx, idx)
            reward_f = self.calc_frwd(idx, idx)
            priority = self._sp_omega[0] * td_f + self._sp_omega[1] * diversity_f + self._sp_omega[2] * reward_f
            return priority
        else:
            td_f = np.array(self.calc_fsp(start_idx, end_idx))
            diversity_f = np.array(self.calc_fds(start_idx, end_idx))
            reward_f = np.array(self.calc_frwd(start_idx, end_idx))
            priority = self._sp_omega[0] * td_f + self._sp_omega[1] * diversity_f + self._sp_omega[2] * reward_f
            return priority.tolist()

    def calc_priority_list(self, idxes: List[int]) -> List[float]:
        td_f = np.array(self.calc_fsp_list(idxes))
        diversity_f = np.array(self.calc_fds_list(idxes))
        reward_f = np.array(self.calc_frwd_list(idxes))
        priority = self._sp_omega[0] * td_f + self._sp_omega[1] * diversity_f + self._sp_omega[2] * reward_f
        return priority.tolist()

    def update_worker(self, stop_event: torch.multiprocessing.Event):
        # 先把网络迁移到GPU上
        if len(self._q_eval_nets) == 2:
            for i in range(2):
                self._q_eval_nets[i].to_gpu()
                self._q_tgt_nets[i].to_gpu()
        else:
            self._q_eval_nets[0].to_gpu()
            self._q_tgt_nets[0].to_gpu()
        self._a_net.to_gpu()
        # 多轮次，并行计算
        iter_cnt = int(math.ceil(len(self._storage) / self._ubatch_size))
        for idx in range(iter_cnt):
            # 处理外部要求的结束事件
            if stop_event.is_set():
                return
            # 逐批次计算样本的priority
            # 确定本批次样本的范围索引
            start_idx = idx * self._ubatch_size
            end_idx = (idx + 1) * self._ubatch_size
            if end_idx > len(self._storage):
                end_idx = len(self._storage)
            batch_transitions = np.array(self._storage)[start_idx: end_idx, :]  # 取出样本
            td_errors = self._calc_td_error(batch_transitions).astype(dtype=float)  # 计算所有样本的TD Error
            self._storage_td_error[start_idx: end_idx] = td_errors.tolist()  # 保存样本的TD Error
            priorities = self.calc_priority(start_idx, end_idx)
            self._max_priority = max(self._max_priority, np.max(np.array(priorities)))
            for item, priority in zip(list(range(start_idx, end_idx)), priorities):
                self._it_sum[item] = priority ** self._alpha
                self._it_min[item] = priority ** self._alpha

    def asynchronous_update(self, q_eval_nets: List[Union[PoCriticNetwork, CriticNetwork]],
                            q_tgt_nets: List[Union[PoCriticNetwork, CriticNetwork]],
                            a_net: Union[PoActorNetwork, ActorNetwork], blocking=False):
        """
        使用multiprocessing，异步更新memory中样本的priority和SP Regularization
        :param blocking: 是否阻塞更新
        :param q_tgt_nets: 当前最新的目标Q网络
        :param q_eval_nets: 当前最新的评估Q网络
        :param a_net: 当前最新的策略网络
        """
        # 判断异步更新进程是否结束，如果没有结束，则强行终止本次更新
        if self.is_asynchronous_work():
            self._worker_event.set()
            self._update_worker.join(0.1)
            if self._update_worker.is_alive():
                self._update_worker.terminate()
            self._worker_event.clear()
        # 更新用于计算样本TD-Error的网络模型
        self._q_eval_nets.clear()
        self._q_tgt_nets.clear()
        if len(q_eval_nets) == 2:
            for i in range(2):
                self._q_eval_nets.append(copy.deepcopy(q_eval_nets[i]))
                self._q_eval_nets[i].to_cpu()
                self._q_tgt_nets.append(copy.deepcopy(q_tgt_nets[i]))
                self._q_tgt_nets[i].to_cpu()
        else:
            self._q_eval_nets.append(copy.deepcopy(q_eval_nets[0]))
            self._q_tgt_nets.append(copy.deepcopy(q_tgt_nets[0]))
        del self._a_net
        self._a_net = copy.deepcopy(a_net)
        self._a_net.to_cpu()
        # 开始更新样本的priority和SP Regularization
        self._update_worker = torch.multiprocessing.Process(target=self.update_worker, args=(self._worker_event,))
        self._update_worker.start()
        if blocking:
            while self.is_asynchronous_work():
                pass

    def is_asynchronous_work(self) -> bool:
        if self._update_worker is None:
            return False
        else:
            return self._update_worker.is_alive()

    def batch_update(self, idxes, abs_errors):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        abs_errors: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(abs_errors)
        # 使用矩阵运算保存TD-Error
        tmp = np.array(self._storage_td_error)
        tmp[idxes] = abs_errors
        self._storage_td_error = tmp.tolist()
        # 计算priorities
        priorities = self.calc_priority_list(idxes)
        # 按顺序给树赋值
        for idx, priority in zip(idxes, priorities):
            if priority <= 1e-5:
                priority = 1e-5
            disc_priority = priority ** self._alpha
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = disc_priority
            self._it_min[idx] = disc_priority
            self._max_priority = max(self._max_priority, priority)

        # for idx, error in zip(idxes, abs_errors):
        #     if error <= 0:
        #         error = 0.00001
        #     assert error > 0
        #     priority = self.calc_priority(idx, idx)
        #     assert 0 <= idx < len(self._storage)
        #     disc_priority = priority ** self._alpha
        #     self._it_sum[idx] = disc_priority
        #     self._it_min[idx] = disc_priority
        #     self._max_priority = max(self._max_priority, priority)
