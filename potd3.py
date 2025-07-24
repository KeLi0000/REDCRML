# _*_ coding : utf-8 _*_
# @Time: 2025/3/6 20:33
# @File : potd3.py
# @Project : iatdrl2
# @Description : 与TD3类似，但网络结构里增加LSTM部分
# @Author : 李珂
import os
from queue import Queue
from typing import Tuple, Union, List

import numpy as np
import torch
from pyds import MovAvg
from torch.utils.tensorboard import SummaryWriter

from memory import ReplayBuffer, PrioritizedReplayBuffer, CurriculumReplayBuffer
from network import PoCriticNetwork, PoActorNetwork, device, CriticLoss
from noise import NormalActionNoise, AdaptiveNormalActionNoise, OrnsteinUhlenbeckActionNoise
from save import save_net_structure_to_xml, save_critic_structure_to_xml

np.random.seed(1)


class PartialObservedTwinDelayedDeepDeterministicPolicyGradient(object):
    def __init__(self, mission_path, action_dim, normal_state_dim, seq_state_dim, state_seq_len, action_bound,
                 critic_input_s1_units=16, critic_input_s2_units=16, critic_input_s2_layers_num=4,
                 critic_input_ah_units=16, critic_input_ah_layers_num=1, critic_input_an_units=16,
                 critic_middle_layers_num=1, critic_middle_layers_units=None, actor_s1_input_units=16,
                 actor_s2_input_units=16, actor_s2_input_layers_num=2, actor_ah_input_units=16,
                 actor_ah_input_layers_num=2, actor_middle_layers_num=4, actor_middle_layers_units=None, actor_bn=True,
                 actor_weight_decay=0.01, critic1_weight_decay=0.01, critic2_weight_decay=0.01,
                 learning_rate_actor=0.001, learning_rate_critic=0.001, reward_decay=0.9, soft_replace=0.01,
                 delay_update_episode=2, smoothing_sigma=np.array([.1]), smoothing_range=np.array([.2]),
                 memory_type='Uniform', memory_size=5000, memory_alpha=0.5, memory_beta0=0.4,
                 memory_beta_increment=0.001, memory_c_omega=None, memory_sp_lambda=0.5, batch_size=32,
                 noise_type='AN', n_noise_sigma=np.array([.1]), ou_noise_sigma=np.array([.1]),
                 an_noise_params=np.array([1.0, 0.1, 1.01]), file_path=None, req_save_params=False):
        """
        TD3 算法
        :param mission_path: 任务路径
        :param action_dim: 动作维度
        :param normal_state_dim: 普通状态维度
        :param seq_state_dim: 序列化状态维度
        :param state_seq_len: 状态序列长度
        :param action_bound:
        :param critic_input_s1_units:
        :param critic_input_s2_units:
        :param critic_input_s2_layers_num:
        :param critic_input_ah_units:
        :param critic_input_ah_layers_num:
        :param critic_input_an_units:
        :param critic_middle_layers_num:
        :param critic_middle_layers_units:
        :param actor_s1_input_units:
        :param actor_s2_input_units:
        :param actor_s2_input_layers_num:
        :param actor_ah_input_units:
        :param actor_ah_input_layers_num:
        :param actor_middle_layers_num:
        :param actor_middle_layers_units:
        :param actor_bn:
        :param actor_weight_decay:
        :param critic1_weight_decay:
        :param critic2_weight_decay:
        :param learning_rate_actor:
        :param learning_rate_critic:
        :param reward_decay:
        :param soft_replace:
        :param delay_update_episode:
        :param smoothing_sigma:
        :param smoothing_range:
        :param memory_type: Replay Buffer 类型，‘Uniform’ 均匀采样，‘Prioritized’ 优先级采样，‘Curriculum’ 课程化采样
        :param memory_size: Buffer大小
        :param memory_alpha: PER的参数
        :param memory_beta0: PER的参数
        :param memory_beta_increment: PER的参数
        :param noise_type: 噪声类型，'OU' OU噪声，'AN' 自适应高斯噪声，'N' 高斯噪声
        :param n_noise_sigma: 高斯 过程噪声参数，sigma
        :param ou_noise_sigma: OU 过程噪声参数，sigma
        :param an_noise_params: AN 过程噪声参数，0: 初始方差，1: 目标方差，2: 方差变化率
        :param batch_size: 训练批的尺寸
        :param file_path:
        :param req_save_params: 是否保存损失变化数据
        """
        # 动作空间维度，状态空间维度，动作空间各个维度上的边界
        if memory_c_omega is None:
            memory_c_omega = [1 / 3.] * 3
        if actor_middle_layers_units is None:
            actor_middle_layers_units = [16] * actor_middle_layers_num
        if critic_middle_layers_units is None:
            critic_middle_layers_units = [16] * critic_middle_layers_num
        self._a_dim = action_dim
        self._s1_dim, self._s2_dim, self._s2_seq_len = normal_state_dim, seq_state_dim, state_seq_len
        self._ah_seq_len = self._s2_seq_len - 1  # Actor Critic的动作输入序列长度
        self._s_dim = self._s1_dim + self._s2_dim * self._s2_seq_len + self._a_dim * self._ah_seq_len
        self._a_bound = action_bound
        # 网络结构参数
        self._a_s1_input_units = actor_s1_input_units
        self._a_s2_input_units = actor_s2_input_units
        self._a_s2_input_layers_num = actor_s2_input_layers_num
        self._a_ah_input_units = actor_ah_input_units
        self._a_ah_input_layers_num = actor_ah_input_layers_num
        self._a_middle_layers_num = actor_middle_layers_num
        self._a_middle_layers_units = actor_middle_layers_units
        self._a_enable_bn = actor_bn
        self._c_s1_input_units = critic_input_s1_units
        self._c_s2_input_units = critic_input_s2_units
        self._c_s2_input_layers_num = critic_input_s2_layers_num
        self._c_ah_input_units = critic_input_ah_units
        self._c_ah_input_layers_num = critic_input_ah_layers_num
        self._c_an_input_units = critic_input_an_units
        self._c_middle_layers_num = critic_middle_layers_num
        self._c_middle_layers_units = critic_middle_layers_units
        # 创建网络
        self._build_nets()
        self._s_queue = Queue(maxsize=self._s2_seq_len)  # 序列化状态存储队列，包含s_t
        for i in range(self._s2_seq_len):
            self._s_queue.put([0] * (self._s1_dim + self._s2_dim))
        self._a_queue = Queue(maxsize=self._s2_seq_len - 1)  # 序列化动作存储队列，不包含a_t
        for i in range(self._a_queue.maxsize):
            self._a_queue.put([0] * self._a_dim)
        # 动作网络学习速率，评价网络学习速率
        self._a_optim = torch.optim.AdamW(self._eval_actor_net.parameters(), lr=learning_rate_actor,
                                          weight_decay=actor_weight_decay, amsgrad=True)
        self._c1_optim = torch.optim.AdamW(self._eval1_critic_net.parameters(), lr=learning_rate_critic,
                                           weight_decay=critic1_weight_decay, amsgrad=True)
        self._c2_optim = torch.optim.AdamW(self._eval2_critic_net.parameters(), lr=learning_rate_critic,
                                           weight_decay=critic2_weight_decay, amsgrad=True)
        # 动作网络更新次数，评价网络更新次数
        self._a_replace_cnt, self._c_replace_cnt = 0, 0
        # 收益折扣系数
        self._gamma = reward_decay
        # delayed policy update 参数
        self._tau = soft_replace
        self._delay_eps = delay_update_episode
        # memory大小，mini batch大小
        self._m_size, self._b_size = memory_size, batch_size
        # 动作网络输出探索 noise
        self._noise_type = noise_type
        if noise_type == 'N':
            self._act_noise = NormalActionNoise(mu=np.zeros(self._a_dim), sigma=n_noise_sigma)
        elif noise_type == 'OU':
            self._act_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._a_dim), sigma=ou_noise_sigma)
        elif noise_type == 'AN':
            self._act_noise = AdaptiveNormalActionNoise(
                mu=np.zeros(self._a_dim), sigma_0=an_noise_params[0], sigma_t=an_noise_params[1],
                sigma_d=an_noise_params[2])
        # 目标策略 Smoothing Regularization noise
        self._smt_noise = NormalActionNoise(mu=np.zeros(self._a_dim), sigma=smoothing_sigma)
        # 目标策略 Smoothing Regularization 范围
        self._smt_range = smoothing_range
        # 总学习次数
        self._train_cnt = 0
        # 参数存储目录
        assert mission_path is not None, '任务名称即任务数据存储路径未设置！'
        self._mission_name = mission_path
        if file_path is not None:
            self._params_path = str(os.path.join(self._mission_name, file_path))
        else:
            self._params_path = str(self._mission_name)
        # Replay Buffer 初始化
        self._m_type = memory_type
        if self._m_type == 'Uniform':
            self._memory = ReplayBuffer(self._m_size, self._b_size)
        elif self._m_type == 'Prioritized':
            self._memory = PrioritizedReplayBuffer(
                self._m_size, self._b_size, alpha=memory_alpha, beta0=memory_beta0, beta_inc=memory_beta_increment)
        elif self._m_type == 'Curriculum':
            self._memory = CurriculumReplayBuffer(
                self._m_size, self._b_size, memory_c_omega, memory_sp_lambda, memory_alpha, memory_beta0,
                memory_beta_increment, self._s_dim, self._a_dim, gamma=self._gamma)
        # 创建 summary writer
        self._actor_loss_scalar = None
        self._critic1_loss_scalar = None
        self._critic2_loss_scalar = None
        self._req_save_params = req_save_params
        self._build_writer()

    def _build_writer(self):
        tb_log_path = os.path.join(self._params_path, 'TrainParams')
        if self._req_save_params:
            self._summary_writer = SummaryWriter(tb_log_path)

    def save_net_structure(self):
        """
        保存网络结构到xml文件

        :return: 文件路径
        """
        file_path = []
        actor_input_dict = {'normal_state_dim': self._s1_dim, 'normal_state_input_units': self._a_s1_input_units,
                            'seq_state_dim': self._s2_dim, 'seq_state_len': self._s2_seq_len,
                            'seq_state_input_units': self._a_s2_input_units,
                            'seq_state_input_layers_num': self._a_s2_input_layers_num, 'activate': 'tanh'}
        actor_middle_dict = {'num': self._a_middle_layers_num, 'units': self._a_middle_layers_units,
                             'activate': ['tanh'] * self._a_middle_layers_num}
        actor_output_dict = {'dim': self._a_dim, 'activate': 'tanh'}
        actor_file_path = save_net_structure_to_xml(
            os.path.join(self._params_path, 'ActorNetStructure.xml'), self._mission_name, actor_input_dict,
            actor_middle_dict, actor_output_dict)
        file_path.append(actor_file_path)
        critic_input_block_dict = {
            'state': {'normal_state_dim': self._s1_dim, 'normal_state_input_units': self._c_s1_input_units,
                      'seq_state_dim': self._s2_dim, 'seq_state_len': self._s2_seq_len,
                      'seq_state_input_units': self._c_s2_input_units,
                      'seq_state_input_layers_num': self._c_s2_input_layers_num, 'activate': 'relu'},
            'action': {'dim': self._a_dim, 'units': self._c_ah_input_units, 'activate': 'relu'}}
        critic_middle_block_dict = {'num': self._c_middle_layers_num, 'units': self._c_middle_layers_units,
                                    'activate': ['tanh'] * self._c_middle_layers_num}
        critic_output_block_dict = {'dim': 1, 'activate': 'None'}
        critic_file_path = save_critic_structure_to_xml(
            os.path.join(self._params_path, 'CriticNetStructure.xml'), self._mission_name, critic_input_block_dict,
            critic_middle_block_dict, critic_output_block_dict)
        file_path.append(critic_file_path)
        return file_path

    def _build_actor_network(self) -> PoActorNetwork:
        return PoActorNetwork(
            self._s1_dim, self._a_s1_input_units, self._s2_dim, self._a_s2_input_units, self._s2_seq_len,
            self._a_s2_input_layers_num, self._a_dim, self._a_ah_input_units, self._ah_seq_len,
            self._a_ah_input_layers_num, self._a_middle_layers_num, self._a_middle_layers_units,
            [None] * self._a_middle_layers_num, torch.nn.Tanh(), torch.Tensor(self._a_bound).to(device),
            # [torch.nn.Tanh()] * self._a_middle_layers_num, torch.nn.Tanh(), torch.Tensor(self._a_bound).to(device),
            self._a_enable_bn
        ).to(device)

    def _build_critic_network(self) -> PoCriticNetwork:
        return PoCriticNetwork(
            self._s1_dim, self._c_s1_input_units, torch.nn.ReLU(), self._s2_dim, self._s2_seq_len,
            self._c_s2_input_units, self._c_s2_input_layers_num, self._a_dim, self._ah_seq_len,
            self._c_ah_input_units, self._c_ah_input_layers_num, self._c_an_input_units, torch.nn.ReLU(),
            self._c_middle_layers_num, self._c_middle_layers_units, [torch.nn.ReLU()] * self._c_middle_layers_num, None
        ).to(device)

    def _build_nets(self):
        # 初始化 Actor 网络
        self._eval_actor_net = self._build_actor_network()
        self._tgt_actor_net = self._build_actor_network()
        self._tgt_actor_net.load_state_dict(self._eval_actor_net.state_dict())
        # 初始化 Critic1 网络
        self._eval1_critic_net = self._build_critic_network()
        self._tgt1_critic_net = self._build_critic_network()
        self._tgt1_critic_net.load_state_dict(self._eval1_critic_net.state_dict())
        # 初始化 Critic2 网络
        self._eval2_critic_net = self._build_critic_network()
        self._tgt2_critic_net = self._build_critic_network()
        self._tgt2_critic_net.load_state_dict(self._eval2_critic_net.state_dict())
        # 初始化 loss 存储序列
        self._a_losses_origin_data = []
        self._a_losses_smooth_data = []
        self._a_losses_smooth = MovAvg(100)
        self._c1_losses_origin_data = []
        self._c1_losses_smooth_data = []
        self._c1_losses_smooth = MovAvg(100)
        self._c2_losses_origin_data = []
        self._c2_losses_smooth_data = []
        self._c2_losses_smooth = MovAvg(100)

    def _get_seq_state(self, obv_queue=None) -> np.ndarray:
        # 构建状态输入
        # 首先，构建序列化状态
        s2_seq = np.array([])
        for i in range(self._s2_seq_len):
            if obv_queue is None:
                item = self._s_queue.get()
                self._s_queue.put(item)
            else:
                item = obv_queue.get()
                obv_queue.put(item)
            s2_seq = np.concatenate((s2_seq, item[-self._s2_dim:]))
        return s2_seq

    def _get_seq_action(self, action_queue=None) -> np.ndarray:
        # 构建动作输入序列
        # 首先，构建序列化动作
        ah_seq = np.array([])
        for i in range(self._a_queue.maxsize):
            if action_queue is None:
                item = self._a_queue.get()
                self._a_queue.put(item)
            else:
                item = action_queue.get()
                action_queue.put(item)
            ah_seq = np.concatenate((ah_seq, item[-self._a_dim:]))
        return ah_seq

    def is_curriculum(self) -> bool:
        if self._m_type == 'Curriculum':
            return True
        else:
            return False

    def get_seq_ha(self, s: np.ndarray) -> np.ndarray:
        """
        根据输入的下一时刻st，生成hta
        :param s: 下一时刻状态
        :return: hta，s0,a0,s1,a1……,st-1,at-1,st
        """
        # 复制一份状态序列
        s_queue = Queue(maxsize=self._s2_seq_len)
        for i in range(self._s2_seq_len):
            item = self._s_queue.get()
            s_queue.put(item)
            self._s_queue.put(item)
        # 弹出序列中一个状态，并保存新的状态
        if s_queue.full():
            s_queue.get()
        s_queue.put(s)
        # 首先，构建序列化状态
        s2_seq = self._get_seq_state(s_queue)
        del s_queue
        # 其次，构建序列化动作
        a_seq = self._get_seq_action()
        # 然后，拼接生成h
        x = np.concatenate((s[:self._s1_dim], s2_seq, a_seq))
        return x

    def choose_action(self, obv: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        训练过程中，选择动作
        :param obv: 观测量
        :return: 动作，以及序列化状态
        """
        # 把状态保存起来
        if self._s_queue.full():
            self._s_queue.get()
        self._s_queue.put(obv)
        with torch.no_grad():
            self._eval_actor_net.eval()
            # 构建状态输入
            # 首先，构建序列化状态
            s2_seq = self._get_seq_state()
            # 其次，构建序列化动作
            a_seq = self._get_seq_action()
            # 然后，拼接生成Actor输入
            x = np.concatenate((obv[:self._s1_dim], s2_seq, a_seq))
            a = self._eval_actor_net(torch.Tensor(x[np.newaxis, :]).to(device)).cpu().numpy()[0]
            self._eval_actor_net.train()
            a = np.clip(a + self._act_noise(), a_min=-self._a_bound, a_max=self._a_bound)
            # 把动作保存起来
            if self._a_queue.full():
                self._a_queue.get()
            self._a_queue.put(10.0 * a / self._a_bound)
            return a, x

    def choose_action_test(self, obv: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        测试过程中，选择动作
        :param obv: 观测量
        :return: 动作
        """
        # 把序列化状态保存起来
        if self._s_queue.full():
            self._s_queue.get()
        self._s_queue.put(obv)
        with torch.no_grad():
            self._tgt_actor_net.eval()
            # 构建状态输入
            # 首先，构建序列化状态
            s2_seq = self._get_seq_state()
            # 其次，构建序列化动作
            a_seq = self._get_seq_action()
            # 然后，把序列化状态拼接到普通状态上
            x = np.concatenate((obv[:self._s1_dim], s2_seq, a_seq))
            a = self._tgt_actor_net(torch.Tensor(x[np.newaxis, :]).to(device)).cpu().numpy()[0]
            self._tgt_actor_net.train()
            a = np.clip(a, a_min=-self._a_bound, a_max=self._a_bound)
            # 把动作保存起来
            if self._a_queue.full():
                self._a_queue.get()
            self._a_queue.put(10.0 * a / self._a_bound)
            return a, x

    def store_transition(self, s, a, r, s_, c):
        """
        存储历史经验
        :param s: 当前时刻状态
        :param a: 当前时刻动作
        :param r: 获得的回报
        :param s_: 下一时刻状态
        :param c: 结束条件
        :return:
        """
        self._memory.store_transition(s, a, s_, r, c)

    def curriculum_update_memory(self):
        if self._m_type == 'Curriculum':
            self._memory.asynchronous_update(
                [self._eval1_critic_net, self._eval2_critic_net], [self._tgt1_critic_net, self._tgt2_critic_net],
                self._eval_actor_net)

    def train(self):
        # 随机抽取历史经验
        mini_batch = None
        b_index, weighted_is = None, None
        if self._m_type == 'Uniform':
            # 从 replay buffer 中采样得到 mini batch
            mini_batch = self._memory.sample()
        elif self._m_type == 'Prioritized' or self._m_type == 'Curriculum':
            b_index, mini_batch, weighted_is = self._memory.sample()
        # 若数目不够，则直接退出训练
        if mini_batch is None:
            return
        # 预处理样本集
        b_s = torch.Tensor(mini_batch[:, :self._s_dim]).to(device)
        b_a = torch.Tensor(mini_batch[:, self._s_dim: self._s_dim + self._a_dim]).to(device)
        b_s_ = torch.Tensor(mini_batch[:, self._s_dim + self._a_dim: 2 * self._s_dim + self._a_dim]).to(device)
        b_r = mini_batch[:, -2]
        b_r = torch.Tensor(np.reshape(b_r, [len(b_r), 1])).to(device)
        b_c = mini_batch[:, -1]
        # 训练 Critic 网络
        if weighted_is is not None:
            c_loss, abs_errors = self._train_critics(b_s, b_a, b_r, b_s_, b_c, torch.Tensor(weighted_is).to(device))
        else:
            c_loss, abs_errors = self._train_critics(b_s, b_a, b_r, b_s_, b_c, weighted_is)
        if self._m_type == 'Prioritized' or self._m_type == 'Curriculum':
            self._memory.batch_update(b_index, abs_errors.detach().cpu().numpy())
        # 保存训练损失 critic loss
        c1_loss_val = float(c_loss[0].detach().cpu().numpy())
        self._c1_losses_origin_data.append(c1_loss_val)
        self._c1_losses_smooth_data.append(self._c1_losses_smooth.update(c1_loss_val))
        c2_loss_val = float(c_loss[1].detach().cpu().numpy())
        self._c2_losses_origin_data.append(c2_loss_val)
        self._c2_losses_smooth_data.append(self._c1_losses_smooth.update(c2_loss_val))
        self._write_critics_loss(c1_loss_val, c2_loss_val)
        # Delayed Policy Updates
        if self._train_cnt % self._delay_eps == 0:
            # 训练 Actor 网络
            a_loss = self._train_actor(b_s)
            # 保存训练损失 actor loss
            a_loss_val = float(a_loss.detach().cpu().numpy())
            self._a_losses_origin_data.append(a_loss_val)
            self._a_losses_smooth_data.append(self._a_losses_smooth.update(a_loss_val))
            self._write_actor_loss(a_loss_val)
            # 软更新目标网络参数
            self._soft_update()
        self._train_cnt += 1

    def _train_actor(self, batch_s: torch.Tensor):
        """
        训练 Actor 网络
        :param batch_s: states of batch.
        :return: Actor Loss
        """
        # 随机选择一个 critic 生成评估值
        if np.random.randint(0, 2, dtype=int) == 0:
            eval_c = self._eval1_critic_net(torch.concat([batch_s, self._eval_actor_net(batch_s)], dim=1))
        else:
            eval_c = self._eval2_critic_net(torch.concat([batch_s, self._eval_actor_net(batch_s)], dim=1))
        # 计算网络损失
        loss = -torch.mean(eval_c)
        loss.requires_grad_(True)
        # 反向传播
        self._a_optim.zero_grad()
        loss.backward()
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_value_(self._eval_actor_net.parameters(), 100)
        self._a_optim.step()
        return loss

    def _train_critics(
            self, batch_s: torch.Tensor, batch_a: torch.Tensor, batch_r: torch.Tensor, batch_s_: torch.Tensor,
            batch_c: torch.Tensor, is_weights: Union[torch.Tensor, None]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # 损失和 PER 的采样概率
        c1_loss = None
        c2_loss = None
        c1_abs_errors = None
        c2_abs_errors = None
        # 计算 Clipped Double Q-Learning Target
        epsilon = torch.Tensor(np.clip(self._smt_noise(), -self._smt_range, self._smt_range)).to(device)
        with torch.no_grad():
            tgt_a = self._tgt_actor_net(batch_s) + epsilon
            tgt1_c = self._tgt1_critic_net(torch.concat([batch_s_, tgt_a], dim=1))
            tgt2_c = self._tgt2_critic_net(torch.concat([batch_s_, tgt_a], dim=1))
            y_target = torch.add(batch_r, torch.multiply(torch.minimum(tgt1_c, tgt2_c), self._gamma))  # 不考虑终止状态的学习目标
            termination_index = np.where(batch_c > 0)[0].tolist()
            # termination_index = np.where(batch_r.cpu().numpy() >= self._term_rwd)[0]  # 寻找终止状态样本索引
            if len(termination_index) > 0:
                y_target[termination_index] = batch_r[termination_index]  # 若为终止状态，则目标更换为回报值，不考虑后续影响
        # 优化 Critic 1
        c1_eval = self._eval1_critic_net(torch.concat([batch_s, batch_a], dim=1))
        if self._m_type == 'Uniform':
            # 计算损失 loss
            criterion = torch.nn.HuberLoss()
            c1_loss = criterion(y_target, c1_eval)
        elif self._m_type == 'Prioritized' or self._m_type == 'Curriculum':
            # 保存 loss ，然后更新 Priorities
            c1_abs_errors = torch.mean(torch.abs(y_target - c1_eval), dim=1)
            criterion = CriticLoss(is_weights)
            c1_loss = criterion(y_target, c1_eval)
        # 反向传播
        self._c1_optim.zero_grad()
        c1_loss.requires_grad_(True)
        c1_loss.backward()
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_value_(self._eval1_critic_net.parameters(), 100)
        self._c1_optim.step()
        # 优化 Critic 2
        c2_eval = self._eval2_critic_net(torch.concat([batch_s, batch_a], dim=1))
        if self._m_type == 'Uniform':
            # 优化网络并保存 loss
            criterion = torch.nn.HuberLoss()
            c2_loss = criterion(y_target, c2_eval)
        elif self._m_type == 'Prioritized' or self._m_type == 'Curriculum':
            # 优化网络并保存 loss ，然后更新 Priorities
            c2_abs_errors = torch.mean(torch.abs(y_target - c2_eval), dim=1)
            criterion = CriticLoss(is_weights)
            c2_loss = criterion(y_target, c2_eval)
        # 反向传播
        self._c2_optim.zero_grad()
        c2_loss.requires_grad_(True)
        c2_loss.backward()
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_value_(self._eval2_critic_net.parameters(), 100)
        self._c2_optim.step()
        # 保存损失和 PER 的采样概率
        loss = [c1_loss, c2_loss]
        abs_errors = None
        if self._m_type == 'Prioritized' or self._m_type == 'Curriculum':
            abs_errors = torch.minimum(c1_abs_errors, c2_abs_errors)
        return loss, abs_errors

    def _soft_update(self):
        # 更新 actor 网络参数
        tgt_a_state_dict = self._tgt_actor_net.state_dict()
        eval_a_state_dict = self._eval_actor_net.state_dict()
        for key in eval_a_state_dict:
            tgt_a_state_dict[key] = (1 - self._tau) * tgt_a_state_dict[key] + self._tau * eval_a_state_dict[key]
        self._tgt_actor_net.load_state_dict(tgt_a_state_dict)
        # 更新 critic1 网络参数
        tgt1_c_state_dict = self._tgt1_critic_net.state_dict()
        eval1_c_state_dict = self._eval1_critic_net.state_dict()
        for key in eval1_c_state_dict:
            tgt1_c_state_dict[key] = (1 - self._tau) * tgt1_c_state_dict[key] + self._tau * eval1_c_state_dict[key]
        self._tgt1_critic_net.load_state_dict(tgt1_c_state_dict)
        # 更新 critic2 网络参数
        tgt2_c_state_dict = self._tgt2_critic_net.state_dict()
        eval2_c_state_dict = self._eval2_critic_net.state_dict()
        for key in eval2_c_state_dict:
            tgt2_c_state_dict[key] = (1 - self._tau) * tgt2_c_state_dict[key] + self._tau * eval2_c_state_dict[key]
        self._tgt2_critic_net.load_state_dict(tgt2_c_state_dict)

    def _write_critics_loss(self, c1_loss: float, c2_loss: float):
        if self._req_save_params:
            self._summary_writer.add_scalar("Critic1 Loss", c1_loss, self._train_cnt)
            self._summary_writer.add_scalar("Critic2 Loss", c2_loss, self._train_cnt)

    def _write_actor_loss(self, a_loss: float):
        if self._req_save_params:
            self._summary_writer.add_scalar("Actor Loss", a_loss, int(self._train_cnt / self._delay_eps))

    def reset_noise(self):
        self._act_noise.reset()

    def update_noise(self):
        if self._noise_type == 'AN':
            self._act_noise.update()

    def reset_seq(self):
        while not self._s_queue.empty():
            self._s_queue.get()
        for i in range(self._s_queue.maxsize):
            self._s_queue.put([0] * (self._s1_dim + self._s2_dim))
        while not self._a_queue.empty():
            self._a_queue.get()
        for i in range(self._a_queue.maxsize):
            self._a_queue.put([0] * self._a_dim)

    def write_simulation_params(self, reward: float, s_rate: float):
        if self._req_save_params:
            self._summary_writer.add_scalar("Episode Rewards", reward, self._train_cnt)
            self._summary_writer.add_scalar("Successful Rate", s_rate, self._train_cnt)

    def load_net_params(self, params_dir: str):
        """
        加载网络参数
        :param params_dir: 参数路径
        :return:
        """
        # 加载历史参数
        params_dir = os.path.join(params_dir, 'NetParams')
        if os.path.isdir(params_dir):
            net_params_path = os.path.join(params_dir, 'Critic1.pkl')
            self._eval1_critic_net.load_state_dict(torch.load(net_params_path))
            self._tgt1_critic_net.load_state_dict(torch.load(net_params_path))
            net_params_path = os.path.join(params_dir, 'Critic2.pkl')
            self._eval2_critic_net.load_state_dict(torch.load(net_params_path))
            self._tgt2_critic_net.load_state_dict(torch.load(net_params_path))
            net_params_path = os.path.join(params_dir, 'Actor.pkl')
            self._eval_actor_net.load_state_dict(torch.load(net_params_path))
            self._tgt_actor_net.load_state_dict(torch.load(net_params_path))

    def save_net_params(self, suffix=None):
        """
        保存网络参数
        :param suffix: 参数保存子文件夹
        :return: 网络参数保存路径
        """
        if suffix is not None:
            params_dir = suffix
        else:
            params_dir = self._params_path
        params_dir = os.path.join(params_dir, 'NetParams')
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        actor_params_path = os.path.join(params_dir, 'Actor.pkl')
        torch.save(self._eval_actor_net.cpu().state_dict(), actor_params_path)
        self._eval_actor_net.to(device)
        critic1_params_path = os.path.join(params_dir, 'Critic1.pkl')
        torch.save(self._eval1_critic_net.cpu().state_dict(), critic1_params_path)
        self._eval1_critic_net.to(device)
        critic2_params_path = os.path.join(params_dir, 'Critic2.pkl')
        torch.save(self._eval2_critic_net.cpu().state_dict(), critic2_params_path)
        self._eval2_critic_net.to(device)
        return params_dir

    def get_critic_loss(self, num=1):
        if num == 1:
            if len(self._c1_losses_origin_data) < 1:
                return .0
            else:
                return self._c1_losses_origin_data[-1]
        elif num == 2:
            if len(self._c2_losses_origin_data) < 1:
                return .0
            else:
                return self._c2_losses_origin_data[-1]
        return None

    def get_actor_loss(self):
        if len(self._a_losses_origin_data) < 1:
            return .0
        else:
            return self._a_losses_origin_data[-1]
