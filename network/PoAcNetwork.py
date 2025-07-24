# _*_ coding : utf-8 _*_
# @Time: 2025/3/6 14:28
# @File : PoAcNetwork.py
# @Project : iatdrl2
# @Author : 李珂
import torch
import torch.nn as nn
from network.NetworkFunc import _reset_layer_weights_bias
from network.Func import device


class PoCriticNetwork(nn.Module):
    def __init__(self, s1_fcn_input_dim: int, s1_fcn_input_units, s1_fcn_input_func, s2_lstm_input_dim: int,
                 s2_lstm_input_seq_len, s2_lstm_input_units, s2_lstm_input_layers_num, a_dim: int,
                 ah_lstm_input_seq_len, ah_lstm_input_units, ah_lstm_input_layers_num, an_fcn_input_units,
                 an_fcn_input_func, middle_layers_num, middle_layers_units, middle_layers_func, output_layer_func):
        """
        用于POMDPs的Critic网络
        :param s1_fcn_input_dim:
        :param s1_fcn_input_units:
        :param s1_fcn_input_func:
        :param s2_lstm_input_dim:
        :param s2_lstm_input_seq_len:
        :param s2_lstm_input_units:
        :param s2_lstm_input_layers_num:
        :param a_dim:
        :param ah_lstm_input_units:
        :param middle_layers_num:
        :param middle_layers_units:
        :param middle_layers_func:
        :param output_layer_func:
        """
        super().__init__()
        self._s1_fc_dim = s1_fcn_input_dim
        self._s2_lstm_dim = s2_lstm_input_dim
        self._s2_lstm_seq_len = s2_lstm_input_seq_len
        self._a_dim = a_dim
        self._ah_lstm_seq_len = ah_lstm_input_seq_len
        # 创建无序列化状态1输入块
        s1_block = [nn.Linear(s1_fcn_input_dim, s1_fcn_input_units)]  # 全连接层
        _reset_layer_weights_bias(s1_block[0], 0.0, 0.2, 0.1)  # 初始化权重和偏差
        if s1_fcn_input_func is not None:
            s1_block.append(s1_fcn_input_func)  # 激活函数
        self._s1_fcn_input_layer = nn.Sequential(*s1_block)
        # 创建序列化状态2输入块
        s2_lstm_block = [nn.LSTM(s2_lstm_input_dim, s2_lstm_input_units, s2_lstm_input_layers_num, batch_first=True,
                                 bidirectional=True)]
        self._s2_lstm_input_layer = nn.Sequential(*s2_lstm_block)
        # 创建历史动作LSTM输入块
        ah_lstm_block = [nn.LSTM(a_dim, ah_lstm_input_units, ah_lstm_input_layers_num, batch_first=True,
                                 bidirectional=True)]
        self._ah_lstm_input_layer = nn.Sequential(*ah_lstm_block)
        # 创建当前动作FCN输入快
        an_fcn_block = [nn.Linear(a_dim, an_fcn_input_units)]  # 全连接层
        _reset_layer_weights_bias(an_fcn_block[0], 0.0, 0.2, 0.1)  # 初始化权重和偏差
        if an_fcn_input_func is not None:
            an_fcn_block.append(an_fcn_input_func)  # 激活函数
        self._an_fcn_input_layer = nn.Sequential(*an_fcn_block)
        # 创建中间输入块
        middle_blocks = [
            nn.Linear(s1_fcn_input_units + s2_lstm_input_units * 2 + ah_lstm_input_units * 2 + an_fcn_input_units,
                      middle_layers_units[0])]  # 全连接层
        _reset_layer_weights_bias(middle_blocks[0], 0.0, 0.2, 0.1)  # 初始化权重和偏差
        if middle_layers_func[0] is not None:
            middle_blocks.append(middle_layers_func[0])  # 激活函数
        for i in range(middle_layers_num - 1):
            middle_blocks.append(nn.Linear(middle_layers_units[i], middle_layers_units[i + 1]))  # 全连接层
            _reset_layer_weights_bias(middle_blocks[-1], 0.0, 0.2, 0.1)  # 初始化权重和偏差
            if middle_layers_func[i + 1] is not None:
                middle_blocks.append(middle_layers_func[i + 1])  # 激活函数
        self._middle_blocks = nn.Sequential(*middle_blocks)
        # 创建输出层
        output_block = [nn.Linear(middle_layers_units[-1], 1)]  # 全连接层
        _reset_layer_weights_bias(output_block[0], 0.0, 0.2, 0.1)  # 初始化权重和偏差
        if output_layer_func is not None:
            output_block.append(output_layer_func)  # 激活函数
        self._output_layer = nn.Sequential(*output_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 切分输入
        x_s1, x_s2, x_ah, x_an = torch.split(x, [self._s1_fc_dim, self._s2_lstm_dim * self._s2_lstm_seq_len,
                                                 self._a_dim * self._ah_lstm_seq_len, self._a_dim], 1)
        x_s2_seq = torch.reshape(x_s2, [x_s2.shape[0], self._s2_lstm_seq_len, self._s2_lstm_dim])
        x_ah_seq = torch.reshape(x_ah, [x_ah.shape[0], self._ah_lstm_seq_len, self._a_dim])
        # 计算状态全连接部分
        x_s1_mdl = self._s1_fcn_input_layer(x_s1)
        # 计算状态LSTM部分
        x_s2_mdl, _ = self._s2_lstm_input_layer(x_s2_seq)
        # 计算历史动作LSTM模块
        x_ah_mdl, _ = self._ah_lstm_input_layer(x_ah_seq)
        # 计算当前动作FCN模块
        x_an_mdl = self._an_fcn_input_layer(x_an)
        # 连接全连接层输入
        x_mdl = torch.concat([x_s1_mdl, x_s2_mdl[:, -1, :], x_ah_mdl[:, -1, :], x_an_mdl], 1)
        # 计算中间层
        x_mdl = self._middle_blocks(x_mdl)
        # 计算输出层
        x_output = self._output_layer(x_mdl)
        return x_output

    def to_cpu(self):
        self._s1_fcn_input_layer = self._s1_fcn_input_layer.to('cpu')
        self._s2_lstm_input_layer = self._s2_lstm_input_layer.to('cpu')
        self._ah_lstm_input_layer = self._ah_lstm_input_layer.to('cpu')
        self._an_fcn_input_layer = self._an_fcn_input_layer.to('cpu')
        self._middle_blocks = self._middle_blocks.to('cpu')
        self._output_layer = self._output_layer.to('cpu')

    def to_gpu(self):
        self._s1_fcn_input_layer = self._s1_fcn_input_layer.to(device)
        self._s2_lstm_input_layer = self._s2_lstm_input_layer.to(device)
        self._ah_lstm_input_layer = self._ah_lstm_input_layer.to(device)
        self._an_fcn_input_layer = self._an_fcn_input_layer.to(device)
        self._middle_blocks = self._middle_blocks.to(device)
        self._output_layer = self._output_layer.to(device)


class PoActorNetwork(nn.Module):
    def __init__(self, s1_fcn_dim: int, s1_fcn_input_units, s2_lstm_dim: int, s2_lstm_input_units,
                 s2_lstm_input_seq_len, s2_lstm_input_layers_num, a_dim: int, ah_lstm_input_units,
                 ah_lstm_input_seq_len, ah_lstm_input_layers_num, middle_layers_num, middle_layers_units,
                 middle_layers_func, output_layer_func, output_gain, enable_bn=True):
        """
        用于POMDPs的Actor网络
        :param s1_fcn_dim:
        :param s1_fcn_input_units:
        :param s2_lstm_dim:
        :param s2_lstm_input_units:
        :param s2_lstm_input_seq_len:
        :param s2_lstm_input_layers_num:
        :param a_dim:
        :param ah_lstm_input_units:
        :param ah_lstm_input_seq_len:
        :param ah_lstm_input_layers_num:
        :param middle_layers_num:
        :param middle_layers_units:
        :param middle_layers_func:
        :param output_layer_func:
        :param output_gain:
        :param enable_bn:
        """
        super().__init__()
        self._s1_dim = s1_fcn_dim
        self._s2_dim = s2_lstm_dim
        self._s2_seq_len = s2_lstm_input_seq_len
        self._a_lstm_dim = a_dim
        self._a_lstm_seq_len = ah_lstm_input_seq_len
        # 创建状态输入全连接层部分
        s1_input_block = [nn.Linear(s1_fcn_dim, s1_fcn_input_units)]  # 全连接层
        _reset_layer_weights_bias(s1_input_block[0], 0.0, 0.1, 0.0)  # 初始化权重和偏差
        if enable_bn:
            s1_input_block.append(nn.BatchNorm1d(s1_fcn_input_units))  # Batch Normalization 层
        self._s1_fc_input_layer = nn.Sequential(*s1_input_block)
        # 创建状态输入LSTM部分
        s2_lstm_block = [nn.LSTM(s2_lstm_dim, s2_lstm_input_units, s2_lstm_input_layers_num, batch_first=True,
                                 bidirectional=True)]
        self._s2_lstm_input_layer = nn.Sequential(*s2_lstm_block)
        # 创建动作输入LSTM部分
        ah_lstm_block = [nn.LSTM(a_dim, ah_lstm_input_units, ah_lstm_input_layers_num, batch_first=True,
                                 bidirectional=True)]
        self._ah_lstm_input_layer = nn.Sequential(*ah_lstm_block)
        # 创建中间层
        middle_blocks = [nn.Linear(s1_fcn_input_units + s2_lstm_input_units * 2 + ah_lstm_input_units * 2,
                                   middle_layers_units[0])]  # 全连接层
        _reset_layer_weights_bias(middle_blocks[0], 0.0, 0.1, 0.0)  # 初始化权重和偏差
        if enable_bn:
            middle_blocks.append(nn.BatchNorm1d(middle_layers_units[0]))  # Batch Normalization 层
        if middle_layers_func[0] is not None:
            middle_blocks.append(middle_layers_func[0])  # 激活函数
        for i in range(middle_layers_num - 1):
            middle_blocks.append(nn.Linear(middle_layers_units[i], middle_layers_units[i + 1]))  # 全连接层
            _reset_layer_weights_bias(middle_blocks[-1], 0.0, 0.1, 0.0)  # 初始化权重和偏差
            if enable_bn:
                middle_blocks.append(nn.BatchNorm1d(middle_layers_units[i + 1]))  # Batch Normalization 层
            if middle_layers_func[i + 1] is not None:
                middle_blocks.append(middle_layers_func[i + 1])  # 激活函数
        self._middle_blocks = nn.Sequential(*middle_blocks)
        # 创建输出层
        output_block = [nn.Linear(middle_layers_units[-1], a_dim)]  # 全连接层
        _reset_layer_weights_bias(output_block[-1], 0.0, 0.1, 0.0)  # 初始化权重和偏差
        if enable_bn:
            output_block.append(nn.BatchNorm1d(a_dim))  # Batch Normalization 层
        if output_layer_func is not None:
            output_block.append(output_layer_func)  # 激活函数
        self._output_layer = nn.Sequential(*output_block)
        # 记录输出增益
        self._output_gain = output_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 切分输入
        x_s1, x_s2, x_a = torch.split(x, [self._s1_dim, self._s2_dim * self._s2_seq_len,
                                          self._a_lstm_dim * self._a_lstm_seq_len], 1)
        x_s2_seq = torch.reshape(x_s2, [x_s2.shape[0], self._s2_seq_len, self._s2_dim])
        x_a_seq = torch.reshape(x_a, [x_a.shape[0], self._a_lstm_seq_len, self._a_lstm_dim])
        # 计算状态全连接部分
        x_s1_mdl = self._s1_fc_input_layer(x_s1)
        # 计算状态LSTM部分
        x_s2_mdl, _ = self._s2_lstm_input_layer(x_s2_seq)
        # 计算动作LSTM部分
        x_a_mdl, _ = self._ah_lstm_input_layer(x_a_seq)
        # 连接3个输入块的输出向量
        x_mdl = torch.concat([x_s1_mdl, x_s2_mdl[:, -1, :], x_a_mdl[:, -1, :]], 1)
        # 计算中间层
        x = self._middle_blocks(x_mdl)
        # 计算输出层
        y = self._output_layer(x)
        y_output = torch.multiply(y, self._output_gain)
        return y_output

    def to_cpu(self):
        self._s1_fc_input_layer = self._s1_fc_input_layer.to('cpu')
        self._s2_lstm_input_layer = self._s2_lstm_input_layer.to('cpu')
        self._ah_lstm_input_layer = self._ah_lstm_input_layer.to('cpu')
        self._middle_blocks = self._middle_blocks.to('cpu')
        self._output_layer = self._output_layer.to('cpu')
        self._output_gain = self._output_gain.to('cpu')

    def to_gpu(self):
        self._s1_fc_input_layer = self._s1_fc_input_layer.to(device)
        self._s2_lstm_input_layer = self._s2_lstm_input_layer.to(device)
        self._ah_lstm_input_layer = self._ah_lstm_input_layer.to(device)
        self._middle_blocks = self._middle_blocks.to(device)
        self._output_layer = self._output_layer.to(device)
        self._output_gain = self._output_gain.to(device)


if __name__ == '__main__':
    # critic = PoCriticNetwork(
    #     3, 128, nn.ReLU(), 4, 8, 128, 4, 2, 128, nn.ReLU(), 2, [128, 128], [nn.ReLU()] * 2, None)
    # x_critic = torch.randn(8, 3 + 4 * 8 + 2)
    # y_critic = critic(x_critic)
    # print(x_critic)
    # print(y_critic)
    # actor = PoActorNetwork(3, 128, 4, 128, 4, 4, 2, 4, [128] * 4, [nn.Tanh()] * 4, nn.Tanh(), 2)
    # x_actor = torch.randn(4, 3 + 4 * 4)
    # y_actor = actor(x_actor)
    # print(x_actor)
    # print(y_actor)
    pass
