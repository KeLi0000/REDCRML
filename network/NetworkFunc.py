# _*_ coding : utf-8 _*_
# @Time: 2025/3/6 14:34
# @File : network_func.py
# @Project : iatdrl2
# @Description : 神经网络相关函数
# @Author : Ke LI
import torch
import torch.nn as nn


def _reset_layer_weights_bias(layer: nn.Linear, mean, std, val):
    torch.nn.init.trunc_normal_(layer.weight, mean, std, a=-2 * std, b=2 * std)  # 初始化权重
    torch.nn.init.constant_(layer.bias, val)  # 初始化偏差
