# _*_ coding : utf-8 _*_
# @Time: 2025/3/6 20:35
# @File : td3_func.py
# @Project : iatdrl2
# @Description : TODO:
# @Author : Your Name
import platform
import torch
import torch.nn as nn

os_type = platform.system()
if os_type == 'Darwin':
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CriticLoss(torch.nn.Module):
    def __init__(self, is_weights):
        super().__init__()
        self._is_weights = is_weights

    def forward(self, q_eval: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
        se = torch.square(q_tgt - q_eval)
        w_se = self._is_weights * se
        return torch.mean(w_se)


class ActorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return -torch.mean(q)

def check_nan_in_parameters(model: nn.Module):
    """
    检查模型参数是否包含 nan
    :param model: 待检查的模型
    :return: 如果包含 nan 返回 True，否则返回 False
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"参数 {name} 包含 nan！")
            return True
    return False
