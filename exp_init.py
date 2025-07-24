# _*_ coding: utf-8 _*_
# @Time: 2025/4/30 15:27
# @File: experiments_initialization.py
# @Project: Navigation
# @Description: 创建环境和智能体
# @Author: LI Ke
import os
import shutil
from typing import List

import gymdynadrop
import numpy as np
from gymdynadrop.envs.dyna_drop_nav import DynaDropNav
from iatdrl2 import DeepDeterministicPolicyGradient as Ddpg
from iatdrl2 import PartialObservedDeepDeterministicPolicyGradient as PoDdpg


def check_dir_path(dir_path: str):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def create_drop_env() -> DynaDropNav:
    # 生成环境
    drop_env = gymdynadrop.make('dynadropnav-v0')
    drop_env = drop_env.unwrapped
    drop_env.enable_train_mode()
    drop_env.set_work_type(1)
    drop_env.set_mission_type('Nav')
    return drop_env


def create_params_path(rwd_type: str, mission_name: str):
    params_path = os.path.join("CONTRAST", rwd_type + '-' + mission_name)
    test_params_path = os.path.join(params_path, "TestParams")
    check_dir_path(test_params_path)
    acmi_params_path = os.path.join(params_path, "ACMIParams")
    check_dir_path(acmi_params_path)
    figs_params_path = os.path.join("CONTRAST", "Figs")
    check_dir_path(figs_params_path)
    return test_params_path, acmi_params_path, figs_params_path
