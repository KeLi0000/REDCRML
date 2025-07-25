import os
from datetime import datetime
from train_test import train, fast_create_potd3_rl_group, save_model_params
from exp_init import create_drop_env

MAX_EPISODES = 1100
MAX_MEM = 100000
BATCH_SIZE = 512
TEST_EPISODES = 10

if __name__ == '__main__':
    # 1、构造环境对象
    tmp_env = create_drop_env()
    alg_name = 'TD3-RAC-CER'
    train_name = os.path.join(alg_name, datetime.now().strftime("%y%m%d_%H%M%S"))
    # 2、创建算法对象，Experience Replay为Curriculum型
    old_params_path = None  # 'PBA/DropNav-250106_173937'  # 网络历史参数路径
    rl1, rl2, rl3 = fast_create_potd3_rl_group(
        tmp_env, train_name, MAX_MEM, BATCH_SIZE, [15, 15, 12],
        [128, 128, 4, 128, 2, 64, 4, 256, 128, 128, 4, 128, 2, 4, 256],
        [128, 128, 4, 128, 2, 64, 4, 256, 128, 128, 4, 128, 2, 4, 256],
        [128, 128, 4, 128, 2, 64, 4, 256, 128, 128, 4, 128, 2, 4, 256],
        'Curriculum', old_params_path)
    # 3、训练
    success_rates_p, episode_rewards_p = train(
        env=tmp_env, rl=[rl1, rl2, rl3], enable_auto_aiming=True, mission_name=train_name, max_episodes=MAX_EPISODES,
        learn_eps=100, noise_eps=200)
    # 4、保存网络参数
    save_model_params([rl1, rl2, rl3])