import ctypes
import os
import pickle as pkl
import time
from queue import Queue
from typing import List

import numpy as np
from gymdynadrop.envs.drop_sim_core import SimModelState
from gymdynadrop.envs.dyna_drop_nav import DynaDropNav
from redcrml import REDCRML

from exp_init import check_dir_path


def save_pkl_files(pkl_path: str, pkl_action, pkl_reward, pkl_state, pkl_other):
    try:
        actions_pkl_path = os.path.join(pkl_path, 'actions.pkl')
        with open(actions_pkl_path, 'wb') as f:
            pkl.dump(pkl_action, f)
        rewards_pkl_path = os.path.join(pkl_path, 'rewards.pkl')
        with open(rewards_pkl_path, 'wb') as f:
            pkl.dump(pkl_reward, f)
        states_pkl_path = os.path.join(pkl_path, 'states.pkl')
        with open(states_pkl_path, 'wb') as f:
            pkl.dump(pkl_state, f)
        others_pkl_path = os.path.join(pkl_path, 'others.pkl')
        with open(others_pkl_path, 'wb') as f:
            pkl.dump(pkl_other, f)
    except IOError:
        print(pkl_path + '数据保存出错！')
    else:
        print(pkl_path + '数据保存成功！')


def fast_create_REDCRML_rl_group(env: DynaDropNav, name, mem_size, batch_size, state_seq_len: List, rl0_net_struct: List,
                               rl1_net_struct: List, rl2_net_struct: List, memory_type: str = 'Curriculum',
                               params_path=None) -> List[REDCRML]:
    rl0, rl1, rl2 = None, None, None
    if memory_type == 'Curriculum':
        rl0 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(1), action_bound=env.guidance_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(1),
            state_seq_len=state_seq_len[0], critic_input_s1_units=rl0_net_struct[0],
            critic_input_s2_units=rl0_net_struct[1],
            critic_input_s2_layers_num=rl0_net_struct[2], critic_input_ah_units=rl0_net_struct[3],
            critic_input_ah_layers_num=rl0_net_struct[4], critic_input_an_units=rl0_net_struct[5],
            critic_middle_layers_num=rl0_net_struct[6],
            critic_middle_layers_units=[rl0_net_struct[7]] * rl0_net_struct[6],
            actor_s1_input_units=rl0_net_struct[8], actor_s2_input_units=rl0_net_struct[9],
            actor_s2_input_layers_num=rl0_net_struct[10], actor_ah_input_units=rl0_net_struct[11],
            actor_ah_input_layers_num=rl0_net_struct[12], actor_middle_layers_num=rl0_net_struct[13],
            actor_middle_layers_units=[rl0_net_struct[14]] * rl0_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-4,
            memory_type='Curriculum', memory_alpha=0.3, memory_beta0=0.1, memory_beta_increment=0.0001,
            memory_size=mem_size, noise_type='AN', an_noise_params=np.array([1.0, 0.01, 1.001]),
            batch_size=batch_size, file_path='rl0', req_save_params=True
        )
        rl1 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(2), action_bound=env.guidance_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(2),
            state_seq_len=state_seq_len[1], critic_input_s1_units=rl1_net_struct[0],
            critic_input_s2_units=rl1_net_struct[1],
            critic_input_s2_layers_num=rl1_net_struct[2], critic_input_ah_units=rl1_net_struct[3],
            critic_input_ah_layers_num=rl1_net_struct[4], critic_input_an_units=rl1_net_struct[5],
            critic_middle_layers_num=rl1_net_struct[6],
            critic_middle_layers_units=[rl1_net_struct[7]] * rl1_net_struct[6],
            actor_s1_input_units=rl1_net_struct[8], actor_s2_input_units=rl1_net_struct[9],
            actor_s2_input_layers_num=rl1_net_struct[10], actor_ah_input_units=rl1_net_struct[11],
            actor_ah_input_layers_num=rl1_net_struct[12], actor_middle_layers_num=rl1_net_struct[13],
            actor_middle_layers_units=[rl1_net_struct[14]] * rl1_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-4,
            memory_type='Curriculum', memory_alpha=0.2, memory_beta0=0.1, memory_beta_increment=0.00001,
            memory_size=mem_size,  # 课程化采样
            noise_type='AN', an_noise_params=np.array([1.0, 0.01, 1.001]),
            batch_size=int(batch_size / 2), file_path='rl1', req_save_params=True
        )
        rl2 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(3), action_bound=env.aim_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(3),
            state_seq_len=state_seq_len[2], critic_input_s1_units=rl2_net_struct[0],
            critic_input_s2_units=rl2_net_struct[1],
            critic_input_s2_layers_num=rl2_net_struct[2], critic_input_ah_units=rl2_net_struct[3],
            critic_input_ah_layers_num=rl2_net_struct[4], critic_input_an_units=rl2_net_struct[5],
            critic_middle_layers_num=rl2_net_struct[6],
            critic_middle_layers_units=[rl2_net_struct[7]] * rl2_net_struct[6],
            actor_s1_input_units=rl2_net_struct[8], actor_s2_input_units=rl2_net_struct[9],
            actor_s2_input_layers_num=rl2_net_struct[10], actor_ah_input_units=rl2_net_struct[11],
            actor_ah_input_layers_num=rl2_net_struct[12], actor_middle_layers_num=rl2_net_struct[13],
            actor_middle_layers_units=[rl2_net_struct[14]] * rl2_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-5,
            memory_type='Curriculum', memory_alpha=0.3, memory_beta0=0.1, memory_size=mem_size,  # 课程化采样
            noise_type='AN', an_noise_params=np.array([0.05, 0.0001, 1.01]),
            batch_size=batch_size, file_path='rl2', req_save_params=True
        )
    elif memory_type == 'Prioritized':
        rl0 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(1), action_bound=env.guidance_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(1),
            state_seq_len=state_seq_len[0], critic_input_s1_units=rl0_net_struct[0],
            critic_input_s2_units=rl0_net_struct[1],
            critic_input_s2_layers_num=rl0_net_struct[2], critic_input_ah_units=rl0_net_struct[3],
            critic_input_ah_layers_num=rl0_net_struct[4], critic_input_an_units=rl0_net_struct[5],
            critic_middle_layers_num=rl0_net_struct[6],
            critic_middle_layers_units=[rl0_net_struct[7]] * rl0_net_struct[6],
            actor_s1_input_units=rl0_net_struct[8], actor_s2_input_units=rl0_net_struct[9],
            actor_s2_input_layers_num=rl0_net_struct[10], actor_ah_input_units=rl0_net_struct[11],
            actor_ah_input_layers_num=rl0_net_struct[12], actor_middle_layers_num=rl0_net_struct[13],
            actor_middle_layers_units=[rl0_net_struct[14]] * rl0_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-4,
            memory_type='Prioritized', memory_alpha=0.3, memory_beta0=0.1, memory_beta_increment=0.00001,
            memory_size=mem_size,  # 优先级采样
            noise_type='AN', an_noise_params=np.array([1.0, 0.01, 1.001]),
            batch_size=batch_size, file_path='rl0', req_save_params=True
        )
        rl1 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(2), action_bound=env.guidance_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(2),
            state_seq_len=state_seq_len[1], critic_input_s1_units=rl1_net_struct[0],
            critic_input_s2_units=rl1_net_struct[1],
            critic_input_s2_layers_num=rl1_net_struct[2], critic_input_ah_units=rl1_net_struct[3],
            critic_input_ah_layers_num=rl1_net_struct[4], critic_input_an_units=rl1_net_struct[5],
            critic_middle_layers_num=rl1_net_struct[6],
            critic_middle_layers_units=[rl1_net_struct[7]] * rl1_net_struct[6],
            actor_s1_input_units=rl1_net_struct[8], actor_s2_input_units=rl1_net_struct[9],
            actor_s2_input_layers_num=rl1_net_struct[10], actor_ah_input_units=rl1_net_struct[11],
            actor_ah_input_layers_num=rl1_net_struct[12], actor_middle_layers_num=rl1_net_struct[13],
            actor_middle_layers_units=[rl1_net_struct[14]] * rl1_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-5,
            memory_type='Prioritized', memory_alpha=0.3, memory_beta0=0.1, memory_beta_increment=0.00001,
            memory_size=mem_size,  # 优先级采样
            noise_type='AN', an_noise_params=np.array([1.0, 0.01, 1.001]),
            batch_size=int(batch_size / 2), file_path='rl1', req_save_params=True
        )
        rl2 = REDCRML(
            mission_path=name, action_dim=env.get_action_dim(3), action_bound=env.aim_action_space.high,
            normal_state_dim=env.get_sel_task_normal_state_dim(), seq_state_dim=env.get_sel_task_seq_state_dim(3),
            state_seq_len=state_seq_len[2], critic_input_s1_units=rl2_net_struct[0],
            critic_input_s2_units=rl2_net_struct[1],
            critic_input_s2_layers_num=rl2_net_struct[2], critic_input_ah_units=rl2_net_struct[3],
            critic_input_ah_layers_num=rl2_net_struct[4], critic_input_an_units=rl2_net_struct[5],
            critic_middle_layers_num=rl2_net_struct[6],
            critic_middle_layers_units=[rl2_net_struct[7]] * rl2_net_struct[6],
            actor_s1_input_units=rl2_net_struct[8], actor_s2_input_units=rl2_net_struct[9],
            actor_s2_input_layers_num=rl2_net_struct[10], actor_ah_input_units=rl2_net_struct[11],
            actor_ah_input_layers_num=rl2_net_struct[12], actor_middle_layers_num=rl2_net_struct[13],
            actor_middle_layers_units=[rl2_net_struct[14]] * rl2_net_struct[13],
            learning_rate_actor=1e-5, learning_rate_critic=1e-5,
            memory_type='Prioritized', memory_alpha=0.3, memory_beta0=0.1, memory_beta_increment=0.00001,
            memory_size=mem_size,  # 优先级采样
            noise_type='AN', an_noise_params=np.array([0.05, 0.0001, 1.01]),
            batch_size=batch_size, file_path='rl2', req_save_params=True
        )
    if params_path is not None:
        rl0.load_net_params(os.path.join(params_path, 'rl0'))
        rl1.load_net_params(os.path.join(params_path, 'rl1'))
        rl2.load_net_params(os.path.join(params_path, 'rl2'))
    return [rl0, rl1, rl2]


def save_model_params(rl_list: List[REDCRML]):
    for rl in rl_list:
        rl.save_net_params()


def train(env: DynaDropNav, rl: List[REDCRML], enable_async=False, enable_auto_aiming=False, mission_name=None,
          max_episodes=2000, learn_start_steps=1000, learn_eps=64, cer_update_eps=10, noise_eps=20):
    task_cnt = env.task_cnt
    assert env is not None, '环境参数未None，请传参！'
    assert rl is not None, '算法参数为None，请传参！'
    assert len(rl) == task_cnt, '算法参数为None，请传参！'
    env.enable_auto_aim(enable_auto_aiming)
    # 训练参数
    total_steps = 0  # 总决策步数
    total_train_cnt = 0  # 总训练次数
    last_train_train_cnt = 0  # 上一次更新时的总训练次数
    # 记录参数
    history_done = Queue(maxsize=50)  # 最近50次实验的结束状态
    success_rates = []  # 任务成功率
    episode_rewards = []  # 周期累积回报
    for i_episode in range(max_episodes):
        task_step_cnt = [0 for _ in range(task_cnt)]
        obv = env.reset()
        for i_rl in range(3):
            if type(rl[i_rl]) == REDCRML:
                rl[i_rl].reset_seq()
        task_label = env.get_task_label()
        if i_episode % noise_eps == 0:
            for i_rl in range(task_cnt):
                rl[i_rl].reset_noise()
                # pass
        ep_rwd = 0
        end_rwd = 0
        end_status = 0
        end_state = None
        fail_status = 0
        end_step = 0
        done = False
        j_step = 0
        eps_elapsed_s = time.perf_counter()
        while not done:
            act, net_obv = rl[task_label].choose_action(obv)
            obv_, rwd, done, infos = env.step(act)
            act = float(env.get_action())
            net_obv_ = rl[task_label].get_seq_ha(obv_)
            pre_task_label = task_label
            task_label = infos['TaskLabel']
            task_done = infos['TaskDone']
            # 设置保存结果
            if task_label == pre_task_label:
                # 根据 task_label 和 task_done 的值确定 flag 的值
                flag = int(task_label != 1 and task_done)
                # 根据 rl[task_label].is_curriculum() 的结果确定前四个参数
                obv1, obv2 = net_obv, net_obv_
                # 调用 store_transition 方法
                rl[task_label].store_transition(obv1, act, rwd, obv2, flag)
            if task_label == 0:
                task_step_cnt[0] += 1
            elif task_label == 1:
                task_step_cnt[1] += 1
            elif task_label == 2:
                task_step_cnt[2] += 1
            obv = obv_
            end_step = j_step
            ep_rwd += rwd
            end_rwd = rwd
            end_status = infos['EnvStatus']
            end_state = infos['EnvState']
            fail_status = infos['FailType']
            total_steps += 1
            j_step += 1

            if total_steps >= learn_start_steps and total_steps % learn_eps == 0:
                # 每隔 learn_eps 个决策周期，训练一次网络
                for i_rl in range(task_cnt):
                    rl[i_rl].train()
                total_train_cnt += 1
            if total_train_cnt > 0 and total_train_cnt % cer_update_eps == 0 and total_train_cnt != last_train_train_cnt and enable_async:
                last_train_train_cnt = total_train_cnt
                # 每隔 cer_update_eps 个训练周期，更新一次memory
                for i_rl in range(task_cnt):
                    rl[i_rl].curriculum_update_memory()

        if history_done.full():
            history_done.get()
        history_done.put(1 if end_status == 1 and task_label == 2 else 0)

        tmp_td = []
        for _ in range(history_done.qsize()):
            tmp_td.append(history_done.get())
            history_done.put(tmp_td[-1])
        success_cnt = np.sum(np.array(tmp_td, dtype=int) == 1)
        success_rates.append(success_cnt / history_done.qsize())
        env_state = SimModelState()
        ctypes.memmove(ctypes.byref(env_state), ctypes.byref(end_state), ctypes.sizeof(SimModelState))
        a_loss = [round(rl[0].get_actor_loss(), 3), round(rl[1].get_actor_loss(), 3)]
        c_loss = [round(rl[0].get_critic_loss(1), 3), round(rl[0].get_critic_loss(2), 3),
                  round(rl[1].get_critic_loss(1), 3), round(rl[1].get_critic_loss(2), 3)]
        if total_steps >= learn_start_steps:
            print(
                'mission: ', mission_name,
                ' episode: ', i_episode,
                ' end task label: ', task_label,
                ' end reward: ', round(end_rwd, 3),
                ' end status: ', end_status,
                ' end failure code: ', bin(fail_status),
                ' DSR: ', round(success_rates[-1], 3),
                ' end d_los: ', round(env.get_tgt_los_dist(), 3),
                ' end step: ', end_step,
                ' steps per task: ', task_step_cnt,
                ' actor loss: ', a_loss,
                ' critic loss: ', c_loss,
                'elapsed time: ', round(time.perf_counter() - eps_elapsed_s, 3),
            )
            episode_rewards.append(ep_rwd)
            for i_rl in range(task_cnt):
                rl[i_rl].write_simulation_params(episode_rewards[-1], success_rates[-1])
            for i_rl in range(task_cnt):
                rl[i_rl].update_noise()
        else:
            print(
                'mission: ', mission_name,
                ' episode: ', i_episode,
                ' end reward: ', round(end_rwd, 3),
                ' end status: ', end_status,
                ' end d_los: ', round(env.get_tgt_los_dist(), 3),
                ' end step: ', end_step,
                ' success rate: ', round(success_rates[len(success_rates) - 1], 3),
                'elapsed time: ', round(time.perf_counter() - eps_elapsed_s, 3),
            )
        if len(success_rates) > 500:
            avg_success_rate = np.mean(success_rates[-500:])
            subset_data = [x for x in success_rates[-500:] if x < avg_success_rate]
            std_success_rate = np.std(subset_data)
            if avg_success_rate >= 0.95 and std_success_rate <= (0.05 / 3):
                break
    print('%s end!' % mission_name)
    return success_rates, episode_rewards


def create_params_path(alg_name: str):
    params_path = os.path.join("CONTRAST", alg_name)
    test_params_path = os.path.join(params_path, "TestParams")
    check_dir_path(test_params_path)
    acmi_params_path = os.path.join(params_path, "ACMIParams")
    check_dir_path(acmi_params_path)
    figs_params_path = os.path.join("CONTRAST", "Figs")
    check_dir_path(figs_params_path)
    return test_params_path, acmi_params_path, figs_params_path
