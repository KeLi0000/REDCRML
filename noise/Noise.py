# encoding: utf-8
"""
@author     :   KeLi
@contact    :   569280177@qq.com
@time       :   2020/12/14 16:49
@file       :   Noise.py
@desc       :   None
@License   :   (C)Copyright 2020-2021, Ke LI
"""
import numpy as np


class AdaptiveParamNoiseSpec(object):
    """
    自适应噪声发生器
    """

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    """
    正态动作噪声发生器
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    OU 噪声发生器
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class AdaptiveNormalActionNoise(AdaptiveParamNoiseSpec):
    """
    自适应高斯噪声发生器
    """

    def __init__(self, mu=0.0, sigma_0=1.0, sigma_t=0.1, sigma_d=1.01):
        """
        自适应高斯噪声发生器
        :param mu: 均值
        :param sigma_0: 初始方差
        :param sigma_t: 目标方差
        :param sigma_d: 方差变化率
        """
        super(AdaptiveNormalActionNoise, self).__init__(
            initial_stddev=sigma_0, desired_action_stddev=sigma_t, adoption_coefficient=sigma_d)
        self.mu = mu

    def reset(self):
        self.current_stddev = self.initial_stddev

    def __call__(self):
        x = np.random.normal(loc=self.mu, scale=self.current_stddev)
        return x

    def update(self):
        self.adapt(self.current_stddev)
