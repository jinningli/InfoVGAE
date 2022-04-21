#!/usr/bin/env python3
# coding: utf-8
from math import exp

class PIDControl():
    """docstring for ClassName"""

    def __init__(self, Kp=0.001, Ki=-0.001):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0 / (1.0 + float(scale) * exp(Err))

    def pid(self, exp_KL, KL_loss):
        """
     position PID algorithm
     Input: KL_loss
     return: weight for KL loss, beta
     """
        error_k = exp_KL - KL_loss
        ## comput U as the control factor
        Pk = self.Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 + self.Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        ## window up for integrator
        if self.W_k1 < 0 and self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        ## min and max value
        if Wk >= 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk
