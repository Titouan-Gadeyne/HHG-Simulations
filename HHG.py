import numpy as np


class ActiveGrating():
    def __init__(self, qeff):
        self.qeff = qeff

    def Hq_NearField(self, IRfield, q):
        return np.abs(IRfield)**self.qeff * np.exp(1j*q*np.angle(IRfield))
