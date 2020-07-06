import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

n = 1000000
dt = 0.001
t = np.arange(0,n+1)*dt

class Adaptive_Hopf_Oscillator():
    def __init__(self,
                 gamma = 2,
                 phi = np.random.random(),
                 n = 1000000,
                 dt = 0.001,
                 mu = 1,
                 eps = 0.9,
                 omega_0 = 40,
                 F = np.sin(2*np.pi*30*t)[:-1]):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.eps = eps
        self.gamma = gamma
        self.omega = omega_0
        self.x = 1
        self.y = 0

        self.Aomega = np.zeros(self.n)
        self.Ax = np.zeros(self.n)
        self.Ay = np.zeros(self.n)
        self.Aphi = np.zeros(self.n)
        self.t = np.arange(0,n)*self.dt
        self.F = F

    def dx(self, x, y, i, omega):
        dx = self.dt*((self.mu - (x**2 + y**2))*x*self.gamma - omega*y + self.eps*self.F[i])
        return dx

    def dy(self, y, x, i, omega):
        dy = self.dt*((self.mu - (x**2 + y**2))*y*self.gamma + omega*x)
        return dy

    def domega(self, x, y, i):
        domega = self.dt*((-1)*self.eps*self.F[i]*y/np.sqrt(x**2 + y**2))
        return domega

    def phase(self, x, y):
        phase = np.arctan(y/x)
        return phase

    def train(self):
        for i in tqdm(range(0, self.n)):
            self.Aphi[i] = self.phase(self.x, self.y)
            self.Ax[i] = self.x
            self.Ay[i] = self.y
            self.Aomega[i] = self.omega
            self.x = self.x + self.dx(self.x, self.y, i, self.omega)
            self.y = self.y + self.dy(self.y, self.x, i, self.omega)
            self.omega = self.omega + self.domega(self.x, self.y, i)
        
    def plot(self):
        fig, axes = plt.subplots(1,2)
        axes[0].plot(self.t, self.Aomega)
        axes[1].plot(self.t, self.F)
        plt.show()

    def main(self):
        self.train()
        self.plot()
#----------Initialization for trajectory in figure 1-----------#
F = np.sin(2*np.pi*30*t)
num = int(1/(15*dt))
print(num)
k = 0
for i in range(n):
    if i%num>=num/4 or F[i]<0.0:
        F[i] = F[i]*0.001
fig, axes = plt.subplots(1,1)
axes.plot(t[:100], F[:100])
axes.set_title('teaching signal')
fig.savefig('../plots/teaching_signal_exp_2_hopf.png')
hopf_osc = Adaptive_Hopf_Oscillator(F=F)
#--------------------------------------------------------------#
print(hopf_osc.Aomega)
print(hopf_osc.Aphi)
hopf_osc.train()
print(hopf_osc.Aomega)
print(hopf_osc.Aphi)
fig, axes = plt.subplots(1, 4)
axes[0].plot(hopf_osc.t[-1000:], hopf_osc.Ax[-1000:])
axes[1].plot(hopf_osc.t[-1000:], hopf_osc.Ay[-1000:])
axes[2].plot(hopf_osc.t[-1000:], hopf_osc.Aomega[-1000:])
axes[3].plot(hopf_osc.t[-1000:], hopf_osc.Aphi[-1000:])
plt.show()
fig.savefig('../plots/learned_hopf_exp_2.png')
