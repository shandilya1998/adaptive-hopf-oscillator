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
                 phi = np.random.random(), 
                 n = 1000000,
                 dt = 0.001,
                 mu = 1,
                 eps = 0.9,
                 omega_0 = 40,
                 F = 2*np.exp(1j*30*t)[:-1]):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.eps = eps
        self.omega = omega_0
        self.z = np.exp(2j*np.random.random()*np.pi) # This is the initialization of the ocillator
        self.Aomega = np.zeros(self.n) # This is the initialization of the frequency of the oscillator
        self.Az = np.zeros(self.n, dtype = np.complex128)
        self.Aph = np.zeros( self.n)
        self.Ar = np.zeros(self.n)
        self.t = np.arange(0,self.n)*self.dt
        self.F = 2*np.exp(1j*30*self.t)

    def dz(self, z, i, omega):
        return self.dt*((self.mu - np.abs(z)**2)*z + 1j*omega*z + self.eps*self.F[i].imag)

    def domega(self, z, i):
        return self.dt*((-1)*self.eps*self.F[i].imag*z.imag/np.abs(z))

    def phase(self, z):
        real = z.real
        imag= z.imag
        phase = np.arctan(imag/real)
        return phase

    def train(self):
        for i in tqdm(range(0, self.n)):
            self.Aph[i] = self.phase(self.z)
            self.Az[i] = self.z
            self.Aomega[i] = self.omega
            self.Ar[i] = np.abs(self.z)
            self.z = self.z + self.dz(self.z, i, self.omega)
            self.omega = self.omega + self.domega(self.z, i)
        
    def plot(self):
        fig, axes = plt.subplots(1,3)
        axes[0].plot(self.t, self.Aomega)
        axes[0].set_title('Frequency Plot against time')
        axes[1].plot(self.t, self.Aph)
        axes[1].set_title('Phase plot against time')
        axes[2].plot(self.t, self.Ar)
        axes[2].set_title('Magnitude Plot against time')
        plt.show()

    def main(self):
        self.train()
        self.plot()

F = 2*np.exp(1j*30*t)
num = int(1/(15*dt))
print(num)
k = 0 
for i in range(n):
    if i%num>=num/4 or F[i]<0.0:
        F[i] = F[i]*0.001 
fig, axes = plt.subplots(1,1)
axes.plot(t[:100], F[:100])
axes.set_title('teaching signal')
fig.savefig('../plots/teaching_signal_exp_1_hopf.png')
hopf_osc = Adaptive_Hopf_Oscillator(F=F)
print(hopf_osc.Aomega)
print(hopf_osc.Ar)
hopf_osc.main()
print(hopf_osc.Aomega)
print(hopf_osc.Ar)
fig, axes = plt.subplots(1,3)
axes[0].plot(hopf_osc.t, hopf_osc.F.real)
axes[1].plot(hopf_osc.t, hopf_osc.F.imag)
axes[2].plot(hopf_osc.Aomega)
plt.show()
plt.savefig('com_adaptive_osc_exp_1.png')
