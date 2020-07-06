import numpy as np 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class com_CPG():
    def __init__(self,
                 n = 1000000,
                 dt = 0.001,
                 mu = 1, 
                 eps = 0.5,
                 etaalp = 0.5,
                 num = 4,):
        self.n = n
        self.dt = dt
        self.mu = mu
        self.eps = eps
        self.etaalp = etaalp
        self.num = num
        self.omega = np.random.random(self.num)*20  # shape (num,)
        self.z = np.exp(1j*np.random.random(self.num)*2*np.pi) # shape (num,)
        self.alpha = np.random.random(self.num) #shape (num,) 
        self.F = np.zeros(self.n)
        self.Aomega = np.zeros((self.num, self.n)) # shape (num, n)
        self.Az = np.zeros((self.num, self.n), dtype = np.complex128)
        self.Aalpha = np.zeros((self.num, self.n)) # shape (num, n)
        self.Ar = np.zeros((self.num, self.n)) # shape (num, n)
        self.Aph = np.zeros((self.num, self.n)) # shape (num, n)
        self.t = np.arange(0,n)*self.dt # shape (n,)
        self.st = np.zeros(self.n) # shape (n,)
        self.et = np.zeros(self.n) # shape (n,)
        omega_input = [5, 10, 15, 20]
        I_max = [1.1, 0.8, 1.5, 0.7]
        phase = [np.pi/3, np.pi/4, np.pi/2, np.pi/5]
        for i in range(4):
            self.F = self.F + I_max[i]*np.sin(omega_input[i]*self.t + phase[i])
        
    def train(self):
        for i in tqdm(range(self.n)):
            self.Ar[:, i] = np.abs(self.z)
            self.Aph[:, i] = self.phase(self.z)
            self.Az[:, i] = self.z
            self.Aomega[:, i] = self.omega
            self.Aalpha[:, i] = self.alpha
            self.st[i] = np.sum(self.alpha*self.z.real)
            self.et[i] = self.F[i] - self.st[i]
            #print(self.dz(self.z, i, self.omega))
            self.z = self.z + self.dz(self.z, i, self.omega)
            #print(self.domega(self.z, i))
            self.omega = self.omega + self.domega(self.z, i)
            #print(self.dalpha(self.z, i))
            self.alpha = self.alpha + self.dalpha(self.z, i)

    def dz(self, z, i, omega):
        return self.dt*((self.mu - np.abs(z)**2)*z + 1j*omega*z + self.eps*self.et[i])

    def domega(self, z, i):
        return self.dt*((-1)*self.eps*self.et[i]*z.imag/np.abs(z))

    def dalpha(self, z, i):
        return self.dt*(self.etaalp*z.real*self.et[i])

    def phase(self, z):
        imag = z.imag
        real = z.real
        phase = np.arctan(imag/real)
        return phase

CPG = com_CPG()
CPG.train()
print(CPG.Aomega)
fig, axes = plt.subplots(2,2)
axes[0, 0].plot(CPG.t, CPG.Aomega[0])
axes[0, 0].set_title('neuron 1')
axes[0, 1].plot(CPG.t, CPG.Aomega[1])
axes[0, 1].set_title('neuron 2')
axes[1, 0].plot(CPG.t, CPG.Aomega[2])
axes[1, 0].set_title('neuron 3')
axes[1, 1].plot(CPG.t, CPG.Aomega[3])
axes[1, 1].set_title('neuron 4')
fig.savefig('com_cpg_plot_omega.png')
fig1, axes1 = plt.subplots(2,2)
axes1[0, 0].plot(CPG.t[-1000:], CPG.Az[0, -1000:].real)
axes1[0, 0].set_title('neuron 1')
axes1[0, 1].plot(CPG.t[-1000:], CPG.Az[1, -1000:].real)
axes1[0, 1].set_title('neuron 2')
axes1[1, 0].plot(CPG.t[-1000:], CPG.Az[2, -1000:].real)
axes1[1, 0].set_title('neuron 3')
axes1[1, 1].plot(CPG.t[-1000:], CPG.Az[3,-1000: ].real)
axes1[1, 1].set_title('neuron 4')
fig1.savefig('com_cpg_plot_z.png')
#plt.show()
