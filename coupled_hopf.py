import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 1000000
dt = 0.001
t = np.arange(0, n+1)*dt

class Hopf(object):
    def __init__(self,
                 v,
                 offset,
                 omega_sw # controls the speed of the robot, also swing frequency
                 mu,
                 beta, # duty factor
                 l,
                 a):
        self.mu = mu
        self.omega_sw = omega_sw
        self.x = 0.0
        self.y = 0.0
        self.omega = 0.0
        self.l = l
        self.beta = beta
        self.offset = offset

    def r(self):
        return np.sqrt((self.x-self.offset)**2+self.z**2)
        
    def dx(self):
        return self.alpha*(self.mu*self.v - (self.r())**2)*(self.x-self.offset)-self.omega()*self.l*self.y

    def dy(self):
        return self.alpha*(self.mu*self.v - (self.r())**2)*self.y-self.omega()*self.l*(self.x-self.offset)

    def omega(self):
        return ((1/self.beta - 2)*self.omega_sw)/(np.exp(-self.a*self.y)+1)+self.omega_sw/(np.exp(self.a*self.z)+1)

    def set_x(self, x):
        self.x = x

    def set_omega(self, omega):
        self.omega = omega

    def set_y(self, y):
        self.y = y

    def set_v(self, v):
        self.v = v

    def set_mu(self, mu):
        self.mu = mu

    def set_omega_sw(self, omega_sw):
        self.omega_sw = omega_sw

    def set_l(self, l):
        self.l = l

    def set_offset(self):
        self.offset = offset

def rotation_matrix(self, theta, l):
    """
        theta is the relative phase between the two oscillator for which the rotation matrix has to be calculated
    """
    R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return l*np.array(R)

class Knee(object):
    def __init__(self, 
                 b,
                 a,
                 theta_sw,
                 theta_st):
        self.b = b
        self.a = a
        self.v = 0.0
        self.z = 0.0 
        self.theta_st = theta_st
        self.theta_sw = theta_sw
    
    def dz(self):
        return self.v

    def dv(self, y):
        return -(self.b**2)*(self.z-self.g(y))/4 - self.b*self.v

    def g(self, y):
        return self.theta_st/(np.exp(-self.a*y)+1) + self.theta_sw/(np.exp(self.a*y)+1)

    def set_z(self, z): 
        self.z = z
    
    def set_v(self, v):
        self.v = v
    
    def set_theta_st(self, theta):
        self.theta_st = theta

    def set_theta_sw(self, theta):
        self.theta_sw = theta

    def set_a(self, a):
        self.a = a

    def set_b(self, b):
        self.b = b
           
class CPG(object):
    def __init__(self, 
                 K_mat,
                 v,  
                 offset,
                 omega_sw # controls the speed of the robot, also swing frequency
                 mu, 
                 beta, # duty factor
                 l,  
                 K_mat,
                 a,
                 b, 
                 theta_st,
                 theta_sw,
                 num = 4,
                 dt = 0.001,
                 n = 1000000):
        self.num = num
        self.dt = dt
        self.n = n
        self.K_mat = K_mat
        self.osc = []
        self.knee = []
        for i in range(self.num):
            self.osc.append(Hopf(v = v[i], 
                                 offset = offset[i],
                                 omega_sw = omega_sw[i],
                                 mu = mu[i],
                                 beta = beta[i],
                                 l = l[i],
                                 a = a[i]))
            self.knee.append(Knee(b = b[i],
                                  a = a[i],
                                  theta_sw = theta_sw[i],
                                  theta_st = theta_st[i]))
        
        self.t = np.arange((self.n))*self.dt
        self.Ax = np.zeros((self.n, self.num))
        self.Ay = np.zeros((self.n, self.num))
        self.Az = np.zeros((self.n, self.num)) 
        self.Av = np.zeros((self.n, self.num))
        self.Aomega = np.zeros((self.n, self.num))     
        self.Ag = np.zeros((self.n, self.num))
        self.x = np.zeros((self.num))
        self.y = np.zeros((self.num))
        self.z = np.zeros((self.num))
        self.v = np.zeros((self.num))
        self.omega = np.zeros((self.num))
        self.g = np.zeros((self.num))

    def simulate(self):
        for i in range(self.n):
            self.Ax[i, :] = self.x
            self.Ay[i, :] = self.y
            self.Az[i, :] = self.z
            self.Av[i, :] = self.v
            self.Aomega[i, :] = self.omega
            self.Ag[i, :] = self.g
            for i in range(self.num):
                self.x[i] += (self.osc[i].dx() + )*self.dt
                self.y[i] += self.osc[i].dy()*self.dt
                self.omega[i] = self.osc[i].omega()
                self.osc[i].set_x(self.x)
                self.osc[i].set_y(self.y)
                self.osc[i].set_omega(self.omega)
                self.z = self.knee[i]  
