import tensorflow as tf
import math
import numpy as np

nT=365
T=1.0
dt=T/nT
sqrtDt=math.sqrt(dt)
K=1
sigma=0.2
rho=-0.5
rhobar=math.sqrt(1-rho*rho)
V0=0.04
Vbar=0.04
kappa=1
alpha=0.5


class HestonNp:
	def __init__(self,n):
		self.n=n

	def price(self):
	    W1=np.random.randn(nT,self.n)
	    W2=rhobar*np.random.randn(nT,self.n)+rho*W1

	    V=np.full((nT,self.n),V0)
	    for i in range(1, nT):
	    	V[i,:]=np.fabs( V[i-1,:] + kappa * (Vbar - V[i-1,:])*dt + alpha * np.sqrt( V[i-1,:]) * sqrtDt * W2[i-1,:])
	 
	    R=np.sqrt(V)*sqrtDt*W1
	    C=np.cumsum(R,axis=0)
	    S=np.exp(C)
	    A=np.mean(S,axis=0)
	    res=np.mean(np.maximum(A-K,0))


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class HestonTf:

	def updateV(prevV, currW):
		return tf.abs( prevV + kappa * (Vbar - prevV)*dt + alpha * tf.sqrt( prevV ) * sqrtDt * currW )

	def __init__(self, n, computeGrad, tf_device_name):
		self.computeGrad=computeGrad

		with tf.device(tf_device_name):
			W1t=tf.random_normal((nT,n))
			W2t=rhobar*tf.random_normal((nT,n))+rho*W1t

			V0t=tf.constant(V0,name="V0t")
			V0tt=tf.fill((1,n),V0t)
			W2t=tf.concat( [ V0tt, W2t[:nT-1,:]], 0 )
			Vt=tf.scan(HestonTf.updateV, W2t)

			Rt=tf.multiply(tf.sqrt(Vt),sqrtDt*W1t)
			Ct=tf.cumsum(Rt,axis=0)
			St=tf.exp(Ct)
			At=tf.reduce_mean(St,axis=0)
			self.Pt=tf.reduce_mean(tf.maximum(At-K,0))
			self.grad=tf.gradients(self.Pt, [V0t])[0]

		self.session=tf.Session()

	def price(self):
		if self.computeGrad:
			res=self.session.run([self.Pt, self.grad])
		else:
			res=self.session.run(self.Pt)