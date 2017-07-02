import tensorflow as tf
import math
import numpy as np

nT=365
T=1.0
dt=T/nT
sqrtDt=math.sqrt(dt)
K=1
sigma=0.2


class BlackScholesNp:
	def __init__(self,n):
		self.n=n

	def price(self):
	    R=sigma*sqrtDt*np.random.randn(nT,self.n)
	    C=np.cumsum(R,axis=0)
	    S=np.exp(C)
	    A=np.mean(S,axis=0)
	    P=np.mean(np.maximum(A-K,0))
	    return P

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BlackScholesTf:

	def __init__(self, n, tf_device_name):
		with tf.device(tf_device_name):
			sigmat=tf.constant(sigma,name="sigmat")
			Rt=sigmat*sqrtDt*tf.random_normal((n,nT))
			Ct=tf.cumsum(Rt,axis=1)
			St=tf.exp(Ct)
			At=tf.reduce_mean(St,axis=1)
			self.Pt=tf.reduce_mean(tf.maximum(At-K,0))
		self.session=tf.Session()


	def price(self):
		res=self.session.run(self.Pt)