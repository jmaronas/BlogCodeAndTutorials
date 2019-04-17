import torch
import torch.nn as nn
torch.manual_seed(1)

import numpy
numpy.random.seed(1)
import math
import matplotlib.pyplot as plt

'''
Author: Juan Maroñas, PRHLT Research Center, Universitat Politècnica de València.
Date: April 2019

This script implement a Bayesian Neural Network using Hamiltonian Monte Carlo to draw samples on a toy problem: a 4 class problem over 2 dimensional input features. Features are simulated from gaussian distributions. Set variable visualize=1 to plot the distributions.
'''

###########################
#####DRAW SOME SAMPLES#####
visualize=0
#sample 100 samples per class
x1=torch.randn(150,2)*1.0+0.5
x2=torch.randn(150,2)*0.5+torch.from_numpy(numpy.array([0.5,-2])).float()
x3=torch.randn(150,2)*0.3-0.5
x4=torch.randn(150,2)*0.8+torch.from_numpy(numpy.array([-1.0,-2])).float()

t1=torch.zeros(100,)
t2=torch.ones(100,)
t3=torch.ones(100,)+1
t4=torch.ones(100,)+2


idx=numpy.random.permutation(400)
x=torch.cat((x1,x2,x3,x4))[idx].float()
t=torch.cat((t1,t2,t3,t4))[idx].long()

if visualize:
	plt.plot(x1[:,0].numpy(),x1[:,1].numpy(),'*r')
	plt.plot(x2[:,0].numpy(),x2[:,1].numpy(),'*g')
	plt.plot(x3[:,0].numpy(),x3[:,1].numpy(),'*b')
	plt.plot(x4[:,0].numpy(),x4[:,1].numpy(),'*k')


''' test set '''
x1=torch.randn(20,2)*1+0.5
x2=torch.randn(20,2)*0.5+torch.from_numpy(numpy.array([0.5,-2])).float()
x3=torch.randn(20,2)*0.3-0.5
x4=torch.randn(20,2)*0.8+torch.from_numpy(numpy.array([-1.0,-2])).float()

t1=torch.zeros(20,)
t2=torch.ones(20,)
t3=torch.ones(20,)+1
t4=torch.ones(20,)+2

idx=numpy.random.permutation(80)
x_test=torch.cat((x1,x2,x3,x4))[idx].float()
t_test=torch.cat((t1,t2,t3,t4))[idx].long()


if visualize:
	plt.plot(x1[:,0].numpy(),x1[:,1].numpy(),'o',color='orange')
	plt.plot(x2[:,0].numpy(),x2[:,1].numpy(),'o',color='lightgreen')
	plt.plot(x3[:,0].numpy(),x3[:,1].numpy(),'o',color='cyan')
	plt.plot(x4[:,0].numpy(),x4[:,1].numpy(),'o',color='gray')
	plt.show()


###########################
##########MODEL ###########
#we project to two hidden layers of 50 neurons and finally and output layer of 4
#use relu activation

class FC(nn.Module):
	def __init__(self,indim,outdim,activation):
		super(FC, self).__init__()
		self.w=torch.randn((indim,outdim),requires_grad=True)
		self.b=torch.randn((outdim),requires_grad=True)
		self.mmu_w=torch.randn(indim,outdim)
		self.mmu_b=torch.randn((outdim))
		self.activation=nn.functional.relu if activation=='relu' else None

	def __resample__(self):
		self.mmu_w.normal_()
		self.mmu_b.normal_()

	def forward(self,x):
		return self.activation(torch.mm(x,self.w)+self.b) if self.activation != None else torch.mm(x,self.w)+self.b

	def zero_grad(self):
		self.w.grad.zero_()
		self.b.grad.zero_()

	def initialize(self):
		self.w.normal_()
		self.b.normal_()

	def set_params(self,w,b):
		self.w=w
		self.b=b

class BNN_HMC(nn.Module):
	def __init__(self):
		#note that these parameters are the initial parameters from the Hamiltonian Monte Carlo Algorithm
		#note there is no need to declare them as parameters as HMC does not perform optimization, only sampling and marginalization
		#I associate to each parameter a momentum tensor, just to avoid reshapes or unfancy code :D
		super(BNN_HMC, self).__init__()
		L1=FC(2,50,'relu')
		L2=FC(50,50,'relu')
		L3=FC(50,4,'linear')
		self.Layers=[L1,L2,L3]

		self.ce=nn.functional.cross_entropy
		self.softmax=nn.functional.softmax
		self.leapfrogsteps=20
		self.epsilon=0.1
		self.pi=torch.tensor(math.pi)

	def __set_parameters__(self,parameters):
		for layer,(w,b) in zip(self.Layers,parameters):
			layer.w=w
			layer.b=b

	def __resample__(self):
		for layer in self.Layers:
			layer.__resample__()

	def __parameters__(self):
		a=[]
		for l in self.Layers:
			a+=[(l.w,l.b)]
		return a
		
	def forward(self,x):
		for layer in self.Layers:
			x=layer(x)
		return x

	def __lognormal__(self):
		LPRIOR=0
		for l in self.Layers:
			cte = -0.5*torch.log(2*self.pi)
			LPRIOR+=((-0.5*l.w**2 + cte).sum() + (-0.5*l.b**2 + cte).sum())
	
		return LPRIOR

	def __kinetic__(self):
		#computes the kinetic function for all the momentum associated with each parameter
		K=0
		for l in self.Layers:
			K+=0.5*((l.mmu_w**2).sum()+(l.mmu_b**2).sum())
		return K

	def __potential__(self,x,t):
		logit=self.forward(x)
		NLOGLH=self.ce(logit,t)	
		NLOGPRIOR=-1*self.__lognormal__()
		return NLOGLH+NLOGPRIOR

	def H(self,x,t):
		#compute H=U(w)+K(m)
		#U(w) = -log [\prod p(t_i|x_i,w) \cdot p(w)]
		#K(m) = 0.5 m^T \cdot m
		K=self.__kinetic__()	
		U=self.__potential__(x,t)
		return K+U

	def sample_from_posterior(self,x,t):

		#resample the momentum variables
		self.__resample__()
		
		#store the Hamiltonian on the previous parameters for the metropolis hasting correction
		H_prev=self.H(x,t)
		init_params=self.__parameters__()

		#simulate hamiltonian dynamics
		for step in range(self.leapfrogsteps):
			logit=self.forward(x)#in order to compute the derivative of p(w|\mathcal{O})
			LLH = -1*self.ce(logit,t) #log likelihood. Part of the gradient of the posterior is computed by backpropagating this error
			LLH.backward()# so we have the gradient of the LLog likelihood w.r.t each parameter
		
			#leapfrog step
			#1) compute m' and w1
			for l in self.Layers:

				gradient_w = l.w.grad.data-l.w.data #derivative or -CE + derivative of log normal (-w)
				m_prime_w=l.mmu_w-self.epsilon/2.*gradient_w

				
				gradient_b = l.b.grad.data-l.b.data #derivative or -CE + derivative of log normal (-w)
				m_prime_b=l.mmu_b-self.epsilon/2.*gradient_b

				l.zero_grad()#this not necessary as the next two lines of code reset grads, however i put it for clarity

				#call data here to use the same l.w variable. If we do l.w= then a new variable is created with requires_grad set to false
				l.w.data=l.w.data -self.epsilon*m_prime_w
				l.b.data=l.b.data -self.epsilon*m_prime_b		
						
				l.mmu_w=m_prime_w
				l.mmu_b=m_prime_b

			#2)compute m1. We need gradients w.r.t the new w1
			#compute forward with parameter w1
			logit=self.forward(x)
			LLH = -1*self.ce(logit,t) 
			LLH.backward()

			for l in self.Layers:

				gradient_w = l.w.grad.data-l.w.data
				l.mmu_w = l.mmu_w - self.epsilon/2. * gradient_w
				
				gradient_b = l.b.grad.data-l.b.data
				l.mmu_b  = l.mmu_b - self.epsilon/2. * gradient_b

		#metropolist hasting correction
		u = numpy.random.uniform()
		alfa = torch.min(torch.tensor(1.0),torch.exp(self.H(x,t)-H_prev))
		if u>alfa:
			self.__set_parameters__(init_params)


	def run_MCMC(self,x,t,iterations,store_all=True,warm_up=0):
		param_group=[]
		for itet in range(iterations):
			print("Running MonteCarlo itet {}".format(itet))
			self.sample_from_posterior(x,t)
			if (store_all and itet>=warm_up):
				param_group.append(self.__parameters__())
			
		return param_group if store_all else self.__parameters__()

	def predictive(self,parameters,x,t):
		prediction=0.0
		for idx,params in enumerate(parameters):
			h=x
			for layer,(w,b) in zip(self.Layers,params):
				layer.set_params(w,b)
				h=layer(h)

			prediction+=self.softmax(h,dim=1)
		return prediction/float(len(parameters))

warm_up=200
mcmc_chain=10000
HMC_sampler=BNN_HMC()
parameters=HMC_sampler.run_MCMC(x,t,10000,True,warm_up=200)
prediction=HMC_sampler.predictive(parameters,x_test,t_test)

print("Drawing {} samples with {} warm_up from a HMC samples getting accuracy of {}".format(mcmc_chain,warm_up,50,(float((prediction.argmax(1)==t_test).sum())*100.)/float(t_test.size(0))))

