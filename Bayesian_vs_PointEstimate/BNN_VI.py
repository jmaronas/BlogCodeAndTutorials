import torch
import torch.nn as nn
torch.manual_seed(1)

import numpy
numpy.random.seed(1)
import matplotlib.pyplot as plt

'''

Author: Juan Maroñas, PRHLT Research Center, Universitat Politècnica de València.
Date: April 2019

This script implement a Bayesian Neural Network using a Variational Inference Network with Factorized Gaussian Variational distribution and Standard Normal Prior on a toy problem: a 4 class problem over 2 dimensional input features. Features are simulated from gaussian distributions. Set variable visualize=1 to plot the distributions.

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
		#p(w,b)=Normal(0,1)
		self.w_mean=nn.Parameter(torch.randn((indim,outdim),requires_grad=True))
		self.w_logvar=nn.Parameter(torch.randn((indim,outdim),requires_grad=True))
		self.b_mean=nn.Parameter(torch.randn((outdim),requires_grad=True))
		self.b_logvar=nn.Parameter(torch.randn((outdim),requires_grad=True))
		self.sampler_w=torch.zeros((indim,outdim))
		self.sampler_b=torch.zeros((outdim,))
		self.activation=nn.functional.relu if activation=='relu' else None

	def sample(self):
		w_s=(self.sampler_w.normal_().data*torch.exp(0.5*self.w_logvar) + self.w_mean)
		b_s=(self.sampler_b.normal_().data*torch.exp(0.5*self.b_logvar) + self.b_mean)
		return w_s,b_s

	def forward(self,x):
		w,b=self.sample()
		return self.activation(torch.mm(x,w)+b) if self.activation != None else torch.mm(x,w)+b


class BNN_VI(nn.Module):
	def __init__(self):

		super(BNN_VI, self).__init__()
		L1=FC(2,50,'relu')
		L2=FC(50,50,'relu')
		L3=FC(50,4,'linear')
		self.Layers=nn.ModuleList([L1,L2,L3])
		self.ce=nn.functional.cross_entropy
		self.softmax=nn.functional.softmax
	def forward(self,x):
		for l in self.Layers:
			x=l(x)
		return x


	def GAUSSKL(self,mean,logvar):
		#computes the DKL(q(x)//p(x)) between the variational and the prior distribution assuming Gaussians distribution with prior=N(0,1)
		var = torch.exp(logvar)
		DKL = -0.5*(torch.tensor(mean.numel()).float() + (logvar-var-torch.pow(mean,2)).sum())
		return DKL
		

	def DKL(self):
		DKL=0.0
		for l in self.Layers:
			w_mean,w_logvar,b_mean,b_logvar=l.w_mean,l.w_logvar,l.b_mean,l.b_logvar
			DKL+=(self.GAUSSKL(w_mean,w_logvar)+self.GAUSSKL(b_mean,b_logvar))
		return DKL

	def ELBO(self,x,t,MC_samples=50,beta=0.1):
		NLLH=0.0
		for mc in range(MC_samples): #stochastic likelihood estimator			
			NLLH+=self.ce(self.forward(x),t)

		NLLH /= float(MC_samples)
		DKL=self.DKL() 
		return NLLH + DKL,NLLH,DKL

	def train(self,x,t,epochs=100,lr=0.1,warm_up=10):
		optimizer=torch.optim.Adam(self.parameters(),lr=lr)#adam goes better for this kind of models. In fact adam is not a correct optimizer, see https://openreview.net/pdf?id=ryQu7f-RZ However It always works fine for models based on reparametrization trick.
		for i in range(epochs):
			loss,NLLH,KL=self.ELBO(x,t)
			loss = loss if i > warm_up else NLLH
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			print("On epoch {} ELBO {:.5f} NNL {:.5f} KL {:.5f}".format(i,loss.data,NLLH.data,KL.data))				

	def predictive(self,x,samples):
		#we get samples using ancestral sampling
		with torch.no_grad():
			prediction=0.0
			for s in range(samples):
				prediction+=self.softmax(self.forward(x),dim=1)

			return prediction/float(samples)

			
			

BNN=BNN_VI()
BNN.train(x,t,50,0.1,10)
BNN.train(x,t,50,0.01,0)

prediction=BNN.predictive(x_test,1000)
print("Getting accuracy of {}".format(100.*float((prediction.argmax(1)==t_test).sum())/float(t_test.size(0))))

