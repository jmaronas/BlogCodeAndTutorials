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

This script implement a Maximum Posterior Neural Network on a toy problem: a 4 class problem over 2 dimensional input features. Features are simulated from gaussian distributions. Set variable visualize=1 to plot the distributions.


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
		self.w=nn.Parameter(torch.randn((indim,outdim),requires_grad=True))
		self.b=nn.Parameter(torch.randn((outdim),requires_grad=True))
		self.activation=nn.functional.relu if activation=='relu' else None

	def forward(self,x):
		return self.activation(torch.mm(x,self.w)+self.b) if self.activation != None else torch.mm(x,self.w)+self.b


class NN_ML(nn.Module):
	def __init__(self):
		#note that these parameters are the initial parameters from the Hamiltonian Monte Carlo Algorithm
		#note there is no need to declare them as parameters as HMC does not perform optimization, only sampling and marginalization
		#I associate to each parameter a momentum tensor, just to avoid reshapes or unfancy code :D
		super(NN_ML, self).__init__()
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

	def return_square_of_parameters(self):
		acc=0
		for l in self.Layers:
			acc+=(l.w**2+l.b**2)			
		return acc

	def train(self,x,t,epochs=200):
		#please note that we can simply pass the weight decay to the optimizer, however to make explicit what a gaussian prior over the parameters does I included in the cost function to be minimized
		optimizer=torch.optim.SGD(self.parameters(),lr=0.1,momentum=0.9)#SGD goes better for point estimate models
		for i in range(epochs):
			logit=self.forward(x)
			loss=self.ce(logit,t)+0.5*self.return_square_of_parameters()#note that the 0.5 comes from assuming variance=1 in our standard normal prior
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			print("On epoch {} Loss {:.5f}".format(i,loss.data))				
			

net=NN_ML()
net.train(x,t)

print("On MAP Network getting accuracy of {}".format(float((net.forward(x_test).argmax(1)==t_test).sum())*100./float(t_test.size(0))))

