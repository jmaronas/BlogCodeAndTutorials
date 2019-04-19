# Bayesian vs Point Estimates implementations.

This repository implement different approaches to probabilistic modelling using a simple toy dataset. I do this so we can focus on the implementations and reduce the code used to other things. I am creating some analysis on the different models I will upload as soon as possible.

If you want to go over the different implementations please follow this order. I attach a pdf to each topic with the explanation and a .py file for running. I have used pytorch 1.0.0 and python3.7 but I think It will work on any version starting from pytorch 0.4.0 and for any python. You should follow this order just because I think it is easy to follow as some things are only commented on some pdfs.


## POINT ESTIMATE:

Point estimate neural networks.  Maximum Likelihood and Maximum Posterior

pdf with explanations: POINTESTIMATE\_NN.pdf

code: 
*  python ML\_NN.py  : maximum likelihood Neural Networks
*  python MAP\_NN.py : maximum posterior Neural Networks

## Bayesian Models:

### Markov Chain Hamiltonian Monte Carlo

[fixing some stuff]

pdf: HMC\_BNN.pdf

code: 

* python BNN\_HMC.py

### Variational Inference

pdf: VI_BNN.pdf

code: 

* python BNN\_VI.py : standard amortized variational inference with Gaussian distributions
* python BNN\_VILR.py : same but with local reparameretization

