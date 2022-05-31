#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to 
a) select a problem to be solved 
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

def LISTA():
    #LISTA.py fig_8_1
    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=None,M=250,N=500,L=1000,pnz=.1,SNR=40) 
    #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    layers_tied = networks.build_LISTA(prob,T=3,initial_lambda=.1,untied=False)
    #layers_untied = networks.build_LISTA(prob,T=3,initial_lambda=.1,untied=True)
    # plan the learning
    training_stages_tied = train.setup_training(layers_tied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    #training_stages_untied = train.setup_training(layers_untied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    # do the learning (takes a while)
    sess = train.do_training(training_stages_tied,prob,'LISTA_bg_giid_tied.npz')
    #sess = train.do_training(training_stages_untied,prob,'LISTA_bg_giid_untied.npz')

def LAMP():
    #LAMP.py   fig_8_2
    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=None,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
    layers_tied = networks.build_LAMP(prob,T=3,shrink='bg',untied=False)
    layers_untied = networks.build_LAMP(prob,T=3,shrink='bg',untied=True)
    # plan the learning
    training_stages_tied = train.setup_training(layers_tied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    training_stages_untied = train.setup_training(layers_untied,prob,trinit=1e-3,refinements=(.5,.1,.01) )

    # do the learning (takes a while)
    sess = train.do_training(training_stages_tied,prob,'LAMP_bg_giid_tied.npz')
    sess = train.do_training(training_stages_untied,prob,'LAMP_bg_giid_untied.npz')


def LISTA_KAPPA():
    #LISTA.py   fig_9_1

    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=15,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    layers_tied = networks.build_LISTA(prob,T=3,initial_lambda=.1,untied=False)
    #layers_untied = networks.build_LISTA(prob,T=3,initial_lambda=.1,untied=True)
    # plan the learning
    training_stages_tied = train.setup_training(layers_tied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    #training_stages_untied = train.setup_training(layers_untied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    # do the learning (takes a while)
    sess = train.do_training(training_stages_tied,prob,'LISTA_bg_giid_tied_kappa.npz')
    #sess = train.do_training(training_stages_untied,prob,'LISTA_bg_giid_untied.npz')


def LAMP_KAPPA():
    #LAMP.py   fig_9_2
    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=15,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
    layers_tied = networks.build_LAMP(prob,T=3,shrink='bg',untied=False)
    layers_untied = networks.build_LAMP(prob,T=3,shrink='bg',untied=True)
    # plan the learning
    training_stages_tied = train.setup_training(layers_tied,prob,trinit=1e-3,refinements=(.5,.1,.01) )
    training_stages_untied = train.setup_training(layers_untied,prob,trinit=1e-3,refinements=(.5,.1,.01) )

    # do the learning (takes a while)
    sess = train.do_training(training_stages_tied,prob,'LAMP_bg_giid_tied_kappa.npz')
    sess = train.do_training(training_stages_untied,prob,'LAMP_bg_giid_untied_kappa.npz')

def LAMP_Shrinkage(shrink_name, untied_flag, kappa_num=None):
    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=kappa_num,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    # build a LAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
    layers = networks.build_LAMP(prob,T=3,shrink=shrink_name,untied=untied_flag)

    # plan the learning
    training_stages = train.setup_training(layers,prob,trinit=1e-3,refinements=(.5,.1,.01) )

    # do the learning (takes a while)
    kappas = 0
    if None != kappa_num:
        kappas = kappa_num
    istied = 'tied'
    if untied_flag:
        istied = 'untied'
    file_name = 'LAMP_bg_giid_%s_%s_%d.npz' % (shrink_name, istied,kappas)
    sess = train.do_training(training_stages,prob,file_name)

def LVAMP_Shrinkage(shrink_name, kappa_num=None):
    # Create the basic problem structure.
    prob = problems.bernoulli_gaussian_trial(kappa=kappa_num,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
    #prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

    # build an LVAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
    layers = networks.build_LVAMP(prob,T=3,shrink=shrink_name)
    #layers = networks.build_LVAMP_dense(prob,T=3,shrink='pwgrid')

    # plan the learning
    training_stages = train.setup_training(layers,prob,trinit=1e-4,refinements=(.5,.1,.01))

    # do the learning (takes a while)
    kappas = 0
    if None != kappa_num:
        kappas = kappa_num
    file_name = 'LVAMP_bg_giid_%s_%d.npz' % (shrink_name, kappas)
    sess = train.do_training(training_stages,prob, file_name)


LVAMP_Shrinkage('bg')
LVAMP_Shrinkage('bg', 15)
LVAMP_Shrinkage('bg', 100)
