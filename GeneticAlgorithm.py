# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:23:02 2023

@author: Rogelio Garcia
"""
import torch
import numpy as np


def desnormalization(V,LMAX,LMIN):
    a=V*LMAX+LMIN*(1-V)
    return a

def normalization(V):
    LMAX = V.max()
    LMIN = V.min()
    a= (V-LMIN)/(LMAX-LMIN)
    return a,LMAX,LMIN

def selecTornN(Pop, Eval, N=2): ## Selection algorithm
    TI, TD = Pop.shape
    Eval_PS=np.zeros(TI)
    if N > TI:
        N = TI
    if N < 1:
        N = 1
    PS = np.zeros((TI, TD))
    for i in range(TI):
        i=i
        win = i
        for n in range(N):
            comp = int(round(desnormalization(np.random.rand(),-0.5,TI-1)))
            if Eval[win] < Eval[comp]:
                win = comp
        PS[i,:] = Pop[win,:]
        Eval_PS[i] = Eval[win]
    return PS, Eval_PS

def GenAGGR(PS, pc, pm, Eval): ## Generation algorithm 
    TI, TD = PS.shape
    P = np.zeros((TI, TD))
    for i in range(TI-1):
        pos = int(round(desnormalization(np.random.rand(),-0.5,TI-1)))
        for g in range(TD):
            if pc[g] > np.random.rand():
                P[i,g] = ((PS[i,g]*Eval[i])+(PS[pos,g]*Eval[pos]))/(Eval[i]+Eval[pos]+0.00000000001)
                # P[i,g] = ((PS[i,g])+(PS[pos,g])/(2))
            else:
                P[i,g] = PS[i,g]
            if pm[g] > np.random.rand():
                P[i,g] = np.random.rand()
    P[TI-1,:]=PS[Eval.argmax(),:]
    return P    

def GenEvoNorm(PS, pc, pm, Eval): ## Generation algorithm 
    TI, TD = PS.shape
    Best = PS[Eval.argmax()]
    P = np.zeros((TI, TD))
    mean = np.mean(PS, axis=0)
    std = np.std(PS, axis=0)
    for i in range(TI-1):
        for g in range(TD):
            if pc[g] > np.random.rand():
                P[i,g] = mean[g] + std[g]*desnormalization(np.random.rand(), 1, -1)
            else:
                P[i,g] = Best[g] + std[g]*desnormalization(np.random.rand(), 1, -1)
            if pm[g] > np.random.rand():
                P[i,g] = desnormalization(np.random.rand(), max(PS[:,g]), min(PS[:,g]))
    P[TI-1,:]=PS[Eval.argmax(),:]
    return P  

class Test_function():  
    def Ackley(x):
        dum1 = 0
        dum2 = 0
        n = len(x)
        for i in range(n):
            dum1 += x[i]**2
            dum2 += np.cos(np.pi*2*x[i])
            
        f = 20 + np.exp(1) - 20*np.exp(-0.2*np.sqrt(dum1/n)) - np.exp(dum2/n)
        return f
    
    def Griewank(x):
        dum1 = 0
        dum2 = 0
        n = len(x)
        for i in range(n):
            dum1 += x[i]**2
            dum2 *= np.cos(x[i]/np.sqrt(i+1))
            
        f = 1 + dum1/4000 - dum2
        
        return f
    
    def Rastrigin(x):
        dum1 = 0
        n = len(x)
        for i in range(n):
            dum1 += x[i]**2 - 10*np.cos(2*np.pi*x[i])
            
        f = 10*n + dum1
        
        return f
    
    def Schwefel(x):
        dum1 = 0
        n = len(x)
        for i in range(n):
            dum1 += x[i] * np.sin(np.sqrt(np.abs(x[i])))
            
        f = 418.982*n - dum1
        
        return f

def DecodeGenes(individual, models, losses, optimizers, WD, LR):
    ind = individual
    model_key = models[round(desnormalization(ind[0], len(models)-0.50001, -0.4999))]
    loss_key = losses[round(desnormalization(ind[1], len(losses)-0.50001, -0.4999))]
    optimizer_key = optimizers[round(desnormalization(ind[2], len(optimizers)-0.50001, -0.4999))]
    weight_decay = desnormalization(ind[3], WD[1], WD[0])
    lr = desnormalization(ind[4], LR[1], LR[0])
    
    return model_key, loss_key, optimizer_key, weight_decay, lr
    
    
# def test_functions(case=0):
    





















