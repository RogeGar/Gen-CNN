# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:54:17 2023

@author: Rogelio Garcia
"""
import torch
import time
import Hyperparameters
import torch.nn.functional as F
import Performance_metrics
import numpy as np
import GeneticAlgorithm
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import os
import Dataset_creator


def training_loop(model_key, optimizer_key, lr, weight_decay, loss_key, train_loader, valid_loader, classes, epochs = 20, 
                  device = 'cpu', Training_time = False, scheduler = False):
    model = Hyperparameters.get_model(model_key, classes)
    model.to(device)
    BT = time.time()
    optimizer = Hyperparameters.get_optimizer(optimizer_key, model.parameters(), lr, weight_decay)
    
    if scheduler:
        it_steps = len(train_loader)
        Scheduler = CosineAnnealingLR(optimizer,
                                      T_max = int(epochs*it_steps), # Maximum number of iterations.
                                     eta_min = optimizer.param_groups[0]['lr']*0.1) # Minimum learning rate.
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            torch.cuda.empty_cache()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = model(images)
            loss = Hyperparameters.get_loss(loss_key, preds, F.one_hot(labels, len(classes)).float())
            del images
            del labels
            del preds
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            del loss
            optimizer.step()
            if scheduler:
                Scheduler.step()
                
                
        AT = time.time()
        if Training_time:
            print('Epoch: ', epoch, 'Time: ', format((AT-BT)/60, '.2f'), ' minutes')
            if scheduler:    
                print('Learning rate: ', optimizer.param_groups[0]['lr'])
    # fig = plt.figure()
    # plt.plot(range(len(loss_track)), loss_track, 'm')
    # plt.axis([0,len(loss_track),0.8*min(loss_track),1.2*max(loss_track)])
    
    # os.chdir('C:/Users/Rogelio Garcia/Documents/Doctorado/6 semestre/Automated pipeline')
    # model.load_state_dict(torch.load('Vit_0.pt'))
        
    return model

def FinalTrainer(directory , DATABASE, eps=50, device='cpu'):
    
    Exp_dir = directory + '/GenCNN/' + DATABASE
    
    directory = directory + '/' + DATABASE
    classes = Dataset_creator.get_classes(directory)
    min_shape = Dataset_creator.get_size(directory + '/' + 'test')
    shape = 224
    batch = 32
    if min(min_shape) > 224 :
        shape = 224 * (min(min_shape) // 224 )
        batch = int(32 / ((min(min_shape) // 224)**1) / 2)
    if min(min_shape) < 224 :
        shape = 224 / (224 // min(min_shape))
        batch = int(32 * (224 // (min(min_shape))**1) / 2)
    train_set, valid_set, test_set = Dataset_creator.get_sets(directory, train_size=(shape,shape), train_crop=(shape,shape), valid_size=(shape,shape),
                                  test_size=(shape,shape))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, num_workers=1,
                                                shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch, num_workers=1,
                                                  shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, num_workers=1,
    #                                               shuffle=True)
    
    Best_genes_dum = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'Best_genes.csv')).to_numpy()[:,1:].tolist()
    Best_genes = []
    for i in range(len(Best_genes_dum)):
        Best_genes.append(Best_genes_dum[i][0])


    Sche = True
    
    print("Training with the best Hyperparameters")
    print(Best_genes)
    model = training_loop(Best_genes[0], Best_genes[2], float(Best_genes[3]), float(Best_genes[4]), Best_genes[1], train_loader, valid_loader, classes, epochs = eps, 
          device = device, Training_time = True, scheduler = Sche)
    
    train_preds, train_labels = Performance_metrics.get_all_preds(model, train_loader, 'cuda')
    B_metrics, B_dums = Performance_metrics.get_performance_metrics(classes, train_labels, train_preds)
    print('B train metrics: \n', B_metrics) 
    
    valid_preds, valid_labels = Performance_metrics.get_all_preds(model, valid_loader, 'cuda')
    B_metrics, B_dums = Performance_metrics.get_performance_metrics(classes, valid_labels, valid_preds)
    print('B valid metrics: \n', B_metrics)
    
    torch.save(model.state_dict(), Exp_dir+'/B_model.pt')

def Hyperparameter_Optimization(Exp_dir, classes, train_loader, valid_loader, models, losses, optimizers, LR, WD, TI = 20, TD = 5, Generations = 10, epochs = 5,
                                pc = [0, 0, 0, 0.9, 0.9], pm = [0.35, 0.35, 0.35, 0.15, 0.15], device = 'cpu', scheduler = False, save_results = True, Prev_data = False):
    
    gen_dumi = 0
    Best_MCC = 0
    if Prev_data:
         Pop = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'Last_Pop.csv')).to_numpy()[:,1:]
         BestMCC_track = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'BestMCC_track.csv')).to_numpy()[:,1:].squeeze().tolist()
         if type(BestMCC_track) is float:
             BestMCC_track = [BestMCC_track]  
         gen_dumi = len(BestMCC_track)
         Best_MCC = max(BestMCC_track)
         Exp_evs = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'Exp_evs.csv')).to_numpy()[:,1:].tolist()
         Best_genes_dum = pd.DataFrame(pd.read_csv(Exp_dir+'/'+'Best_genes.csv')).to_numpy()[:,1:].tolist()
         Best_genes = []
         for i in range(len(Best_genes_dum)):
             Best_genes.append(Best_genes_dum[i][0])
         Best_model = Hyperparameters.get_model(Best_genes[0], classes)
         Best_model.load_state_dict(torch.load(Exp_dir+'/'+'Best_model.pt'))
    else:
        Pop = np.random.rand(TI,TD)
        BestMCC_track = []
        Exp_evs = []
    
    for gen in range(Generations-gen_dumi):
        # print(Pop)
        Evs = []
        for indi in range(TI):   
            ind = Pop[indi]
            
            model_key, loss_key, optimizer_key, weight_decay, lr = GeneticAlgorithm.DecodeGenes(ind, models, losses, optimizers, WD, LR)
            
            print('\nGeneration: ', gen+gen_dumi, 'Individual: ', indi)
            print('Model: ', model_key, ', Loss: ', loss_key, ', Optimizer: ', optimizer_key, ', Weight decay: ', format(weight_decay, '.4f'), ', LR: ', format(lr, '.4f'))
            
            model = training_loop(model_key, optimizer_key, lr, weight_decay, loss_key, train_loader, valid_loader, classes, epochs = epochs, 
                  device = device, Training_time = True, scheduler=scheduler)
            
            model.eval()
            train_preds, train_labels = Performance_metrics.get_all_preds(model, train_loader, device)
            train_metrics, train_dums = Performance_metrics.get_performance_metrics(classes, train_labels, train_preds)
            print('Train MCC: ', format(train_metrics.loc['mcc'][0], '.4f'))
         
            valid_preds, valid_labels = Performance_metrics.get_all_preds(model, valid_loader, device)
            valid_metrics, valid_dums = Performance_metrics.get_performance_metrics(classes, valid_labels, valid_preds)
            mcc_valid = valid_metrics.loc['mcc'][0]
            print('Validation MCC: ', format(mcc_valid, '.4f'))
            
            Evs.append(mcc_valid)
            
            if mcc_valid > Best_MCC:
                print('\nNew best model found\n')
                Best_model = model
                Best_MCC = mcc_valid
                
                best_arch, best_loss, best_optimizer, best_wd, best_lr = model_key, loss_key, optimizer_key, weight_decay, lr
                Best_genes = [best_arch, best_loss, best_optimizer, best_wd, best_lr]
            
        BestMCC_track.append(max(Evs))
        Exp_evs.append(Evs)
        
        Evs_n, _, _ = GeneticAlgorithm.normalization(np.array(Evs))
        PS, Evs_PS = GeneticAlgorithm.selecTornN(Pop, Evs_n, N=2)
        if gen < Generations-1:
            # print('Evs:', Evs)
            
            Pop = GeneticAlgorithm.GenAGGR(PS, pc, pm, Evs_PS)
            # print('New population:', Pop)
        print('\nEvs evolutions: \n', BestMCC_track)
        
        if save_results:
            os.chdir(Exp_dir)
            pd.DataFrame(Pop).to_csv(Exp_dir+'/'+'Last_Pop.csv')
            pd.DataFrame(BestMCC_track).to_csv(Exp_dir+'/'+'BestMCC_track.csv')
            pd.DataFrame(Best_genes).to_csv(Exp_dir+'/'+'Best_genes.csv')
            torch.save(Best_model.state_dict(), 'Best_model.pt')
            pd.DataFrame(Exp_evs).to_csv(Exp_dir+'/'+'Exp_evs.csv')
    
    convergence_arch, convergance_loss, convergance_optimizer, convergance_wd, convergance_lr = GeneticAlgorithm.DecodeGenes(Pop[np.argmax(Evs_n)], models, losses, optimizers, WD, LR)
    Convergence_genes = [convergence_arch, convergance_loss, convergance_optimizer, convergance_wd, convergance_lr]
    
    if save_results:
        os.chdir(Exp_dir)
        pd.DataFrame(BestMCC_track).to_csv(Exp_dir+'/'+'_BestMCC_track.csv')
        pd.DataFrame(Best_genes).to_csv(Exp_dir+'/'+'_Best_genes.csv')
        pd.DataFrame(Convergence_genes).to_csv(Exp_dir+'/'+'_Convergence_genes.csv')
        # torch.save(Best_model.state_dict(), '_Best_model.pt')
        pd.DataFrame(Exp_evs).to_csv(Exp_dir+'/'+'_Exp_evs.csv')
    
    return BestMCC_track, Best_genes, Convergence_genes, Best_MCC, Exp_evs 

def GetOptHP(directory, DATABASE, gens = 15, eps = 3, PC = [0, 0, 0, 0.9, 0.9], PM = [0.4, 0.4, 0.4, 0.2, 0.2], individuals = 35, device='cpu'):
        
    save_results = True

    prev = str(input('Are there previous results on this experiment?  \n'))
    if prev in ['yes', 'YES', 'y', 'Y', 'si', 'SI', 's']:
        Prev_data = True
    else:
        Prev_data = False
    Exp_dir = directory + '/GenCNN/' + DATABASE
    
    if os.path.isdir(Exp_dir) is False:
        os.makedirs(Exp_dir)
    
    directory = directory + '/' + DATABASE
    classes = Dataset_creator.get_classes(directory)
    
    train_set, valid_set, test_set = Dataset_creator.get_sets(directory, train_size=(124,124), train_crop=(112,112), valid_size=(112,112),
                                 test_size=(112,112))
    
    models = Hyperparameters.Mobile_Available_models
    losses = Hyperparameters.Available_losses
    optimizers = Hyperparameters.Available_optimizers
    LR = [0.000001, 1]
    WD = [0.000000001, 0.1]
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=1,
                                                shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, num_workers=1,
                                                  shuffle=True)
    
     
    
    BestMCC_track, Best_genes, Convergence_genes, Best_MCC, Exp_evs = Hyperparameter_Optimization(Exp_dir, classes, train_loader, valid_loader, models, losses, optimizers, LR, WD, TI = individuals, 
                                                                                                                        TD = 5, Generations = gens, epochs = eps, pc = PC, pm = PM, device = device, scheduler = False, save_results = save_results, Prev_data = Prev_data)


