# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:45:35 2023

@author: Rogelio Garcia
"""

import Performance_metrics
import Training_functions
import torch

"""
INSTRUCTIONS:
    
    1.- THE CODE IS GOING TO ASK FOR A DIRECTORY, THAT IS THE ADDRESS WHERE YOU HAVE THE 
        FOLDER CONTAINING THE DATASET. THE DATASET ONLY HAS TO HAVE THE IMAGES IN FOLDERS
        WITH THE NAME OF THEIR RESPECTIVE CLASSES.
        
        E.G. IN MY COMPUTER I HAVE THE KVASIR FOLDER IN
            C:/Users/Rogelio Garcia/Documents/Images datasets
            
    2.- THEN, THE CODE IS GOING TO ASK FOR THE NAME OF THE DATASET, THAT IS THE NAME OF 
        THE FOLDER WHERE THE DATA IS.
        
        TO CONTINUE WITH THE SAME EXAMPLE, I HAVE A FOLDER WITH THE NAME
        KVASIR-v2 IN THE DIRECTORY THAT I ENTERED IN THE FIRST POINT. INSIDE THE KVASIR-v2
        FOLDER ARE 8 FOLDERS, ONE FOR EACH CLASS, AND EACH FOLDER CONTAINS THE IMAGES
        OF THAT CLASS.
        
    3.- THE ALGORITHM WILL TAKE IT FROM HERE. HOWEVER, SINCE IT CAN TAKE TIME AND THERE 
        MAY ARISE PROBLEMS SUCH AS OUT OF MEMORY OR A LOST OF POWER, THE ALGORITHM SAVES
        THE RESULTS AFTER EVERY HYPERPARAMETER OPTIMIZATION EPOCH. IT IS GOING TO ASK YOU
        IF YOU HAVE PREVIOUS RESULTS IN THE DATASET. SO, IF YOU WANT TO RESUME A PREVIOUS
        PROCEDURE JUST TYPE: YES, OR yes, OR Y, OR y
        
        THE ALGORITHM IS GOING TO SAVE ALL THE RESULTS (INCLUDING THE RESULTING MODEL)
        IN A FOLDER WITH THE NAME OF THE DATASET INSIDE A FOLDER CALLED GenCNN THAT IT 
        IS GOING TO CREATE IN THE DIRECTORY THAT YOU PROVIDED IN STEP 1.

"""

if __name__ == "__main__":
        
    directory = str(input('Please type directory of the dataset: \n'))
    DATABASE = str(input('Please type the dataset name: \n'))
    
    if torch.cuda.is_available():
        device = 'cuda'
    
    print('Your device is: ', device)
    
    """
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
   
       FIRST STEP
       
       THIS CREATES THE DATASETS PARTITONS, FOLDERS, AND PERFORMS AND SAVES THE INFO
       ABOUT THE HYPERPARAMETER OPTIMIZATION
       
       THIS STEP CAN TAKE FROM HOURS TO MINUTES, DEPENDING ON THE DATASET SIZE, THE 
       HYPERPARAMETERS OF THE GENETIC ALGORITHM, AND HARDWARE SPECS
       
       I SET THE GENERATION TO 15 AND INDIVIDUALS TO 35, THIS SETTING SHOULD PROVIDE
       GOOD ENOUGH RESULTS FOR MANY SATASETS. HOWEVER, I LET THESE HYPERPARAMETERS OPEN
       TO CHANGE, SO FEEL FREE TO TRY DIFFERENT SETTINGS BY CHANGING THE gens AND
       individuals VALUES INSIDE THE Training_functions.GetOptHP FUNCTION
   
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
    """   
    
    Training_functions.GetOptHP(directory, DATABASE, gens = 3, eps = 3, PC = [0, 0, 0, 0.9, 0.9], PM = [0.4, 0.4, 0.4, 0.2, 0.2], individuals = 5, device=device)
    
    """
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
   
       SECOND STEP
       
       THIS GENERATES THE MODEL USING THE HYPERPARAMETERS OBTAINED DURING THE FIRST
       STEP. 
       
       YOU CAN CHANGE THE TOTAL NUMBER OF TRAINING EPOCHS BY MODIFYING THE eps VALUE
       INSIDE THE Training_functions.FinalTrainer FUNCTION
   
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
    """
    
    Training_functions.FinalTrainer(directory , DATABASE, eps=50, device=device)
    
    """
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
   
       THIRD STEP
       
       THIS IS ONLY TO CALCULATE THE METRICS ON THE TESTING PARTITION
   
    ************************************************************************************
    ************************************************************************************
    ************************************************************************************
    """
    
    B_metrics, B_CM = Performance_metrics.Test_on_TestPartition(directory, DATABASE, device=device)
    
    
    
    