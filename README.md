# Gen-CNN
Gen-CNN: An all-in-one framework for the automated generation of CNNs for image classification

Gen-CNN is an AutoML framework based on a genetic algorithm for hyperparameter optimization, which automatically generates optimized CNNs for image classification tasks.

Gen-CNN takes a dataset as input and does everything else automatically. It selects and trains a CNN to classify images from the same distribution as the provided dataset. In the end, the framework is going to provide performance metrics on a partition of the provided dataset that is left out of any other procedure (test partition) to give an indication of the expected model performance.

You only need to run the "GENCNN_0.2" file. There you can find instructions to use the framework. You can find the same instructions in the end of this file.

To use current version of Gen-CNN you need the following:
- Python 3
- Numpy
- Pandas
- Pytorch 1.13.1 and CUDA 11.6

GENCNN_0.2
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
        
    3.- THE ALGORITHM WILL TAKE IT FROM HERE. HOWEVER, SINCE IT CAN TAKE TIME AND PROBLEMS 
        MAY ARISE, SUCH AS OUT OF MEMORY OR A LOST OF POWER, THE ALGORITHM SAVES
        THE RESULTS AFTER EVERY HYPERPARAMETER OPTIMIZATION EPOCH. IT IS GOING TO ASK YOU
        IF YOU HAVE PREVIOUS RESULTS IN THE DATASET. SO, IF YOU WANT TO RESUME A PREVIOUS
        PROCEDURE JUST TYPE: YES, OR yes, OR Y, OR y, IF YOU DONT, TYPE ANYTHING ELSE, E.G. NO.
        
        THE ALGORITHM IS GOING TO SAVE ALL THE RESULTS (INCLUDING THE RESULTING MODEL)
        IN A FOLDER WITH THE NAME OF THE DATASET INSIDE A FOLDER CALLED GenCNN THAT IT 
        IS GOING TO CREATE IN THE DIRECTORY THAT YOU PROVIDED IN STEP 1.


        
