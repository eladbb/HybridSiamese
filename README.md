# HybridSiamese

## Install Matconvnet

1) Download the beta [23 version](http://www.vlfeat.org/matconvnet/download/) of matconvnet 

2) unpack the downloaded file into a directory of your choosing
   Call the path to this directory <MatConvNet>
   
3) Compile matconvnet (GPU) by following the instructions specified at the original matconvnet [website](http://www.vlfeat.org/matconvnet/install/)

## Create the NIR dataset

1) Dowload the original [dataset](https://ivrl.epfl.ch/supplementary_material/cvpr11/) 

2) Extract the dataset to /nirscenes

3) Using Matlab, run the CreateDataset script inside /nirscenes 
   (this might take a while and will need sufficient amount of memory)
   

## Train

in order to train any of the models specified in the paper just edit the TrainModels.m script as your wish.

To replicate training as specified in the paper, follow the details in the paper:
* Use default learning parameters:
    * Train for 40 epochs
    * Use Learning rate of 0.01 and weight decay of 0.0005
* Softmax config - trained with 80% of the data and hard mining factor of 0.8
* L2 config - trained with 95% of the data and hard mining factor of 0.8


## Eval

We supply our trained models, the NIR benchmark can be evaluated using the EvaluateFar.m script



