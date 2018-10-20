function [ dagNet ] = LoadNetworkModel( modelPath)
%LOADSOFTMAXMODEL Summary of this function goes here
%   Detailed explanation goes here
load(modelPath)

dagNet = dagnn.DagNN.loadobj(net);
dagNet.mode = 'test';

dagNet.vars(end-2).precious = 1;
dagNet.vars(end-3).precious = 1; 
   
useGpu = 1;

if (useGpu)
    dagNet.move('gpu') ;
end
end

