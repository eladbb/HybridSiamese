clear 
clc
opts = [];
opts.train = struct() ;
matConvNetPath = '..\Matlab3rdParties\matconvnet-1.0-beta23';
run(fullfile(matConvNetPath,'matlab','vl_setupnn.m'))
addpath('Models')
rng('shuffle');

config = 'softmax';
trainData = 'country64x64PositiveOnly_0.95.mat'; % 'country64x64PositiveOnly_0.8.mat'
experimentName = 'trial';

learningRate = 0.01;
weightDecay = 0.0005;
hmFactor = 0.8;

if strcmp(config,'softmax')
    opts.networkType = 'softmax';
    opts.modelType = 'Hybrid_Siamese_Multiple_Softmax';
    rootDir = fullfile('TrainingSessions',strcat('softmax_',experimentName));    
    opts.imdbPath = fullfile('nirscenes', trainData);
    opts.train.derOutputs = {'hybridObjective',1,'hybridError',1,...
        'SymmetricObjective',1,'SymmetricError',1,...
        'AsymmetricObjective',1,'AsymmetricError',1};
else
    opts.networkType = 'l2';
    opts.modelType = 'Hybrid_Siamese_Multiple_L2';
    rootDir = fullfile('TrainingSessions',strcat('l2_',experimentName));
    opts.imdbPath = fullfile('nirscenes', trainData);
    paramsPath = fullfile('TrainingSessions','lcsis\l2\params.mat');
    
%   hybridModel implementation in L2 config is slightly different than the softmax.
%   for better understanding we choose to observe the overall loss separated to
%   positive and negative examples. The InfoNeg metric specifies the
%   precentage of informative negative examples (distance lower than 1).
    
    opts.train.derOutputs = {'SymmetricPos',1,'SymmetricNeg',1,'SymmetricInfoNeg',0,...
                        'AsymmetricPos',1,'AsymmetricNeg',1,'AsymmetricInfoNeg',0,...
                        'HybridPos',1,'HybridNeg',1,'HybridInfoNeg',0};

end

opts.train.gpus = 1;

opts.expDir = fullfile(rootDir,num2str(hmFactor));
opts.hardMiningFactor = hmFactor;
imdb = load(opts.imdbPath) ;
if ~exist(opts.expDir, 'dir')
	mkdir(opts.expDir)  ;
end
opts.train.weightDecay = weightDecay;
if strcmp(config,'softmax')
	opts.train.learningRate = learningRate*ones(1,75);
	opts.train.numEpochs = 105;
else
	opts.train.learningRate = learningRate*ones(1,40);
	opts.train.numEpochs = 40;
end
[net, info] = cnn_siamese(imdb,opts);
    save(strcat(opts.expDir,'\netResults.mat'),'net','info');
