function [net, info] = cnn_siamese(imdb,varargin)

run('vl_setupnn.m') ;

opts.modelType = '' ;
opts.hardMiningFactor = 1;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.initializeFrom = '';
opts.expDir = '' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = '';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = '' ;
opts.train = struct() ;
opts.train.learningRate = 0.001 ;
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 100;

opts.train.derOutputs = {'Objective', 1};
opts.train.gpus = 1;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------
networkInitHandle = [];
switch opts.modelType
    case 'Hybrid_Siamese_Multiple_L2'
        net = Hybrid_Siamese_Multiple_L2_init();
        networkInitHandle = @Hybrid_Siamese_Multiple_L2_init;
    case 'Hybrid_Siamese_Multiple_Softmax'
        net = Hybrid_Siamese_Multiple_Softmax_init();
        networkInitHandle = @Hybrid_Siamese_Multiple_Softmax_init;
 
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
    case 'softmax', trainfn = @cnn_train;
    case 'l2', trainfn = @cnn_train;    
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'networkModel',networkInitHandle,...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'softmax'
    bopts = struct('numGpus', numel(opts.train.gpus),'hardMiningFactor',opts.hardMiningFactor) ;
    fn = @(x,y,net,mode) getDagNNBatchSoftmaxConfig(bopts,x,y,net,mode) ;
  case 'l2'
    bopts = struct('numGpus', numel(opts.train.gpus),'hardMiningFactor',opts.hardMiningFactor) ;
    fn = @(x,y,net,mode) getDagNNBatchL2Config(bopts,x,y,net,mode) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatchSoftmaxConfig(opts, imdb, batch,net,mode)
% ----------------------------------------------------------------  ---------
if (strcmp(mode,'train')  && ~isempty(net))
    if opts.hardMiningFactor > 0
        inputs = getBatchForTrainSoftmaxHardMining(opts, imdb, batch,net); 
    else
        inputs = getBatchForTrainSoftmax(opts, imdb, batch,net); 
    end
else
    % Validation epoch, same as first iteration
    inputs = getBatchForValidationEpoch(opts, imdb, batch);
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatchL2Config(opts, imdb, batch,net,mode)
% ----------------------------------------------------------------  ---------
if (strcmp(mode,'train')  && ~isempty(net))
    if opts.hardMiningFactor > 0
        inputs = getBatchForTrainL2HardMining(opts, imdb, batch,net);
    else
        inputs = getBatchForTrainL2(opts, imdb, batch,net);
    end
    
else
    % Validation epoch, same as first iteration
    inputs = getBatchForValidationEpoch(opts, imdb, batch);
end

% -------------------------------------------------------------------------
function inputs = getBatchForValidationEpoch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ; 

images = reshape(images,size(imdb.images.data,1),size(imdb.images.data,1),1,length(batch)*2);

imagesType1 = single(images(:,:,:,1:2:length(batch)*2));
imagesType2 = single(images(:,:,:,2:2:length(batch)*2));

visImagesAsymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanIrImg);

visImagesSymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanImg);
irImagesSymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanImg);


if opts.numGpus > 0
  visImagesSymmetric = gpuArray(visImagesSymmetric);
  irImagesSymmetric = gpuArray(irImagesSymmetric);
  visImagesAsymmetric = gpuArray(visImagesAsymmetric);
  irImagesAsymmetric = gpuArray(irImagesAsymmetric);
end

inputs = {'siamese_left_symmetric_input',visImagesSymmetric,'siamese_right_symmetric_input',irImagesSymmetric,...
    'siamese_left_Asymmetric_input',visImagesAsymmetric,'siamese_right_Asymmetric_input',irImagesAsymmetric,...
    'labels', imdb.images.labels(batch)} ;

% -------------------------------------------------------------------------
function inputs = getBatchForTrainSoftmax(opts, imdb, batch,net)
% -------------------------------------------------------------------------
batchSize = length(batch);

labels = imdb.images.labels(batch);

images = imdb.images.data(:,:,:,batch) ;
% split images from 2channel format to folowing separate images
images = reshape(images,size(imdb.images.data,1),size(imdb.images.data,1),1,length(batch)*2);

images = augmentImages(images);

% Dipart images to different vectors, one for each type
imagesType1 = single(images(:,:,:,1:2:length(batch)*2));
imagesType2 = single(images(:,:,:,2:2:length(batch)*2));

% Normalize the data by subtracting the mean for each type and net branch (symmetric, aSymmetric)
visImagesAsymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanIrImg);
visImagesSymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanImg);
irImagesSymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanImg);

if opts.numGpus > 0
    visImagesSymmetric = gpuArray(visImagesSymmetric);
    irImagesSymmetric = gpuArray(irImagesSymmetric);
    visImagesAsymmetric = gpuArray(visImagesAsymmetric);
    irImagesAsymmetric = gpuArray(irImagesAsymmetric);
end

finalVisImagesAsymmetric = cat(4,visImagesAsymmetric,visImagesAsymmetric);
finalIrImagesAsymmetric = cat(4,irImagesAsymmetric,flip(irImagesAsymmetric,4));
finalVisImagesSymmetric = cat(4,visImagesSymmetric,visImagesSymmetric);
finalIrImagesSymmetric = cat(4,irImagesSymmetric,flip(irImagesSymmetric,4));

inputs = {'siamese_left_symmetric_input',finalVisImagesSymmetric,'siamese_right_symmetric_input',finalIrImagesSymmetric,...
    'siamese_left_Asymmetric_input',finalVisImagesAsymmetric,'siamese_right_Asymmetric_input',finalIrImagesAsymmetric,...
    'labels', [ones(batchSize,1);2*ones(batchSize,1)]} ;

% -------------------------------------------------------------------------
function inputs = getBatchForTrainSoftmaxHardMining(opts, imdb, batch,net)
% -------------------------------------------------------------------------
batchSize = length(batch);

labels = imdb.images.labels(batch);

images = imdb.images.data(:,:,:,batch) ;
% split images from 2channel format to folowing separate images
images = reshape(images,size(imdb.images.data,1),size(imdb.images.data,1),1,length(batch)*2);

images = augmentImages(images);

% Dipart images to different vectors, one for each type
imagesType1 = single(images(:,:,:,1:2:length(batch)*2));
imagesType2 = single(images(:,:,:,2:2:length(batch)*2));

% Normalize the data by subtracting the mean for each type and net branch (symmetric, aSymmetric)
visImagesAsymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanIrImg);
visImagesSymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanImg);
irImagesSymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanImg);

if opts.numGpus > 0
    visImagesSymmetric = gpuArray(visImagesSymmetric);
    irImagesSymmetric = gpuArray(irImagesSymmetric);
    visImagesAsymmetric = gpuArray(visImagesAsymmetric);
    irImagesAsymmetric = gpuArray(irImagesAsymmetric);
end

inputs = {'siamese_left_symmetric_input',visImagesSymmetric,'siamese_right_symmetric_input',irImagesSymmetric,...
    'siamese_left_Asymmetric_input',visImagesAsymmetric,'siamese_right_Asymmetric_input',irImagesAsymmetric,...
    'labels', labels} ;

% THIS SECTION IS DESIGNED TO FIND HARD NEGATIVES FOR THE CURRENT BATCH

net.mode = 'test';
net.vars(net.getVarIndex('cnv7_left_siamese_x')).precious = 1;
net.vars(net.getVarIndex('cnv7_right_siamese_x')).precious = 1;
% Forward the original samples throuh the dagNet so we can have the final
% features for each sample, now we can check any possible combination of images by simply
% forwarding the combination features throuh the decision part of the net
net.eval(inputs);
% Features are stored in 'cnv7_left_siamese_x' and 'cnv7_right_siamese_x'
type1Features = net.vars(net.getVarIndex('cnv7_left_siamese_x')).value;
type2Features = net.vars(net.getVarIndex('cnv7_right_siamese_x')).value;

% cut and initialize the decision network
decisionNet = net.copy();
decisionNet.params = net.params;
% relevantParams = [];
while ~strcmp(decisionNet.layers(1).name,'decsionConcatLayer')
    decisionNet.removeLayer(decisionNet.layers(1).name);
end
decisionNet.rebuild();
decisionNet.vars(end-2).precious = 1;decisionNet.vars(end).precious = 1;

decisionNet.mode = 'test';
decisionNet.move('gpu') ;

finalVisImagesAsymmetric = cat(4,visImagesAsymmetric,visImagesAsymmetric);
finalIrImagesAsymmetric = cat(4,irImagesAsymmetric,flip(irImagesAsymmetric,4));
finalVisImagesSymmetric = cat(4,visImagesSymmetric,visImagesSymmetric);
finalIrImagesSymmetric = cat(4,irImagesSymmetric,flip(irImagesSymmetric,4));
hardNegatives = randperm(batchSize,floor(batchSize * opts.hardMiningFactor));
taken = [];
% losses = zeros(batchSize,1);
for sampleIdx = 1:length(hardNegatives)
        patchIdx = hardNegatives(sampleIdx);
        type1Inputs = repmat(type1Features(1,1,:,patchIdx),[1 1 1 batchSize]);
        type2Inputs = type2Features;
        taken = [taken patchIdx];
        % Evaluate the combinations
        decisionNetInputs = {'cnv7_left_siamese_x',type1Inputs,'cnv7_right_siamese_x',type2Inputs,...
                            'labels', 2*ones(batchSize,1)} ;
        decisionNet.eval(decisionNetInputs);

        % Extract scores and sort them
        scores = squeeze(decisionNet.vars(end-2).value)';
        scoresNormalized = exp(scores(:,2))./sum(exp(scores),2);
        loss = gather(-log(scoresNormalized));
        [sortedLoss,sorted_index] = sort(loss,'descend');
        [C,ia] = setdiff(sorted_index,taken,'stable' );

        taken(end) = sorted_index(ia(1)); 
%         losses(sampleIdx) = sortedLoss(taken(end))
        finalIrImagesAsymmetric(:,:,:,batchSize + patchIdx) = irImagesAsymmetric(:,:,:,taken(end));
        finalIrImagesSymmetric(:,:,:,batchSize + patchIdx) = irImagesSymmetric(:,:,:,taken(end));
end

% figure
% for idx = 129:1:2*batchSize
%     subplot(2,2,1)
%     imshow(uint8(finalVisImagesAsymmetric(:,:,:,idx) + imdb.meta.meanVisImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,2)
%     imshow(uint8(finalIrImagesAsymmetric(:,:,:,idx) + imdb.meta.meanIrImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,3)
%     imshow(uint8(finalVisImagesSymmetric(:,:,:,idx) + imdb.meta.meanImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,4)
%     imshow(uint8(finalIrImagesSymmetric(:,:,:,idx) + imdb.meta.meanImg));
% title(strcat(',idx = ',num2str(idx)))%     title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     waitforbuttonpress
% end
inputs = {'siamese_left_symmetric_input',finalVisImagesSymmetric,'siamese_right_symmetric_input',finalIrImagesSymmetric,...
    'siamese_left_Asymmetric_input',finalVisImagesAsymmetric,'siamese_right_Asymmetric_input',finalIrImagesAsymmetric,...
    'labels', [ones(batchSize,1);2*ones(batchSize,1)]} ;
% net.mode = 'test';

% -------------------------------------------------------------------------
function images = augmentImages(images)
% -------------------------------------------------------------------------
augmentChoise = randi(4);
switch augmentChoise
   case 1
      images = flipud(images);
   case 2
      images = permute(images, [2 1 3 4]);
   case 3
      images = fliplr(images);
    otherwise    
end


% -------------------------------------------------------------------------
function inputs = getBatchForTrainL2(opts, imdb, batch,net)
% -------------------------------------------------------------------------
batchSize = length(batch);

labels = imdb.images.labels(batch);

images = imdb.images.data(:,:,:,batch) ;
% split images from 2channel format to folowing separate images
images = reshape(images,size(imdb.images.data,1),size(imdb.images.data,1),1,length(batch)*2);

images = augmentImages(images);

% Dipart images to different vectors, one for each type
imagesType1 = single(images(:,:,:,1:2:length(batch)*2));
imagesType2 = single(images(:,:,:,2:2:length(batch)*2));

% Normalize the data by subtracting the mean for each type and net branch (symmetric, aSymmetric)
visImagesAsymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanIrImg);
visImagesSymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanImg);
irImagesSymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanImg);

if opts.numGpus > 0
    visImagesSymmetric = gpuArray(visImagesSymmetric);
    irImagesSymmetric = gpuArray(irImagesSymmetric);
    visImagesAsymmetric = gpuArray(visImagesAsymmetric);
    irImagesAsymmetric = gpuArray(irImagesAsymmetric);
end

finalVisImagesAsymmetric = cat(4,visImagesAsymmetric,visImagesAsymmetric);
finalIrImagesAsymmetric = cat(4,irImagesAsymmetric,flip(irImagesAsymmetric,4));
finalVisImagesSymmetric = cat(4,visImagesSymmetric,visImagesSymmetric);
finalIrImagesSymmetric = cat(4,irImagesSymmetric,flip(irImagesSymmetric,4));

inputs = {'siamese_left_symmetric_input',finalVisImagesSymmetric,'siamese_right_symmetric_input',finalIrImagesSymmetric,...
    'siamese_left_Asymmetric_input',finalVisImagesAsymmetric,'siamese_right_Asymmetric_input',finalIrImagesAsymmetric,...
    'labels', [ones(batchSize,1);-1*ones(batchSize,1)]} ;
% net.mode = 'test';

% -------------------------------------------------------------------------
function inputs = getBatchForTrainL2HardMining(opts, imdb, batch,net)
% -------------------------------------------------------------------------
batchSize = length(batch);

labels = imdb.images.labels(batch);

images = imdb.images.data(:,:,:,batch) ;
% split images from 2channel format to folowing separate images
images = reshape(images,size(imdb.images.data,1),size(imdb.images.data,1),1,length(batch)*2);

images = augmentImages(images);

% Dipart images to different vectors, one for each type
imagesType1 = single(images(:,:,:,1:2:length(batch)*2));
imagesType2 = single(images(:,:,:,2:2:length(batch)*2));

% Normalize the data by subtracting the mean for each type and net branch (symmetric, aSymmetric)
visImagesAsymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanIrImg);
visImagesSymmetric = bsxfun(@minus,imagesType1,imdb.meta.meanImg);
irImagesSymmetric = bsxfun(@minus,imagesType2,imdb.meta.meanImg);

if opts.numGpus > 0
    visImagesSymmetric = gpuArray(visImagesSymmetric);
    irImagesSymmetric = gpuArray(irImagesSymmetric);
    visImagesAsymmetric = gpuArray(visImagesAsymmetric);
    irImagesAsymmetric = gpuArray(irImagesAsymmetric);
end

inputs = {'siamese_left_symmetric_input',visImagesSymmetric,'siamese_right_symmetric_input',irImagesSymmetric,...
    'siamese_left_Asymmetric_input',visImagesAsymmetric,'siamese_right_Asymmetric_input',irImagesAsymmetric,...
    'labels', labels} ;

% THIS SECTION IS DESIGNED TO FIND HARD NEGATIVES FOR THE CURRENT BATCH

net.mode = 'test';
net.vars(net.getVarIndex('left_Normalized_x')).precious = 1;
net.vars(net.getVarIndex('right_Normalized_x')).precious = 1;
% Forward the original samples throuh the dagNet so we can have the final
% features for each sample, now we can check any possible combination of images by simply
% forwarding the combination features throuh the decision part of the net
net.eval(inputs);
% Features are stored in 'cnv7_left_siamese_x' and 'cnv7_right_siamese_x'
type1Features = net.vars(net.getVarIndex('left_Normalized_x')).value;
type2Features = net.vars(net.getVarIndex('right_Normalized_x')).value;

% cut and initialize the decision network
decisionNet = net.copy();
decisionNet.params = net.params;
decisionNet.mode = 'test';
decisionNet.move('gpu') ;

% relevantParams = [];
while ~strcmp(decisionNet.layers(1).name,'l2Dist_Final')
    decisionNet.removeLayer(decisionNet.layers(1).name);
end
decisionNet.rebuild();
decisionNet.vars(end-3).precious = 1;decisionNet.vars(end-1).precious = 1;decisionNet.vars(end-2).precious = 1;decisionNet.vars(end).precious = 1;



finalVisImagesAsymmetric = cat(4,visImagesAsymmetric,visImagesAsymmetric);
finalIrImagesAsymmetric = cat(4,irImagesAsymmetric,flip(irImagesAsymmetric,4));
finalVisImagesSymmetric = cat(4,visImagesSymmetric,visImagesSymmetric);
finalIrImagesSymmetric = cat(4,irImagesSymmetric,flip(irImagesSymmetric,4));
hardNegatives = randperm(batchSize,floor(batchSize * opts.hardMiningFactor));
taken = [];
% losses = zeros(batchSize,1);
for sampleIdx = 1:length(hardNegatives)
        patchIdx = hardNegatives(sampleIdx);
        type1Inputs = repmat(type1Features(1,1,:,patchIdx),[1 1 1 batchSize]);
        type2Inputs = type2Features;
        taken = [taken patchIdx];
        % Evaluate the combinations
        decisionNetInputs = {'left_Normalized_x',type1Inputs,'right_Normalized_x',type2Inputs,...
                            'labels', -1*ones(batchSize,1)} ;
        decisionNet.eval(decisionNetInputs);

        % Extract scores and sort them
        distance = gather(squeeze(decisionNet.vars(end-3).value)');
        [sortedDistance,sorted_index] = sort(distance);
        % Prevent from patches already in the batch to be chosen again
        % (this includes the positive sample also)
        [C,ia] = setdiff(sorted_index,taken,'stable' );

        taken(end) = sorted_index(ia(1)); 
%         losses(sampleIdx) = sortedLoss(taken(end))
        finalIrImagesAsymmetric(:,:,:,batchSize + patchIdx) = irImagesAsymmetric(:,:,:,taken(end));
        finalIrImagesSymmetric(:,:,:,batchSize + patchIdx) = irImagesSymmetric(:,:,:,taken(end));
end

% figure
% for idx = 129:1:2*batchSize
%     subplot(2,2,1)
%     imshow(uint8(finalVisImagesAsymmetric(:,:,:,idx) + imdb.meta.meanVisImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,2)
%     imshow(uint8(finalIrImagesAsymmetric(:,:,:,idx) + imdb.meta.meanIrImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,3)
%     imshow(uint8(finalVisImagesSymmetric(:,:,:,idx) + imdb.meta.meanImg));title(strcat(',idx = ',num2str(idx)))%title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     subplot(2,2,4)
%     imshow(uint8(finalIrImagesSymmetric(:,:,:,idx) + imdb.meta.meanImg));
% title(strcat(',idx = ',num2str(idx)))%     title(strcat(num2str(labels(idx)),',idx = ',num2str(idx)))
%     waitforbuttonpress
% end
inputs = {'siamese_left_symmetric_input',finalVisImagesSymmetric,'siamese_right_symmetric_input',finalIrImagesSymmetric,...
    'siamese_left_Asymmetric_input',finalVisImagesAsymmetric,'siamese_right_Asymmetric_input',finalIrImagesAsymmetric,...
    'labels', [ones(batchSize,1);-1*ones(batchSize,1)]} ;
% net.mode = 'test';
