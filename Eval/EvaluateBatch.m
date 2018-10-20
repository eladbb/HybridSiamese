function [ hybridScores ] = EvaluateBatch( dagNet, batchSize,testData, meta, modelType )
%EVALUATEBATCH Summary of this function goes here
%   Detailed explanation goes here
hybridScores = [];
visPatches = single(testData(:,:,1,:));
irPatches = single(testData(:,:,2,:));
visImagesSymmetric = bsxfun(@minus,visPatches,meta.meanImg);
irImagesSymmetric = bsxfun(@minus,irPatches,meta.meanImg);
visImagesAsymmetric = bsxfun(@minus,visPatches,meta.meanVisImg);
irImagesAsymmetric = bsxfun(@minus,irPatches,meta.meanIrImg);

for batchIterStart = 1:batchSize:length(visPatches)
    batchIterEnd = min([batchIterStart+batchSize-1 length(visPatches)]);
    visBatchSymmetric = gpuArray(visImagesSymmetric(:,:,:,batchIterStart:batchIterEnd));
    irBatchSymmetric = gpuArray(irImagesSymmetric(:,:,:,batchIterStart:batchIterEnd));
    visBatchAsymmetric = gpuArray(visImagesAsymmetric(:,:,:,batchIterStart:batchIterEnd));
    irBatchAsymmetric = gpuArray(irImagesAsymmetric(:,:,:,batchIterStart:batchIterEnd));
    if strcmp(modelType,'Softmax')
        inputs = {'siamese_left_symmetric_input',visBatchSymmetric,'siamese_right_symmetric_input',irBatchSymmetric,...
            'siamese_left_Asymmetric_input',visBatchAsymmetric,'siamese_right_Asymmetric_input',irBatchAsymmetric,...
            'labels', gpuArray(ones(size(visBatchSymmetric,4),1))} ;
        dagNet.eval(inputs);
        hybridScores = [hybridScores;squeeze(gather(dagNet.getVar('cnv9_Scores_siamese_x').value(:,:,:,1:size(visBatchSymmetric,4))))'];
    else
        inputs = {'siamese_left_symmetric_input',visBatchSymmetric,'siamese_right_symmetric_input',irBatchSymmetric,...
                    'siamese_left_Asymmetric_input',visBatchAsymmetric,'siamese_right_Asymmetric_input',irBatchAsymmetric,...
                    'labels', gpuArray(ones(size(visBatchSymmetric,4),1))} ;
        dagNet.eval(inputs);
        hybridScores = [hybridScores;squeeze(gather(dagNet.getVar('l2distOutput').value(:,:,:,1:size(visBatchSymmetric,4))))];       
    end
end
end

