
sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'};
sequencesResults = cell(length(sequences),1);

networkPath = 'TrainedModels\L2Model.mat';

load(networkPath)
dagNet = dagnn.DagNN.loadobj(net);
dagNet.mode = 'test';

dagNet.vars(end-2).precious = 1;
dagNet.vars(end-3).precious = 1;
useGpu = 1;
if (useGpu)
    dagNet.move('gpu') ;
end
load('nirscenes\country64x64PositiveOnly_0.95.mat','meta');
far = [];
% -- For each sequence
for sequenceIdx=1:length(sequences)
    disp(strcat('Processing - ',sequences(sequenceIdx)));
    %   -- Read csv file
    filename = fullfile('nirscenes','csv',strcat(sequences(sequenceIdx),'.csv'));
    keypointsData = table2cell(readtable(filename{1}));
    imagesNames = unique(char(keypointsData(:,2)),'rows');
    hybridScores = [];
    labels = [];
    for imgIdx = 1:length(imagesNames)
        disp(strcat(num2str(imgIdx),'/', num2str(length(imagesNames))))
        imgPatchesDataPath = fullfile('nirscenes',sequences{sequenceIdx},strrep(imagesNames(imgIdx,:), 'ppm', 'mat'));
        load(imgPatchesDataPath);
        numSamples = size(imgBenchmarkData,4);
        visPatches = single(imgBenchmarkData(:,:,1,:));
        irPatches = single(imgBenchmarkData(:,:,2,:));
        visImagesSymmetric = bsxfun(@minus,visPatches,meta.meanImg);
        irImagesSymmetric = bsxfun(@minus,irPatches,meta.meanImg);
        visImagesAsymmetric = bsxfun(@minus,visPatches,meta.meanVisImg);
        irImagesAsymmetric = bsxfun(@minus,irPatches,meta.meanIrImg);
        
        batchSize = 1024;
        for batchIterStart = 1:batchSize:length(visPatches)
            batchIterEnd = min([batchIterStart+batchSize-1 length(visPatches)]);
            
            visBatchSymmetric = gpuArray(visImagesSymmetric(:,:,:,batchIterStart:batchIterEnd));
            irBatchSymmetric = gpuArray(irImagesSymmetric(:,:,:,batchIterStart:batchIterEnd));
            visBatchAsymmetric = gpuArray(visImagesAsymmetric(:,:,:,batchIterStart:batchIterEnd));
            irBatchAsymmetric = gpuArray(irImagesAsymmetric(:,:,:,batchIterStart:batchIterEnd));
            
            inputs = {'siamese_left_symmetric_input',visBatchSymmetric,'siamese_right_symmetric_input',irBatchSymmetric,...
                'siamese_left_Asymmetric_input',visBatchAsymmetric,'siamese_right_Asymmetric_input',irBatchAsymmetric,...
                'labels', gpuArray(ones(size(visBatchSymmetric,4),1))} ;
            
            dagNet.eval(inputs);
            hybridScores = [hybridScores;squeeze(gather(dagNet.getVar('l2distOutput').value(:,:,:,1:size(visBatchSymmetric,4))))];
        end
        labels = [labels; imgBenchmarkDataLabels];
    end
    far  = Far95Recall( hybridScores,labels,0.95,'l2');
    finalStats.config = 'l2';
    finalStats.sequence = sequences{sequenceIdx};
    finalStats.hybridScores = hybridScores;
    finalStats.labels = labels;
    finalStats.far = far;
    sequencesResults{sequenceIdx} = finalStats;
end
save('TrainedModels\L2ModelResults.mat','sequencesResults');