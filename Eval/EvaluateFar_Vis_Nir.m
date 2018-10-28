clear
clc
close all
matconvnetPath = '..\..\Matlab3rdParties\matconvnet-1.0-beta23';
run(fullfile(matconvnetPath,'matlab','vl_setupnn.m'))

addpath('..\Layers')

sequences = {'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'};

modelsDir = '..\TrainedModels\Vis-Nir';
datasetPath = '..\datasets\Vis-Nir\';
outputResultsDir = 'Vis-Nir';

networkNames = {'Softmax_100_model_hm_0'};%, 'Softmax_100_model_hm_0.8','L2_25_model_hm_0','L2_25_model_hm_0.8'};

batchSize = 1024;

load(fullfile(datasetPath,'Vis-Nir_Train.mat'),'meta')

stats = zeros(length(networkNames));
for networkIdx = 1:length(networkNames)
    disp(networkNames{networkIdx});
    modelPath = fullfile(modelsDir,strcat(networkNames{networkIdx},'.mat'));
    resultsPath = fullfile(outputResultsDir,strcat(networkNames{networkIdx},'_results.mat'));
    if ~exist(resultsPath,'file')
        if isempty(strfind(networkNames{networkIdx},'Softmax'))
            modelType = 'l2';
        else
            modelType = 'Softmax';
        end
        dagNet = LoadNetworkModel( modelPath);
        
        hybridFar = zeros(1,length(sequences));
        
        for sequenceIdx = 1:length(sequences)
            disp(strcat(' Processing  - ',sequences(sequenceIdx)));
            testPath = fullfile(datasetPath,strcat(sequences{sequenceIdx},'_Test.mat'));
            load(testPath);
            hybridScores = [];
            for idx = 1:50000:size(testData,4)
                endIdx = min(idx + 50000 - 1, size(testData,4));
                hybridScores = [hybridScores;EvaluateBatch( dagNet, batchSize ,testData(:,:,:,idx:endIdx),meta, modelType)];
            end
            hybridFar(sequenceIdx) = Far95Recall( hybridScores,testLabels,0.95,modelType);
        end
        categories = [240896,376832,60672,151296,101376,164608,147712,143104];
        meanHybrid = sum(hybridFar.*categories)/sum(categories);
        
        results.config = modelType;
        results.hybridScores = [];
        results.labels = [];
        results.farSeq = hybridFar;
        results.far = meanHybrid;
        save(resultsPath,'results');
    else
        load(resultsPath)
    end
    hybridCategories = '';
    for sequenceIdx = 1:length(sequences)
        hybridCategories = strcat(hybridCategories,' ; ',sequences{sequenceIdx},' = ', num2str(hybridFar(sequenceIdx)));
    end
    
    resultStr = strcat( networkNames{networkIdx},';  Mean: ',num2str(meanHybrid));
    disp(resultStr)
    disp(strcat('Hybrid: ',hybridCategories))
    disp('------------------------------------------------------------')
end