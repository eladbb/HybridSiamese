clear
matconvnetPath = '..\..\Matlab3rdParties\matconvnet-1.0-beta23';
run(fullfile(matconvnetPath,'matlab','vl_setupnn.m'))

dataset = 'vedai';%'cuhk';%'Vis-Nir_grid';%

addpath('..\Layers')

modelsDir = fullfile('..\TrainedModels',dataset);

outputResultsDir = dataset;

networkNames = {'Softmax_100_model_hm_0', 'Softmax_100_model_hm_0.8',...
                'L2_40_model_hm_0', 'L2_40_model_hm_0.8'};

%Evaluation requires to create the datasets first            
testPath = fullfile('..\datasets',dataset,strcat(dataset,'_Test.mat'));
load(testPath);

batchSize = 1024;

for networkIdx = 1:length(networkNames)
		modelPath = fullfile(modelsDir,strcat(networkNames{networkIdx},'.mat'));
		resultsPath = fullfile(outputResultsDir,strcat(networkNames{networkIdx},'_results.mat'));
        if isempty(strfind(networkNames{networkIdx},'Softmax'))
            modelType = 'l2';
        else
            modelType = 'Softmax';
        end
        if ~exist(resultsPath,'file')
            
            dagNet  = LoadNetworkModel( modelPath);

            hybridScores = EvaluateBatch( dagNet, batchSize ,testData,meta, modelType);
            
            far  = Far95Recall( hybridScores,testLabels,0.95,modelType);
            results.config = modelType;
            results.hybridScores = hybridScores;
            results.labels = testLabels;
            results.far = far;
            save(resultsPath,'results');
        else
            load(resultsPath,'results')
            far = results.far;
        end  
        disp([networkNames{networkIdx}, ': ', num2str(far)])    
end