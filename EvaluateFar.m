clear
clc
close all
matconvnetPath = '..\..\Matlab3rdParties\matconvnet-1.0-beta23';
run(fullfile(matconvnetPath,'matlab','vl_setupnn.m'))

addpath('..')

sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'};

networkPaths = {'TrainedModels\L2ModelResults.mat', 'TrainedModels\SoftmaxModelResults.mat'};
models{'L2Model', 'SoftmaxModel'}

stats = zeros(length(networkPaths),length(epochs));
for networkIdx = 1:length(networkPaths)
	if ~exist('TrainedModels\SoftmaxModelResults.mat','file')
		% Create results file 
		EvaluateValidationSetSoftmax
		EvaluateValidationSetL2
	end

    load(networkPaths{networkIdx})  
    hybridFar = zeros(1,length(sequencesResults)-1);
    for seqIdx = 1:length(sequencesResults)                         
        hybridFar(seqIdx) = Far95Recall( sequencesResults{seqIdx}.hybridScores,sequencesResults{seqIdx}.labels,0.95,sequencesResults{seqIdx}.config);                               
    end
    categories = [277504,240896,376832,60672,151296,101376,164608,147712,143104];
    meanHybrid = sum(hybridFar(2:end).*categories(2:end))/sum(categories(2:end));
    
    resultStr = strcat( models{networkIdx},';  Mean: ',num2str(meanHybrid));
    disp(resultStr)            
    hybridCategories = '';
    for categoryIdx = 1:length(sequencesResults)  
        hybridCategories = strcat(hybridCategories,' ; ',sequencesResults{categoryIdx}.sequence,' = ', num2str(hybridFar(categoryIdx)));
    end
    disp(strcat('Hybrid: ',hybridCategories))
    disp('------------------------------------------------------------')
end    