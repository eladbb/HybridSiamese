function [ far ] = Far95Recall( scores,labels,recall_goal,config)
%FAR95RECALL Summary of this function goes here
%   Detailed explanation goes here
if strcmp(config,'Softmax')
    scoresNormalized = exp(scores(:,1))./sum(exp(scores),2);
    [sortedScores,sorted_index] = sort(scoresNormalized,'descend');
else
    [sortedScores,sorted_index] = sort(scores);
end

number_of_true_matches = sum(labels == 1);
threshold_number = recall_goal * number_of_true_matches;
tp = 0;
count = 0;
% Run until find 95 % recall
for i=1:length(sorted_index)
    count = count + 1.0;
    if (labels(sorted_index(i)) == 1)
        tp = tp + 1.0;
    end
    if (tp >= threshold_number)
        break;
    end
end
far =  ((count - tp) / count) * 100.0;

end

