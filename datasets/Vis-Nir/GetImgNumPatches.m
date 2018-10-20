function [ numPatches,startIdx,endIdx ] = GetImgNumPatches( keypointsData,imgName )
%GETIMGNUMPATCHES Summary of this function goes here
%   Detailed explanation goes here
numPatches = length(keypointsData);
result = char(keypointsData(:,2)) == repmat(imgName,numPatches,1);
numPatches = sum(result(:,1).*result(:,2).*result(:,3).*result(:,4));
startIdx = find(result(:,1).*result(:,2).*result(:,3).*result(:,4),1);
endIdx = find(result(:,1).*result(:,2).*result(:,3).*result(:,4),1,'last');
end

