function [ data, labels, outputCsv ] = GetPatches( imgNames, dataImgIndices, patchSize )
%ASSMBLEDATA Summary of this function goes here
%   Detailed explanation goes here

data = [];
csvData = {};
imgCounter = 1;      
for imgIdx = dataImgIndices
    visImg = rgb2gray(imread(fullfile('data',imgNames(imgIdx+1).name)));
    irImg = imread(fullfile('data',imgNames(imgIdx).name)); 
    imgData = [];
    rowSpacing = 1:16:size(visImg,1);
    colSpacing = 1:16:size(visImg,2);
    sampleIdx = 1;
    for rowIdx = rowSpacing
        for colIdx = colSpacing
            if (((colIdx + patchSize - 1) > size(visImg,2)) || ((rowIdx + patchSize - 1) > size(visImg,1)))
                continue
            else
                visPatch = visImg(rowIdx: rowIdx + patchSize - 1, colIdx:colIdx + patchSize - 1);
                irPatch = irImg(rowIdx: rowIdx + patchSize - 1, colIdx:colIdx + patchSize - 1);   
                sample = cat(3,visPatch,irPatch);
                imgData = cat(4,imgData,sample); 
                sampleIdx = sampleIdx + 1;         
                csvData = [csvData;{'positive',strrep(imgNames(imgIdx+1).name,'jpg','ppm'),colIdx+31,rowIdx+31,strrep(imgNames(imgIdx).name,'jpg','ppm'),colIdx+31,rowIdx+31}];
            end
        end
    end
    data = cat(4,data,imgData);
    disp(strcat(num2str(imgCounter),'/',num2str(length(dataImgIndices))))
    imgCounter = imgCounter + 1; 
end

% Add negatives
shuffled = 1;
while shuffled > 0 
    disp('trial')
    shuffle = randperm(size(data,4));
    shuffled = length(find(shuffle == 1:size(data,4)));
end
negData = data;
negData(:,:,2,:) = data(:,:,2,shuffle);
data = cat(4,data,negData);
labels =  [ones(size(negData,4),1); 2*ones(size(negData,4),1)];

negCsvData = csvData;
negCsvData(:,1) = {'negative'};
negCsvData(:,5:7) = csvData(shuffle,5:7);

outputCsv = {};
imagesIds = unique({csvData{:,2}});
for id = imagesIds
    outputCsv = [outputCsv;csvData(strcmp(id,csvData(:,2)),:);negCsvData(strcmp(id,csvData(:,2)),:)];
end

% csvData = [csvData;negCsvData];

shuffle = randperm(size(data,4));
data = data(:,:,:,shuffle);
labels = labels(shuffle);

end

