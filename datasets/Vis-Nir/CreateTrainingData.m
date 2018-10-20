
clear
sequence = {'country'};

filename = fullfile('data','csv',strcat(sequence{1},'.csv'));
keypointsData = table2cell(readtable(filename));
imagesNames = unique(char(keypointsData(:,2)),'rows');
trainData = zeros(64,64,2,length(keypointsData),'uint8');
trainLabels = zeros(1,length(keypointsData));
disp(sequence{1})
for imgIdx = 1:length(imagesNames)
    disp(strcat(num2str(imgIdx),'/', num2str(length(imagesNames))));
    visImgPath = fullfile('data',sequence{1},strrep(imagesNames(imgIdx,:), 'ppm', 'tiff'));
    irImgPath = fullfile('data',sequence{1},strrep(imagesNames(imgIdx,:), 'rgb.ppm', 'nir.tiff'));
    [ numPatches,startIdx,endIdx ] = GetImgNumPatches( keypointsData,imagesNames(imgIdx,:) );
    imgKeypointsData = keypointsData(startIdx:endIdx,:);
    [imgData, imgLabels] = CropImagePatches(visImgPath, irImgPath, imgKeypointsData);
    trainData(:,:,:,startIdx:endIdx) = imgData;
    trainLabels(startIdx:endIdx) = imgLabels;
end
for idx = 1:size(trainData,4)
    if sum(sum(sum(trainData(:,:,:,idx)))) == 0
        idx
    end
end

shuffle = randperm(length(trainLabels));
Data = trainData(:,:,:,shuffle);
Labels = trainLabels(shuffle);

positiveOnly = Data(:,:,:,Labels == 1);
negativeOnly = Data(:,:,:,Labels ~= 1);

%for softmax
trainDataRatio = 0.8;
trainLength = floor(trainDataRatio*length(positiveOnly));
testLength = 2*(length(positiveOnly) - trainLength);

images.data = positiveOnly(:,:,:,1:trainLength);
images.set = ones(trainLength,1);
images.labels = ones(trainLength,1);
trainData = cat(4,positiveOnly(:,:,:,trainLength+1:end),negativeOnly(:,:,:,1:(length(positiveOnly) - trainLength)));
trainLabels = [ones(testLength/2,1);2*ones(testLength/2,1);];
testPerm = randperm(length(trainLabels));
trainData = trainData(:,:,:,testPerm);
trainLabels = trainLabels(testPerm);

images.data = cat(4,images.data,trainData);
images.set = [images.set; 3*ones(testLength,1)];
images.labels = [images.labels;trainLabels];

meanImgVis = mean(images.data(:,:,1,:),4);
meanImgIr = mean(images.data(:,:,2,:),4);
meanImg = (meanImgVis + meanImgIr)/2;
meta.meanImg = meanImg;
meta.meanIrImg = meanImgIr;
meta.meanVisImg = meanImgVis;

save(fullfile('Vis-Nir_Train.mat'),'images','meta','-v7.3');
