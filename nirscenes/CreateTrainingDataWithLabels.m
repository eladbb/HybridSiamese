clear
augment = 0;

sequence = 'country';
filename = fullfile('csv',strcat(sequence,'.csv'));
keypointsData = table2cell(readtable(filename));
imagesNames = unique(char(keypointsData(:,2)),'rows');
trainData = [];
trainLabels = [];
for imgIdx = 1:length(imagesNames)
    imgPatchesDataPath = fullfile(sequence,strrep(imagesNames(imgIdx,:), 'ppm', 'mat'));
    load(imgPatchesDataPath);
    numSamples = size(imgBenchmarkData,4);
%     currentImgPathces = impyramid(single(imgBenchmarkData),'reduce');
    currentImgPathces = single(imgBenchmarkData);
    trainData = cat(4,trainData,currentImgPathces);
    imgBenchmarkDataLabels(imgBenchmarkDataLabels == 0) = 2;
    trainLabels = cat(1,trainLabels,imgBenchmarkDataLabels);
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
testData = cat(4,positiveOnly(:,:,:,trainLength+1:end),negativeOnly(:,:,:,1:(length(positiveOnly) - trainLength)));
testLabels = [ones(testLength/2,1);2*ones(testLength/2,1);];
testPerm = randperm(length(testLabels));
testData = testData(:,:,:,testPerm); 
testLabels = testLabels(testPerm);

images.data = cat(4,images.data,testData);
images.set = [images.set; 3*ones(testLength,1)];
images.labels = [images.labels;testLabels];

meanImgVis = mean(images.data(:,:,1,:),4);
meanImgIr = mean(images.data(:,:,2,:),4);
meanImg = (meanImgVis + meanImgIr)/2;
meta.meanImg = meanImg;
meta.meanIrImg = meanImgIr;
meta.meanVisImg = meanImgVis;

save(fullfile('country64x64PositiveOnly_0.8.mat'),'images','meta','-v7.3');

% for l2
trainDataRatio = 0.95;
trainLength = floor(trainDataRatio*length(positiveOnly));
testLength = 2*(length(positiveOnly) - trainLength);

images.data = positiveOnly(:,:,:,1:trainLength);
images.set = ones(trainLength,1);
images.labels = ones(trainLength,1);
testData = cat(4,positiveOnly(:,:,:,trainLength+1:end),negativeOnly(:,:,:,1:(length(positiveOnly) - trainLength)));
testLabels = [ones(testLength/2,1);-1*ones(testLength/2,1);];
testPerm = randperm(length(testLabels));
testData = testData(:,:,:,testPerm); 
testLabels = testLabels(testPerm);

images.data = cat(4,images.data,testData);
images.set = [images.set; 3*ones(testLength,1)];
images.labels = [images.labels;testLabels];

meanImgVis = mean(images.data(:,:,1,:),4);
meanImgIr = mean(images.data(:,:,2,:),4);
meanImg = (meanImgVis + meanImgIr)/2;
meta.meanImg = meanImg;
meta.meanIrImg = meanImgIr;
meta.meanVisImg = meanImgVis;

save(fullfile('country64x64PositiveOnly_0.95.mat'),'images','meta','-v7.3');
