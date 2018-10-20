clear
rng('default');
rng(1);

listDir = dir('data');
for fileIdx = 3:length(listDir)
    filePath = fullfile('data',listDir(fileIdx).name);
    if strcmp(listDir(fileIdx).name(1:2), 'F2')
        newFilePath = fullfile('data',strcat('f',listDir(fileIdx).name(3:end)));
        movefile(filePath,newFilePath);
    end
    if strcmp(listDir(fileIdx).name(1:2),'M2')
        newFilePath = fullfile('data',strcat('m',listDir(fileIdx).name(3:end)));
        movefile(filePath,newFilePath);
    end
end

trainSplit = [0.7, 0.1, 0.2];
cuhkTrain = 'cuhkTrain.csv';
cuhkTest = 'cuhkTest.csv';
trainPosOnly = 1;

listDir = dir('data');
listDir(1:2) = [];
imgIdxs = 1:2:length(listDir);
imgIdxs = imgIdxs(randperm(length(imgIdxs)));


nTrainImgs = round(length(imgIdxs)*trainSplit(1));
nValImgs = round(length(imgIdxs)*trainSplit(2));
nTestImgs = length(imgIdxs) - nTrainImgs - nValImgs;

'Train data'
trainImgs = imgIdxs(1:nTrainImgs);
'Validation data'
valImgs = imgIdxs((nTrainImgs+1):(nTrainImgs+nValImgs));
'Test data'
testImgs = imgIdxs((nTrainImgs+nValImgs+1):end);

csvData = {'type','rgb','rgb_x','rgb_y','nir','nir_x','nir_y'};

[trainData, trainLabels, trainCsvData] = GetPatches( listDir, trainImgs, 64 );
[valData, valLabels, valCsvData] = GetPatches( listDir, valImgs, 64 );
[testData, testLabels, testCsvData] = GetPatches( listDir, testImgs, 64 );

cell2csv(cuhkTrain,[csvData;trainCsvData;valCsvData]);
cell2csv(cuhkTest,[csvData;testCsvData]);

if trainPosOnly == 1
    trainData(:,:,:,trainLabels == 2) = [];
    trainLabels(trainLabels == 2) = [];
end

%for softmax
images.data = cat(4,trainData,valData);
images.set = [ones(size(trainData,4),1); 3*ones(length(valLabels),1)];
images.labels = [ones(size(trainData,4),1);valLabels];

meanImgVis = mean(images.data(:,:,1,:),4);
meanImgIr = mean(images.data(:,:,2,:),4);
meanImg = (meanImgVis + meanImgIr)/2;
meta.meanImg = meanImg;
meta.meanIrImg = meanImgIr;
meta.meanVisImg = meanImgVis;

save(fullfile('cuhk_Train.mat'),'images','meta','-v7.3');
save(fullfile('cuhk_Test.mat'),'testData','testLabels','meta','-v7.3');


