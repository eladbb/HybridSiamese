clear
rng('default');
rng(1);

trainSplit = [0.7, 0.1, 0.2];
nirsceneTrain = 'Vis-Nir_grid_Train.csv';
nirsceneTest = 'Vis-Nir_grid_Test.csv';
trainPosOnly = 1;

dataPath = '..\Vis-Nir\data';

sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'};


overallTrainData = [];
overallTrainLabels = [];
overallTrainCsvData = [];
overallValData = [];
overallValLabels = [];
overallValCsvData = [];
overallTestData = [];
overallTestLabels = [];
overallTestCsvData = [];

for seq = sequences
    disp(seq)
    seqDirPath = fullfile(dataPath,seq);
    listDir = dir(fullfile(seqDirPath{1},'*.tiff'));
    for imgIdx = 1:length(listDir)
        listDir(imgIdx).name = fullfile(seq{1}, listDir(imgIdx).name);
    end
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
    
    [valData, valLabels, valCsvData] = GetPatches(dataPath, listDir, valImgs, 64 );
    [trainData, trainLabels, trainCsvData] = GetPatches(dataPath, listDir, trainImgs, 64 );
    [testData, testLabels, testCsvData] = GetPatches(dataPath, listDir, testImgs, 64 );
    
    if strcmp(seq{1}, 'country')
        overallTrainData = trainData;
        overallTrainLabels = trainLabels;
        overallTrainCsvData = trainCsvData;
        overallValData = valData;
        overallValLabels = valLabels;
        overallValCsvData = valCsvData;
        overallTestData = testData;
        overallTestLabels = testLabels;
        overallTestCsvData = testCsvData;
    else
        overallTrainData = cat(4,overallTrainData ,trainData);
        overallTrainLabels = [overallTrainLabels ;trainLabels];
        overallTrainCsvData = [ overallTrainCsvData;trainCsvData];  
        
        overallValData = cat(4,overallValData ,valData);
        overallValLabels = [overallValLabels ;valLabels];
        overallValCsvData = [ overallValCsvData;valCsvData];
        
        overallTestData = cat(4,overallTestData ,testData);
        overallTestLabels = [overallTestLabels ;testLabels];
        overallTestCsvData = [ overallTestCsvData;testCsvData];
    end
    
end


csvData = {'type','rgb','rgb_x','rgb_y','nir','nir_x','nir_y'};

cell2csv(nirsceneTrain,[csvData;overallTrainCsvData;overallValCsvData]);
cell2csv(nirsceneTest,[csvData;overallTestCsvData]);

if trainPosOnly == 1
    overallTrainData(:,:,:,overallTrainLabels == 2) = [];
    overallTrainLabels(overallTrainLabels == 2) = [];
end

%for softmax
images.data = cat(4,overallTrainData,overallValData);
images.set = [ones(size(overallTrainData,4),1); 3*ones(length(overallValLabels),1)];
images.labels = [ones(size(overallTrainData,4),1);overallValLabels];

meanImgVis = mean(images.data(:,:,1,:),4);
meanImgIr = mean(images.data(:,:,2,:),4);
meanImg = (meanImgVis + meanImgIr)/2;
meta.meanImg = meanImg;
meta.meanIrImg = meanImgIr;
meta.meanVisImg = meanImgVis;

save(fullfile('Vis-Nir_grid_Train.mat'),'images','meta','-v7.3');
testData = overallTestData;
testLabels = overallTestLabels;
save(fullfile('Vis-Nir_grid_Test.mat'),'testData','testLabels','meta','-v7.3');

