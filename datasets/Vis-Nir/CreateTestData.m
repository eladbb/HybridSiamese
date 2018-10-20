clear
sequences = {'field'};
% sequences = {'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'};

for sequence = sequences
    filename = fullfile('data','csv',strcat(sequence{1},'.csv'));
    keypointsData = table2cell(readtable(filename));
    imagesNames = unique(char(keypointsData(:,2)),'rows');
    testData = zeros(64,64,2,length(keypointsData),'uint8');
    testLabels = zeros(1,length(keypointsData));
    disp(sequence{1})
    for imgIdx = 1:length(imagesNames)
        disp(strcat(num2str(imgIdx),'/', num2str(length(imagesNames))));
        visImgPath = fullfile('data',sequence{1},strrep(imagesNames(imgIdx,:), 'ppm', 'tiff'));
        irImgPath = fullfile('data',sequence{1},strrep(imagesNames(imgIdx,:), 'rgb.ppm', 'nir.tiff'));
        [ numPatches,startIdx,endIdx ] = GetImgNumPatches( keypointsData,imagesNames(imgIdx,:) );
        imgKeypointsData = keypointsData(startIdx:endIdx,:);
        [imgData, imgLabels] = CropImagePatches(visImgPath, irImgPath, imgKeypointsData);
        testData(:,:,:,startIdx:endIdx) = imgData;
        testLabels(startIdx:endIdx) = imgLabels;
    end
    for idx = 1:size(testData,4)
        if sum(sum(sum(testData(:,:,:,idx)))) == 0
            idx
        end
    end
    save(strcat(sequence{1},'_Test.mat'),'testData','testLabels','-v7.3');
    clear('testData');
end


