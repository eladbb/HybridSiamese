clear
sequences = {'country', 'field', 'forest', 'indoor', 'mountain', 'oldbuilding', 'street', 'urban', 'water'}

% -- For each sequence
for i=1:length(sequences)
    disp(['Processing ',sequences(i)]);
%   -- Read csv file
    filename = fullfile('csv',strcat(sequences(i),'.csv'));
    keypointsData = table2cell(readtable(filename{1}));
    numPatches = length(keypointsData);

    labels = all(char(keypointsData(:,1)) == repmat('positive',numPatches,1),2);
    trainingData = zeros(64,64,2,sum(labels),'single');
    imagesNames = unique(char(keypointsData(:,2)),'rows');
    numOfImages = length(imagesNames); 
    postiveSamplesIter = 1;
    for imgIdx = 1:numOfImages
        disp(['Processing ',sequences(i),'\',imagesNames(imgIdx,:)]);
        visImgPath = fullfile(sequences{i},strrep(imagesNames(imgIdx,:), 'ppm', 'tiff'));
        irImgPath = fullfile(sequences{i},strrep(imagesNames(imgIdx,:), 'rgb.ppm', 'nir.tiff'));
        [ numPatches,startIdx,endIdx ] = GetImgNumPatches( keypointsData,imagesNames(imgIdx,:) );
        visImg = rgb2gray(imread(visImgPath));
        irImg = imread(irImgPath);
        imgBenchmarkData = zeros(64,64,2,numPatches,'single');
        imgBenchmarkDataLabels = zeros(numPatches,1);
        for patchesPairIdx = startIdx:endIdx
            imgBenchmarkDataLabels(patchesPairIdx-startIdx+1) = all(char(keypointsData{patchesPairIdx,1}) == 'positive');
            rgb_y = keypointsData{patchesPairIdx,4};
            rgb_x = keypointsData{patchesPairIdx,3};
            nir_y = keypointsData{patchesPairIdx,7};
            nir_x = keypointsData{patchesPairIdx,6};
            imgBenchmarkData(:,:,1,patchesPairIdx-startIdx+1) = visImg(rgb_y-31:rgb_y+32,rgb_x-31:rgb_x+32);
            imgBenchmarkData(:,:,2,patchesPairIdx-startIdx+1) = irImg(nir_y-31:nir_y+32,nir_x-31:nir_x+32);
            if (imgBenchmarkDataLabels(patchesPairIdx-startIdx+1))
                trainingData(:,:,1,postiveSamplesIter) = imgBenchmarkData(:,:,1,patchesPairIdx-startIdx+1);
                trainingData(:,:,2,postiveSamplesIter) = imgBenchmarkData(:,:,2,patchesPairIdx-startIdx+1);
                postiveSamplesIter = postiveSamplesIter+1;
            end
%             subplot(1,2,1);imshow(imgBenchmarkData(:,:,1,patchesPairIdx-startIdx+1));
%             subplot(1,2,2);imshow(imgBenchmarkData(:,:,2,patchesPairIdx-startIdx+1));
        end 
        save(fullfile(sequences{i},strrep(imagesNames(imgIdx,:), 'ppm', 'mat')),'imgBenchmarkData','imgBenchmarkDataLabels','-v7.3'); 
    end
%     if( strcmp(sequences{i},'country'))
%         meanImg = zeros(64,64,1,'single');
%         meanIrImg = zeros(64,64,1,'single');
%         meanVisImg = zeros(64,64,1,'single');
%         for j = 1:length(trainingData)
%             %subplot(1,2,1);imshow(imagesGray(:,:,1,i)),subplot(1,2,2);imshow(imagesGray(:,:,2,i))
%             meanImg = meanImg + single(trainingData(:,:,1,j)) + single(trainingData(:,:,2,j));
%             meanVisImg = meanVisImg + single(trainingData(:,:,1,j));
%             meanIrImg = meanIrImg + single(trainingData(:,:,2,j));
%         end
%         meanImg = meanImg/(length(trainingData)*2);
%         meanVisImg = meanVisImg/length(trainingData);
%         meanIrImg = meanIrImg/length(trainingData);
% 
%         meta.meanImg = meanImg;
%         meta.meanVisImg = meanVisImg;
%         meta.meanIrImg = meanIrImg;
%         meta.sets = {'train', 'val', 'test'};
%         images.data = trainingData;
%         images.set = [ones(1,length(trainingData) - 15000) 3*ones(1,15000)];
%         save(strcat(sequences{i},'.mat'),'images','meta','-v7.3');
%     end
end
 