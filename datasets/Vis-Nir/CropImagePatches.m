function [ imgData, imgLabels ] = CropImagePatches( visImgPath, irImgPath,keypointsData )
%CROPIMAGEPATCHES Summary of this function goes here
%   Detailed explanation goes here
imgData = zeros(64,64,2,length(keypointsData),'uint8');
imgLabels = zeros(length(keypointsData),1);
visImg = rgb2gray(imread(visImgPath));
irImg = imread(irImgPath);

for patchesPairIdx = 1:length(keypointsData)
    if strcmp(keypointsData{patchesPairIdx,1},'positive')
        imgLabels(patchesPairIdx) = 1;
    else
        imgLabels(patchesPairIdx) = 2;
    end
    rgb_y = keypointsData{patchesPairIdx,4};
    rgb_x = keypointsData{patchesPairIdx,3};
    nir_y = keypointsData{patchesPairIdx,7};
    nir_x = keypointsData{patchesPairIdx,6};
    imgData(:,:,1,patchesPairIdx) = visImg(rgb_y-31:rgb_y+32,rgb_x-31:rgb_x+32);
    imgData(:,:,2,patchesPairIdx) = irImg(nir_y-31:nir_y+32,nir_x-31:nir_x+32);

    %subplot(1,2,1);imshow(imgBenchmarkData(:,:,1,patchesPairIdx-startIdx+1));
    %subplot(1,2,2);imshow(imgBenchmarkData(:,:,2,patchesPairIdx-startIdx+1));
end
end

