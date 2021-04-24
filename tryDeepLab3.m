clear all;
close all;
clc;

gTruth=open('gTruth.mat');
labels = open('labels.mat');
[imds,pxds] = pixelLabelTrainingData(gTruth.gTruth); 

[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');

imageSize = [416 416 3];
imgLayer = imageInputLayer(imageSize,"Name","input_1");

numClasses = 2;
network = 'resnet18';
lgraph = deeplabv3plusLayers(imageSize,numClasses,network,'DownsamplingFactor',16);
%lgraph = replaceLayer(lgraph,"input_1",[imgLayer dropoutLayer(0.5,"Name","drop1")]);
% imgLayer = imageInputLayer(imageSize,"Name","input_1");
% filterSize = 3;
% numFilters = 32;
% conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
% relu = reluLayer();
% poolSize = 2;
% maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2);
% downsamplingLayers = [
%     conv
%     relu
%     maxPoolDownsample2x
%     conv
%     relu
%     maxPoolDownsample2x
%     ];
% 
% filterSize = 4;
% transposedConvUpsample2x = transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
% upsamplingLayers = [
%     transposedConvUpsample2x
%     relu
%     transposedConvUpsample2x
%     relu
%     ];
% 
% conv1x1 = convolution2dLayer(1,numClasses);
% finalLayers = [
%     conv1x1
%     softmaxLayer()
%     pixelClassificationLayer()
%     ];
% 
% 
% 
% lgraph = [
%     imgLayer    
%     downsamplingLayers
%     upsamplingLayers
%     finalLayers
%     ];

        
         

options = trainingOptions('adam', ...
       'InitialLearnRate', 0.001, ...
       'Verbose',true, ...
       'validationData',[],...
       'validationFrequency',15,...
       'MiniBatchSize',2, ...
       'MaxEpochs',300, ...
       'Plots','training-progress', ...
       'Shuffle','every-epoch', ...
       'VerboseFrequency',10); 
   
cds = combine(imdsTrain,pxds);   
   
net = trainNetwork(cds,lgraph,options);