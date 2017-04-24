clear all;
clc;
[y,Fs] = audioread('done.wav');
tr = Trainer;
ir = ImageReader;
cl = Classifier;

load('trainHOG_8x8_Cells.mat');
%load('train_4x4_Blocks.mat');
%load('train_8x8_Blocks.mat');

%train
%{
[dataClasses, imagePaths2D] = ir.read('dataset');
tic; %start stopwatch
%[trainedSetHOG, trainedSetClassesHOG] = tr.TrainHOG(dataClasses, imagePaths2D, 1, [8 8]);
[trainedSet, trainedSetClasses] = tr.Train(dataClasses, imagePaths2D, 1, [4 4]);
elapsedTrainingTimeMinutes = toc/60;
sound(y,Fs);
%}

%test

testObjectsHOG = cl.getImgReadyHOG('test.png', 1, [8 8]);
tic; %start stopwatch
for i=1:numel(testObjectsHOG(:,1))
	classTypeHOG = cl.weightedKNN(trainedSetHOG, trainedSetClassesHOG, testObjectsHOG(i,:), 3, 0)
end
elapsedClassificationTime = toc;


%{
testObjects = cl.getImgReady('test.png', 1, [4 4]);
tic; %start stopwatch
for i=1:numel(testObjects(:,1))
	classType = cl.weightedKNN(trainedSet, trainedSetClasses, testObjects(i,:), 3, 0)
end
elapsedClassificationTime = toc;
%}
