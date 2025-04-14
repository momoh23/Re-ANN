% @author M. R2024b 
% 1. Load MRI Dataset
load('mri.mat'); % Loads the mri data
% 2. Preprocess the Data
%  -  Verify the structure of 'mri'.  It should contain 'images' and 'labels'.
%  -  Normalize and reshape the images, and split into training and testing sets.
if ~isstruct(mri) || ~isfield(mri, 'images') || ~isfield(mri, 'labels')
    error('mri.mat must be a struct with fields "images" and "labels".');
end
images = mri.images;
labels = mri.labels;
% Check if the images are already in the correct format (256x256x1xN)
if ndims(images) ~= 4 || size(images, 1) ~= 256 || size(images, 2) ~= 256
    error('Images must be of size 256x256x1xN.');
end
% Normalize the images to the range [0, 1]
images = double(images) / 255;
% Split data into training and testing sets (80% training, 20% testing)
rng('default'); % For reproducibility
numImages = size(images, 4);
indices = randperm(numImages);
splitPoint = floor(0.8 * numImages);
trainImages = images(:,:,:,indices(1:splitPoint));
trainLabels = labels(indices(1:splitPoint));
testImages = images(:,:,:,indices(splitPoint+1:end));
testLabels = labels(indices(splitPoint+1:end));
% 3. Define Different CNN Architectures
architectures = {
    % CNN Architecture 1 (Simple)
    [
        imageInputLayer([256 256 1], 'Name', 'input1')
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1_1')
        reluLayer('Name', 'relu1_1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_1')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_1') % Use max(labels) for output size
        softmaxLayer('Name', 'softmax1')
        classificationLayer('Name', 'classoutput1')
    ],
    % CNN Architecture 2 (Slightly Deeper)
    [
        imageInputLayer([256 256 1], 'Name', 'input2')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1_2')
        reluLayer('Name', 'relu1_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_2')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2_2')
        reluLayer('Name', 'relu2_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_2')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_2')
        softmaxLayer('Name', 'softmax2')
        classificationLayer('Name', 'classoutput2')
    ],
    % CNN Architecture 3 (More Filters)
    [
        imageInputLayer([256 256 1], 'Name', 'input3')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_3')
        reluLayer('Name', 'relu1_3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_3')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_3')
        reluLayer('Name', 'relu2_3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_3')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_3')
        softmaxLayer('Name', 'softmax3')
        classificationLayer('Name', 'classoutput3')
    ],
    % CNN Architecture 4 (Different Kernel Size)
    [
        imageInputLayer([256 256 1], 'Name', 'input4')
        convolution2dLayer(5, 8, 'Padding', 'same', 'Name', 'conv1_4')
        reluLayer('Name', 'relu1_4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_4')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2_4')
        reluLayer('Name', 'relu2_4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_4')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_4')
        softmaxLayer('Name', 'softmax4')
        classificationLayer('Name', 'classoutput4')
    ],
    % CNN Architecture 5 (With Batch Normalization)
    [
        imageInputLayer([256 256 1], 'Name', 'input5')
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1_5')
        batchNormalizationLayer('Name', 'bn1_5')
        reluLayer('Name', 'relu1_5')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_5')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2_5')
        batchNormalizationLayer('Name', 'bn2_5')
        reluLayer('Name', 'relu2_5')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_5')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_5')
        softmaxLayer('Name', 'softmax5')
        classificationLayer('Name', 'classoutput5')
    ],
    % CNN Architecture 6 (Fewer Layers, More Filters Initially)
    [
        imageInputLayer([256 256 1], 'Name', 'input6')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_6')
        reluLayer('Name', 'relu1_6')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_6')
        fullyConnectedLayer(max(labels), 'Name', 'fc1_6')
        softmaxLayer('Name', 'softmax6')
        classificationLayer('Name', 'classoutput6')
    ]
};
architectureNames = {'CNN 1 (Simple)', 'CNN 2 (Deeper)', 'CNN 3 (More Filters)', 'CNN 4 (Diff Kernel)', 'CNN 5 (BN)', 'CNN 6 (Fewer Layers)'};
numArchitectures = numel(architectures);
numEpochs = 10;
miniBatchSize = 128;
% 4. Training Options (Simplified)
options = trainingOptions('adam', ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', {testImages, testLabels}, ...
    'ValidationFrequency', ceil(size(trainImages, 4) / miniBatchSize) + 1, ...
    'Verbose', false, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'cpu'); % Select GPU if available or force CPU
% 5. Performance Monitoring
allFinalAccuracies = zeros(1, numArchitectures);
allTrainingTimes = zeros(1, numArchitectures);
trainedNetworks = cell(1, numArchitectures);
numClasses = max(labels); % Determine the number of classes dynamically
allRecalls = zeros(numClasses, numArchitectures);
allPrecisions = zeros(numClasses, numArchitectures);
allF1Scores = zeros(numClasses, numArchitectures);
% 6. Parallel Training and Evaluation
tic;
parfor i = 1:numArchitectures
    currentLayers = architectures{i};
    disp(['Training ', architectureNames{i}, '...']);
    tic_start = tic;
    net = trainNetwork(trainImages, trainLabels, currentLayers, options);
    allTrainingTimes(i) = toc(tic_start);
    trainedNetworks{i} = net; % Store the trained network
    predictedLabels = classify(net, testImages);
    accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
    allFinalAccuracies(i) = accuracy * 100; % Store final accuracy
    % Calculate Recall, Precision, and F1-score
    confusionMatrix = confusionmat(testLabels, predictedLabels);
    for class = 1:numClasses
        allRecalls(class, i) = confusionMatrix(class, class) / sum(confusionMatrix(class, :));
        allPrecisions(class, i) = confusionMatrix(class, class) / sum(confusionMatrix(:, class));
        allF1Scores(class, i) = 2 * (allPrecisions(class, i) * allRecalls(class, i)) / (allPrecisions(class, i) + allRecalls(class, i));
    end
end
totalParallelTime = toc;
% 7. Visualize Performance Variation (Recall, Precision, F1-score)
figure;
h = []; % Handle for legend.
for i = 1:numArchitectures
    subplot(2, 3, i); % Grid of 2x3 subplots
    plot(1:numClasses, allRecalls(:, i), 'r-');
    hold on;
    plot(1:numClasses, allPrecisions(:, i), 'g-');
    plot(1:numClasses, allF1Scores(:, i), 'b-');
    hold off;
    title(architectureNames{i});
    xlabel('MRI Classes');
    ylabel('Score');
    grid on;
end
% Create legend in the last subplot.
subplot(2,3,6);
h = {'Recall', 'Precision', 'F1-Score'};
legend(h);
sgtitle('Recall, Precision, and F1-Score per CNN Architecture for MRI Data');
% 8. Visualize Accuracy
figure;
bar(categorical(architectureNames), allFinalAccuracies);
title('Final Validation Accuracy of Different CNN Architectures (Re-ANN Parallel Training)');
xlabel('CNN Architecture');
ylabel('Final Validation Accuracy (%)');
ylim([min(allFinalAccuracies) - 5, max(allFinalAccuracies) + 5]); % Adjust y-axis limits
grid on;
% 9. Display Results
disp(' ');
disp('Training Time per Architecture (seconds):');
for i = 1:numArchitectures
    disp([architectureNames{i}, ': ', num2str(allTrainingTimes(i))]);
end
disp(' ');
disp('Final Validation Accuracy per Architecture (%):');
for i = 1:numArchitectures
    disp([architectureNames{i}, ': ', num2str(allFinalAccuracies(i))]);
end
disp(' ');
disp(['Total Parallel Training Time (seconds): ', num2str(totalParallelTime)]);

% 10. Target Time Check
targetTime = 30; % Example target time in seconds
if totalParallelTime <= targetTime
    disp(['SUCCESS: Total parallel training time (' num2str(totalParallelTime) 's) is within the target time of ' num2str(targetTime) 's.']);
else
    disp(['FAILURE: Total parallel training time (' num2str(totalParallelTime) 's) exceeds the target time of ' num2str(targetTime) 's.']);
end