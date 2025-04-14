@author M.Kentour R2024b v.

% 1. Load MNIST Dataset
[trainImages, trainLabels] = digitTrain4DArrayData;
[testImages, testLabels] = digitTest4DArrayData;
% 2. Define Different ANN Architectures
architectures = {
    % Architecture 1
    [
        imageInputLayer([28 28 1], 'Name', 'input1')
        fullyConnectedLayer(50, 'Name', 'fc1_1')
        reluLayer('Name', 'relu1_1')
        fullyConnectedLayer(10, 'Name', 'fc2_1')
        softmaxLayer('Name', 'softmax1')
        classificationLayer('Name', 'classoutput1')
    ],
    % Architecture 2
    [
        imageInputLayer([28 28 1], 'Name', 'input2')
        fullyConnectedLayer(100, 'Name', 'fc1_2')
        reluLayer('Name', 'relu1_2')
        fullyConnectedLayer(10, 'Name', 'fc2_2')
        softmaxLayer('Name', 'softmax2')
        classificationLayer('Name', 'classoutput2')
    ],
    % Architecture 3
    [
        imageInputLayer([28 28 1], 'Name', 'input3')
        fullyConnectedLayer(100, 'Name', 'fc1_3')
        reluLayer('Name', 'relu1_3')
        fullyConnectedLayer(50, 'Name', 'fc2_3')
        reluLayer('Name', 'relu2_3')
        fullyConnectedLayer(10, 'Name', 'fc3_3')
        softmaxLayer('Name', 'softmax3')
        classificationLayer('Name', 'classoutput3')
    ],
    % Architecture 4
    [
        imageInputLayer([28 28 1], 'Name', 'input4')
        fullyConnectedLayer(75, 'Name', 'fc1_4')
        reluLayer('Name', 'relu1_4')
        fullyConnectedLayer(25, 'Name', 'fc2_4')
        reluLayer('Name', 'relu2_4')
        fullyConnectedLayer(10, 'Name', 'fc3_4')
        softmaxLayer('Name', 'softmax4')
        classificationLayer('Name', 'classoutput4')
    ],
    % Architecture 5
    [
        imageInputLayer([28 28 1], 'Name', 'input5')
        fullyConnectedLayer(128, 'Name', 'fc1_5')
        reluLayer('Name', 'relu1_5')
        fullyConnectedLayer(64, 'Name', 'fc2_5')
        reluLayer('Name', 'relu2_5')
        fullyConnectedLayer(10, 'Name', 'fc3_5')
        softmaxLayer('Name', 'softmax5')
        classificationLayer('Name', 'classoutput5')
    ],
    % Architecture 6
    [
        imageInputLayer([28 28 1], 'Name', 'input6')
        fullyConnectedLayer(64, 'Name', 'fc1_6')
        reluLayer('Name', 'relu1_6')
        fullyConnectedLayer(32, 'Name', 'fc2_6')
        reluLayer('Name', 'relu2_6')
        fullyConnectedLayer(16, 'Name', 'fc3_6')
        reluLayer('Name', 'relu3_6')
        fullyConnectedLayer(10, 'Name', 'fc4_6')
        softmaxLayer('Name', 'softmax6')
        classificationLayer('Name', 'classoutput6')
    ]
};
architectureNames = {'ANN 1 (50)', 'ANN 2 (100)', 'ANN 3 (100-50)', 'ANN 4 (75-25)', 'ANN 5 (128-64)', 'ANN 6 (64-32-16)'};
numArchitectures = numel(architectures);
numEpochs = 10;
miniBatchSize = 128;
% 3. Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', {testImages, testLabels}, ...
    'ValidationFrequency', ceil(size(trainImages, 4) / miniBatchSize) + 1, ... % Set to slightly more than iterations per epoch
    'Verbose', false, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'gpu'); % Use GPU/CPU if available, make selection
% 4. Performance Monitoring
allFinalAccuracies = zeros(1, numArchitectures);
allTrainingTimes = zeros(1, numArchitectures);
trainedNetworks = cell(1, numArchitectures);
allRecalls = zeros(10, numArchitectures); % 10 classes, num architectures
allPrecisions = zeros(10, numArchitectures);
allF1Scores = zeros(10, numArchitectures);
% 5. Parallel Training and Evaluation with Target Time Control
tic;
targetTime = 60;          % Desired target time in seconds.  Adjust as needed.
timeMargin = 10;            % Allowable deviation from the target time (e.g., +/- 10 seconds)
for i = 1:numArchitectures
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
    for class = 1:10
        allRecalls(class, i) = confusionMatrix(class, class) / sum(confusionMatrix(class, :));
        allPrecisions(class, i) = confusionMatrix(class, class) / sum(confusionMatrix(:, class));
        allF1Scores(class, i) = 2 * (allPrecisions(class, i) * allRecalls(class, i)) / (allPrecisions(class, i) + allRecalls(class, i));
    end
end
totalParallelTime = toc;
% 6. Visualize Performance Variation (Recall, Precision, F1-score)
figure;
h = []; % Handle for legend
for i = 1:numArchitectures
    subplot(2, 3, i); % Grid of 2x3 subplots
    h(1) = plot(1:10, allRecalls(:, i), 'r-');
    hold on;
    h(2) = plot(1:10, allPrecisions(:, i), 'g-');
    h(3) = plot(1:10, allF1Scores(:, i), 'b-');
    hold off;
    title(architectureNames{i});
    xlabel('Digit Classes');
    ylabel('Score');
    grid on;
    if i == 1
        legend(h, {'Recall', 'Precision', 'F1-Score'}, 'Location', 'best');
    end
end
sgtitle('Recall, Precision, and F1-Score per ANN (MLP) Architecture');
% 7. Display Results
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
% 8. Target Time Check and Feedback
if totalParallelTime <= targetTime + timeMargin && totalParallelTime >= targetTime - timeMargin
    disp(['SUCCESS: Total parallel training time (' num2str(totalParallelTime) 's) is within the target time of ' num2str(targetTime) 's (margin of +/- ' num2str(timeMargin) 's).']);
elseif totalParallelTime < targetTime - timeMargin
    disp(['WARNING: Total parallel training time (' num2str(totalParallelTime) 's) is significantly below the target time of ' num2str(targetTime) 's.  Consider increasing numEpochs or miniBatchSize for longer training.']);
else
    disp(['FAILURE: Total parallel training time (' num2str(totalParallelTime) 's) exceeds the target time of ' num2str(targetTime) 's (margin of +/- ' num2str(timeMargin) 's).  Consider decreasing numEpochs or miniBatchSize for faster training.']);
end
