% @ author M.Kentour R2024b v.

% 1. Load MNIST Dataset
[trainImages, trainLabels] = digitTrain4DArrayData;
[testImages, testLabels] = digitTest4DArrayData;

% 2. Network Architecture
layers = [
    imageInputLayer([28 28 1])
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% 3. Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'ValidationData', {testImages, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % Use GPU if available

% 4. User Input for Time Limits
cpuTimeLimit = input('Enter the target CPU time per epoch (seconds): ');
gpuTimeLimit = input('Enter the target GPU time per epoch (seconds): ');

% 5. Train the Network and Measure Performance with CPU/GPU Monitoring
trainingTime = 0;
cpuTimeEpoch = [];
gpuMemoryUsedEpoch = []; % Changed to GPU memory usage
gpuPercentUsedEpoch = [];

for epoch = 1:options.MaxEpochs
    epochTic = tic;
    [net, info] = trainNetwork(trainImages, trainLabels, layers, options);
    epochTime = toc(epochTic);

    if exist('gpuDevice', 'file') && gpuDeviceCount > 0
        if epochTime > gpuTimeLimit
            disp(['GPU time limit exceeded in epoch ', num2str(epoch), '. Training stopped.']);
            trainingTime = trainingTime + gpuTimeLimit;
            cpuTimeEpoch = [cpuTimeEpoch, gpuTimeLimit];
            gpuMemoryUsedEpoch = [gpuMemoryUsedEpoch, NaN];
            gpuPercentUsedEpoch = [gpuPercentUsedEpoch, NaN];
            break; % Stop training
        else
            trainingTime = trainingTime + epochTime;
            cpuTimeEpoch = [cpuTimeEpoch, epochTime];
            try
                gpuUtil = gpuDevice(1);
                gpuMemoryUsedEpoch = [gpuMemoryUsedEpoch, gpuUtil.MemoryUsed];
                
                % use nvidia-smi to get the gpu percentage used.
                [status, result] = system('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'); %CUDA install
                if status == 0
                    gpuUtilizations = str2double(strsplit(result));
                    gpuPercentUsedEpoch = [gpuPercentUsedEpoch, gpuUtilizations(1)];
                else
                    gpuPercentUsedEpoch = [gpuPercentUsedEpoch, NaN]; % Store NaN if nvidia-smi fails.
                end
            catch ME
                disp(['Error measuring GPU utilization in epoch ', num2str(epoch), ': ', ME.message]);
                gpuMemoryUsedEpoch = [gpuMemoryUsedEpoch, NaN];
                gpuPercentUsedEpoch = [gpuPercentUsedEpoch, NaN];
            end
        end
    else
        if epochTime > cpuTimeLimit
            disp(['CPU time limit exceeded in epoch ', num2str(epoch), '. Training stopped.']);
            trainingTime = trainingTime + cpuTimeLimit;
            cpuTimeEpoch = [cpuTimeEpoch, cpuTimeLimit];
            gpuMemoryUsedEpoch = [gpuMemoryUsedEpoch, NaN];
            gpuPercentUsedEpoch = [gpuPercentUsedEpoch, NaN];
            break; % Stop training
        else
            trainingTime = trainingTime + epochTime;
            cpuTimeEpoch = [cpuTimeEpoch, epochTime];
            gpuMemoryUsedEpoch = [gpuMemoryUsedEpoch, NaN];
            gpuPercentUsedEpoch = [gpuPercentUsedEpoch, NaN];
        end
    end
end

% 6. Test the Network
predictedLabels = classify(net, testImages);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);

% 7. Visualize Re-ANN-Sequential performance

% 8. Visualize Loss and Accuracy
figure;
subplot(2, 1, 1);
plot(info.TrainingLoss, 'b-', 'LineWidth', 2);
hold on;
plot(info.ValidationLoss, 'r-', 'LineWidth', 2);
hold off;
title('Training and Validation Loss');
xlabel('Iteration');
ylabel('Loss');
legend('Training Loss', 'Validation Loss');

subplot(2, 1, 2);
plot(info.TrainingAccuracy, 'b-', 'LineWidth', 2);
hold on;
plot(info.ValidationAccuracy, 'r-', 'LineWidth', 2);
hold off;
title('Training and Validation Accuracy');
xlabel('Iteration');
ylabel('Accuracy');
legend('Training Accuracy', 'Validation Accuracy');

% 9. Visualize CPU/GPU Usage per Epoch
figure;
if exist('gpuDevice', 'file') && gpuDeviceCount > 0
    subplot(3, 1, 1);
    plot(1:length(gpuMemoryUsedEpoch), gpuMemoryUsedEpoch, 'b-', 'LineWidth', 2);
    title('GPU Memory Used per Epoch (Bytes)');
    xlabel('Epoch');
    ylabel('Memory (Bytes)');

    subplot(3, 1, 2);
    plot(1:length(gpuPercentUsedEpoch), gpuPercentUsedEpoch, 'g-', 'LineWidth', 2);
    title('GPU Percent Utilization per Epoch (%)');
    xlabel('Epoch');
    ylabel('Utilization (%)');

    subplot(3, 1, 3);
    plot(1:length(cpuTimeEpoch), cpuTimeEpoch, 'r-', 'LineWidth', 2);
    title('CPU Time per Epoch (seconds)');
    xlabel('Epoch');
    ylabel('Time (s)');
else
    plot(1:length(cpuTimeEpoch), cpuTimeEpoch, 'r-', 'LineWidth', 2);
    title('CPU Time per Epoch (seconds)');
    xlabel('Epoch');
    ylabel('Time (s)');
end

% 10. Display Results
disp(['Accuracy: ', num2str(accuracy * 100), '%']);
disp(['Total Training Time: ', num2str(trainingTime), ' seconds']);