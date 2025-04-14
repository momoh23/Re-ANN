@ author M.Kentour R2024b v.

% 1. Load the MNIST Dataset
[trainImages, trainLabels] = digitTrain4DArrayData;
[testImages, testLabels] = digitTest4DArrayData;

% 2. Basic Information
disp('---------------------- MNIST Dataset Information ----------------------');
fprintf('  Training Images Size: %d x %d x %d x %d (Height x Width x Channels x Number)\n', size(trainImages));
fprintf('  Number of Training Labels: %d\n', numel(trainLabels));
fprintf('  Training Label Categories: ');
disp(categories(trainLabels));
fprintf('  Test Images Size: %d x %d x %d x %d\n', size(testImages));
fprintf('  Number of Test Labels: %d\n', numel(testLabels));
fprintf('  Test Label Categories: ');
disp(categories(testLabels));
fprintf('------------------------------------------------------------------\n');

% 3. Convert to Double and Normalize to [0, 1]
trainImagesNormalized = double(trainImages) / 255;
testImagesNormalized = double(testImages) / 255;

disp('------------------- Normalized MNIST Data Info -------------------');
fprintf('  Normalized Training Images Data Type: %s\n', class(trainImagesNormalized));
fprintf('  Normalized Intensity Range: [%.2f, %.2f]\n', min(trainImagesNormalized(:)), max(trainImagesNormalized(:)));
fprintf('  Normalized Test Images Data Type: %s\n', class(testImagesNormalized));
fprintf('  Normalized Test Images Intensity Range: [%.2f, %.2f]\n', min(testImagesNormalized(:)), max(testImagesNormalized(:)));
fprintf('-------------------------------------------------------------\n');

% 4. Visualize Original vs. Normalized Images (Sample)
numSamplesToCompare = 5;
rng(42); % For reproducibility
randomIndices = randperm(size(trainImages, 4), numSamplesToCompare);

figure('Name', 'Original vs. Normalized MNIST Images (Sample)', 'NumberTitle', 'off');
tiledlayout(numSamplesToCompare, 2, 'TileSpacing', 'Compact', 'Padding', 'Compact');

for i = 1:numSamplesToCompare
    nexttile;
    imshow(trainImages(:,:,:,randomIndices(i)));
    colormap gray;
    title(['Original ', num2str(i)]);
    axis off;

    nexttile;
    imshow(trainImagesNormalized(:,:,:,randomIndices(i)));
    colormap gray;
    title(['Normalized ', num2str(i)]);
    axis off;
end

% 5. Interactive Sample Visualization
numSlices = size(trainImagesNormalized, 4);
figure('Name', 'Interactive MNIST Sample Viewer', 'NumberTitle', 'off');
sliceAxes = subplot(1, 1, 1);
colormap(sliceAxes, gray); % Set colormap to grayscale
sliceSlider = uicontrol('Style', 'slider', ...
    'Min', 1, 'Max', numSlices, 'Value', 1, ...
    'SliderStep', [1/(numSlices-1), min(1, 10/(numSlices-1))], ...
    'Position', [20 20 560 20], ...
    'Callback', @(es, ed) updateSlice(trainImagesNormalized, trainLabels, numSlices, es.Value, sliceAxes)); % Pass sliceAxes

    function updateSlice(imageData, labelData, nSlices, slice, ax) % Receive sliceAxes as 'ax'
        sliceIndex = round(slice);
        if sliceIndex < 1
            sliceIndex = 1;
        elseif sliceIndex > nSlices
            sliceIndex = nSlices;
        end
        imshow(imageData(:,:,:,sliceIndex), 'Parent', ax); % Use the passed axes handle 'ax'
        title(ax, sprintf('Training Image %d - Label: %s', sliceIndex, char(labelData(sliceIndex))));
    end

% Initial display
updateSlice(trainImagesNormalized, trainLabels, numSlices, 1, sliceAxes);

% 6. Distribution of Training Labels
figure('Name', 'Distribution of Training Labels', 'NumberTitle', 'off');
histogram(trainLabels);
title('Distribution of Training Labels');
xlabel('Digit');
ylabel('Frequency');

% 7. Visualization of Average Digit Images
averageDigitImages = zeros(28, 28, 1, 10);
for i = 0:9
    digitIndices = find(trainLabels == string(i));
    digitImages = trainImagesNormalized(:,:,:,digitIndices);
    averageImage = mean(digitImages, 4);
    averageDigitImages(:,:,:,i+1) = averageImage;
end

figure('Name', 'Average Digit Images', 'NumberTitle', 'off');
montage(averageDigitImages, 'DisplayRange', [0 1]);
title('Average Image for Each Digit (0-9)');
colormap(gray);

% 8. Preprocessed Data (Ready for Model)
trainImagesPreprocessed = trainImagesNormalized; % Only normalization
testImagesPreprocessed = testImagesNormalized;

disp('---------------------- Preprocessed Data Info ----------------------');
fprintf('  Preprocessed Training Images Size: %d x %d x %d x %d\n', size(trainImagesPreprocessed));
fprintf('  Preprocessed Training Images Data Type: %s\n', class(trainImagesPreprocessed));
fprintf('  Preprocessed Test Images Size: %d x %d x %d x %d\n', size(testImagesPreprocessed));
fprintf('  Preprocessed Test Images Data Type: %s\n', class(testImagesPreprocessed));
fprintf('------------------------------------------------------------------\n')
