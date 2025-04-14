 
% @ author M.Kentour R2024b v.

% 1. Load the MRI Dataset
load('mri.mat'); % Loads D, map

% 2. Basic Information Display
disp('---------------------- MRI Dataset Information ----------------------');
mriSize = size(D);
fprintf('  MRI Data Size: %d x %d x %d (Width x Height x Slices)\n', mriSize(1), mriSize(2), mriSize(3));
numSlices = mriSize(3);
fprintf('  Number of Slices (Samples): %d\n', numSlices);
dataType = class(D);
fprintf('  Data Type: %s\n', dataType);
minIntensity = min(D(:));
maxIntensity = max(D(:));
fprintf('  Intensity Range: [%.2f, %.2f]\n', minIntensity, maxIntensity);
fprintf('-----------------------------------------------------------------\n');

% 3. Preprocessing: Normalization
normalizedD = (D - minIntensity) / (maxIntensity - minIntensity);
disp('------------------- Normalized MRI Data Info -------------------');
fprintf('  Normalized Data Type: %s\n', class(normalizedD));
fprintf('  Normalized Intensity Range: [%.2f, %.2f]\n', min(normalizedD(:)), max(normalizedD(:)));
fprintf('-------------------------------------------------------------\n');

% 4. Interactive Slice Visualization with Colormap
figure('Name', 'Interactive MRI Slice Viewer (Colormap)', 'NumberTitle', 'off');
sliceAxesOriginalColored = subplot(1, 2, 1);
sliceAxesNormalizedColored = subplot(1, 2, 2);
sliceSliderColored = uicontrol('Style', 'slider', ...
    'Min', 1, 'Max', numSlices, 'Value', floor(numSlices / 2), ...
    'SliderStep', [min(1, 1/(max(1, numSlices-1))), min(1, 10/(max(1, numSlices-1)))], ...
    'Position', [20 20 560 20], ...
    'Callback', @(es, ed) updateSliceColored(D, normalizedD, map, es.Value, sliceAxesOriginalColored, sliceAxesNormalizedColored)); % Pass map and axes handles
    function updateSliceColored(originalData, normalizedData, colorMap, slice, axOriginal, axNormalized) % Receive map and axes handles
        sliceIndex = round(slice);
        if sliceIndex < 1
            sliceIndex = 1;
        elseif sliceIndex > numSlices
            sliceIndex = numSlices;
        end
        imshow(originalData(:,:,sliceIndex), colorMap, 'Parent', axOriginal); % Use map for original
        title(axOriginal, sprintf('Original MRI Slice %d (Colormap)', sliceIndex));
        imshow(normalizedData(:,:,sliceIndex), colorMap, 'Parent', axNormalized); % Use map for normalized
        title(axNormalized, sprintf('Normalized MRI Slice %d (Colormap)', sliceIndex));
        colormap(axOriginal, colorMap); % Apply colormap to the original axes
        colormap(axNormalized, colorMap); % Apply colormap to the normalized axes
    end
% Initial display with colormap
updateSliceColored(D, normalizedD, map, floor(numSlices / 2), sliceAxesOriginalColored, sliceAxesNormalizedColored); % Pass map and axes handles

% 5. Visualize Intensity Frequency of the First Slice (Original vs. Normalized)
firstSliceIndex = 1; % Directly use the first slice
figure('Name', 'Intensity Frequency Comparison (First Slice)', 'NumberTitle', 'off');
% For the original first slice
originalSlice = D(:,:,firstSliceIndex);
subplot(1, 2, 1);
histogram(originalSlice(:), 'DisplayName', 'Original'); % Linearize the slice
title(sprintf('Intensity Frequency - Original Slice %d', firstSliceIndex));
xlabel('Intensity Value');
ylabel('Frequency');
legend('show');
% For the normalized first slice
normalizedSlice = normalizedD(:,:,firstSliceIndex);
subplot(1, 2, 2);
histogram(normalizedSlice(:), 'DisplayName', 'Normalized'); % Linearize the slice
title(sprintf('Intensity Frequency - Normalized Slice %d', firstSliceIndex));
xlabel('Normalized Intensity (0-1)');
ylabel('Frequency');
legend('show');
% 6. Visualize Intensity Frequency of Slice 4 (Original vs. Normalized)
slice4Index = 4;
if slice4Index <= numSlices
    figure('Name', 'Intensity Frequency Comparison (Slice 4)', 'NumberTitle', 'off');
    % For the original slice 4
    originalSlice4 = D(:,:,slice4Index);
    subplot(1, 2, 1);
    histogram(originalSlice4(:), 'DisplayName', 'Original'); % Linearize the slice
    title(sprintf('Intensity Frequency - Original Slice %d', slice4Index));
    xlabel('Intensity Value');
    ylabel('Frequency');
    legend('show');
    % For the normalized slice 4
    normalizedSlice4 = normalizedD(:,:,slice4Index);
    subplot(1, 2, 2);
    histogram(normalizedSlice4(:), 'DisplayName', 'Normalized'); % Linearize the slice
    title(sprintf('Intensity Frequency - Normalized Slice %d', slice4Index));
    xlabel('Normalized Intensity (0-1)');
    ylabel('Frequency');
    legend('show');
else
    fprintf('Warning: Slice 4 does not exist as the total number of slices is %d.\n', numSlices);
end
% 7. Color Map Information and Visualization
disp('---------------------- Color Map Information ----------------------');
mapSz = size(map);
fprintf('  Map Size: %d x %d (Rows x Columns)\n', mapSz(1), mapSz(2));
fprintf('  First 5 Color Map Values:\n');
disp(map(1:min(5, mapSz(1)), :));
% Visualize the color map (shows how normalized intensities would be colored)
figure('Name', 'Color Map Visualization', 'NumberTitle', 'off');
subplot(1, 2, 1);
imshow(linspace(0, 1, size(map, 1))', map);
title('Color Map Visualization (Gradient)');
ylabel('Index (Normalized Intensity 0-1)');
xlabel('Color');
subplot(1, 2, 2);
plot(map);
title('Color Map Visualization (RGB Components)');
xlabel('Index (Normalized Intensity 0-1)');
ylabel('RGB Value');
fprintf('-----------------------------------------------------------------\n');
% 8. Graymap Handling and Visualization (Keep this for comparison)
disp('---------------------- GrayMap Information ----------------------');
if exist('graymap', 'var')
    graymapSz = size(graymap);
    fprintf('  GrayMap Size: %d x %d (Rows x Columns)\n', graymapSz(1), graymapSz(2));
    fprintf('  First 5 GrayMap Values:\n');
    disp(graymap(1:min(5, graymapSz(1)), :));
    % Visualize the graymap
    figure('Name', 'GrayMap Visualization', 'NumberTitle', 'off'); % Give it a name
    subplot(1, 2, 1);
    imshow(linspace(0, 1, size(graymap, 1))', graymap);
    title('GrayMap Visualization (Gradient)');
    ylabel('Index (Normalized Intensity 0-1)');
    xlabel('Gray Level');
    subplot(1, 2, 2);
    plot(graymap);
    title('GrayMap Visualization (RGB Components)');
    xlabel('Index (Normalized Intensity 0-1)');
    ylabel('RGB Value');
else
    disp('  GrayMap not found, creating a simple grayscale colormap.');
    graymap = gray(256); % Create a grayscale colormap
    graymapSz = size(graymap);
    fprintf('  Created GrayMap Size: %d x %d (Rows x Columns)\n', graymapSz(1), graymapSz(2));
    fprintf('  First 5 Created GrayMap Values:\n');
    disp(graymap(1:min(5, graymapSz(1)), :));
    % Visualize the created graymap
    figure('Name', 'Created GrayMap', 'NumberTitle', 'off'); % Give it a name
    subplot(1, 2, 1);
    imshow(linspace(0, 1, size(graymap, 1))', graymap);
    title('Created GrayMap (Gradient)');
    ylabel('Index (Normalized Intensity 0-1)');
    xlabel('Gray Level');
    subplot(1, 2, 2);
    plot(graymap);
    title('Created GrayMap Visualization (RGB Components)');
    xlabel('Index (Normalized Intensity 0-1)');
    ylabel('RGB Value');
end
fprintf('-----------------------------------------------------------------\n');
% 9. Parameter Exploration (Optional - Add more as needed)
disp('---------------------- Parameter Exploration ----------------------');
fprintf('  Number of Dimensions in MRI Data: %d\n', ndims(D));
fprintf('  Class of the Color Map Variable ("map"): %s\n', class(map));
if exist('graymap', 'var')
    fprintf('  Class of the Gray Map Variable ("graymap"): %s\n', class(graymap));
end
fprintf('-----------------------------------------------------------------\n');

% 10. Interactive Slice Visualization (Grayscale - for comparison)
figure('Name', 'MRI brain Slice Viewer (Grayscale)', 'NumberTitle', 'off');
sliceAxesOriginalGray = subplot(1, 2, 1);
sliceAxesNormalizedGray = subplot(1, 2, 2);
sliceSliderGray = uicontrol('Style', 'slider', ...
    'Min', 1, 'Max', numSlices, 'Value', floor(numSlices / 2), ...
    'SliderStep', [min(1, 1/(max(1, numSlices-1))), min(1, 10/(max(1, numSlices-1)))], ...
    'Position', [20 20 560 20], ...
    'Callback', @(es, ed) updateSliceGray(D, normalizedD, es.Value, sliceAxesOriginalGray, sliceAxesNormalizedGray)); % Pass axes handles
    function updateSliceGray(originalData, normalizedData, slice, axOriginal, axNormalized) % Receive axes handles
        sliceIndex = round(slice);
        if sliceIndex < 1
            sliceIndex = 1;
        elseif sliceIndex > numSlices
            sliceIndex = numSlices;
        end
        imshow(originalData(:,:,sliceIndex), [], 'Parent', axOriginal); % Default grayscale for original
        title(axOriginal, sprintf('Original MRI brain Slice %d (Grayscale)', sliceIndex));
        imshow(normalizedData(:,:,sliceIndex), [], 'Parent', axNormalized); % Default grayscale for normalized
        title(axNormalized, sprintf('Normalized MRI brain Slice %d (Grayscale)', sliceIndex));
        colormap(axOriginal, gray); % Ensure grayscale colormap
        colormap(axNormalized, gray); % Ensure grayscale colormap
    end
% Initial display (Grayscale)
updateSliceGray(D, normalizedD, floor(numSlices / 2), sliceAxesOriginalGray, sliceAxesNormalizedGray); % Pass axes handles