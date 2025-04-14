@author M.Kentour 

% R2024b v. Parallel PID Time Controller for ANN development

% 1. Create a new Simulink Model
new_system('ParallelANNController');
open_system('ParallelANNController');

% 2. Neural Network Subsystems (Parallel)
numANNs = 2; % Number of parallel ANNs (CPU and GPU)

for i = 1:numANNs
    add_block('simulink/Commonly Used Blocks/Subsystem', ['ParallelANNController/ANN' num2str(i)]);
    open_system(['ParallelANNController/ANN' num2str(i)]);

    % Simulate ANN delay
    add_block('simulink/Continuous/Transfer Fcn', ['ParallelANNController/ANN' num2str(i) '/TimeDelay']);
    if i == 1 % CPU
        set_param(['ParallelANNController/ANN' num2str(i) '/TimeDelay'], 'Numerator', '[1]');
        set_param(['ParallelANNController/ANN' num2str(i) '/TimeDelay'], 'Denominator', '[0.1 1]');
    else % GPU
        set_param(['ParallelANNController/ANN' num2str(i) '/TimeDelay'], 'Numerator', '[1]');
        set_param(['ParallelANNController/ANN' num2str(i) '/TimeDelay'], 'Denominator', '[0.05 1]');
    end
    add_block('simulink/Sinks/Out1', ['ParallelANNController/ANN' num2str(i) '/ANNOut']);
    add_block('simulink/Sources/In1', ['ParallelANNController/ANN' num2str(i) '/ANNIn']);
    add_line(['ParallelANNController/ANN' num2str(i)], 'ANNIn/1', 'TimeDelay/1');
    add_line(['ParallelANNController/ANN' num2str(i)], 'TimeDelay/1', 'ANNOut/1');

    close_system(['ParallelANNController/ANN' num2str(i)], false);
end

% 3. Target Time Inputs
add_block('simulink/Sources/Constant', 'ParallelANNController/Target1');
set_param('ParallelANNController/Target1', 'Value', '1'); % Target time for ANN1 (CPU)
add_block('simulink/Sources/Constant', 'ParallelANNController/Target2');
set_param('ParallelANNController/Target2', 'Value', '0.5'); % Target time for ANN2 (GPU)

% 4. PID Controllers (Parallel)
for i = 1:numANNs
    add_block('simulink/Continuous/PID Controller', ['ParallelANNController/PID' num2str(i)]);
    set_param(['ParallelANNController/PID' num2str(i)], 'P', '1');
    set_param(['ParallelANNController/PID' num2str(i)], 'I', '0.1');
    set_param(['ParallelANNController/PID' num2str(i)], 'D', '0.01');
end

% 5. Error Calculation (Parallel)
for i = 1:numANNs
    add_block('simulink/Math Operations/Subtract', ['ParallelANNController/Error' num2str(i)]);
end

% 6. Connections (Parallel)
for i = 1:numANNs
    add_line('ParallelANNController', ['Target' num2str(i) '/1'], ['Error' num2str(i) '/1']);
    add_line('ParallelANNController', ['ANN' num2str(i) '/ANNOut/1'], ['Error' num2str(i) '/2']);
    add_line('ParallelANNController', ['Error' num2str(i) '/1'], ['PID' num2str(i) '/1']);
    add_line('ParallelANNController', ['PID' num2str(i) '/1'], ['ANN' num2str(i) '/ANNIn/1']);
end

% 7. Simulation Parameters
set_param('ParallelANNController', 'StopTime', '10');

% 8. Run Simulation
sim('ParallelANNController');

% 9. Plot Results (Optional)
for i = 1:numANNs
    time(i,:) = simout.get(['ParallelANNController/ANN' num2str(i) '/ANNOut']).Values.Data;
    error(i,:) = simout.get(['ParallelANNController/Error' num2str(i)]).Values.Data;
end

figure;
for i = 1:numANNs
    subplot(numANNs,1,i);
    plot(time(i,:));
    title(['ANN' num2str(i) ' Time']);
end

figure;
for i = 1:numANNs
    subplot(numANNs,1,i);
    plot(error(i,:));
    title(['ANN' num2str(i) ' Error']);
end
