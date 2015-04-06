clear;
clc;

%%%
%%% Iteration through each of the datasets with different parameter variations
%%%
datasets = {'adultData-unaggregated-preprocessed.csv', 'adultData-aggregated-preprocessed.csv', 'adultData-PCA-preprocessed.csv'}; 
trainingSizePercentage = [0.75 0.85];
epochTrials = [200 700];
somDimensionTrials = [2 5 10];
learningRateTrials = [0.45 0.9];
neighbourhoodSizeTrials = [1 4 8];
topologyFunctions = {'gridtop','hextop','randtop'}; %SOM topology functions to try

total_iterations=size(datasets,2)*size(trainingSizePercentage,2)*size(epochTrials,2)*size(somDimensionTrials,2)*size(neighbourhoodSizeTrials,2)*size(topologyFunctions,2)*size(learningRateTrials,2);
iteration_number = 0;

% Will build up statistics of each model produced by different parameter variation defined above
statistics_headers = {'model_id' 'epochs' 'learningRate' 'dimensions' 'neighbourhoodSize' 'topologyFunc' 'TP' 'TN' 'FP' 'FN' 'accuracy' 'kappa' 'sensitivity' 'specificity' 'precision', 'time'};        
statistics = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

for dataset_index=1:size(datasets,2)
    
    x = csvread(char(datasets(1,dataset_index)),1);
    t = x(:,end);
    x = x(:,1:(end-1));
    inputs = x';
    targets = t';
    numb_datapoints = size(inputs,2);
    
    for trainingSizePercentage_index=1:size(trainingSizePercentage,2)
        
        trainPct = trainingSizePercentage(1,trainingSizePercentage_index);
        trainIndexEnd = floor(trainPct*numb_datapoints);
        testIndexStart = trainIndexEnd + 1;
        training_inputs = inputs(:,1:trainIndexEnd);
        training_targets = targets(:,1:trainIndexEnd);
        test_inputs = inputs(:,testIndexStart:end);
        test_targets = targets(:,testIndexStart:end);
        
        for epoch_index=1:size(epochTrials,2)
            
            for som_dimension_index=1:size(somDimensionTrials,2)
                som_dimension = somDimensionTrials(1,som_dimension_index);
                
                for neighbourhood_size_index=1:size(neighbourhoodSizeTrials,2)
                    ns = neighbourhoodSizeTrials(1,neighbourhood_size_index);
                    
                    if ns <= som_dimension
                     
                        for topology_index=1:size(topologyFunctions,2)
                            
                            for lr_index=1:size(learningRateTrials,2)
                            
                                tic;

                                coverSteps = 100;
                                topologyFcn = char(topologyFunctions(1,topology_index));
                                distanceFcn = 'linkdist';

                                %Set Parameters
                                net = selforgmap([som_dimension som_dimension], coverSteps, ns, topologyFcn, distanceFcn);
                                net.trainParam.epochs = epochTrials(1,epoch_index);
                                net.trainParam.lr = learningRateTrials(1,lr_index);
                                
                                [net,tr] = train(net,training_inputs);
                                outputs = net(training_inputs);
                                %classes = vec2ind(outputs);

                                training_neuron_classes = zeros(som_dimension*som_dimension, 2);
                                for i=1:size(outputs,2)
                                    winning_neuron_index = find(outputs(:,i) == 1);

                                    if training_targets(1,i) == 0
                                        training_neuron_classes(winning_neuron_index,1) = training_neuron_classes(winning_neuron_index,1) + 1;
                                    else
                                        training_neuron_classes(winning_neuron_index,2) = training_neuron_classes(winning_neuron_index,2) + 1;
                                    end
                                end

                                %decide on winner for each neuron based on class counts of that neuron
                                neuron_winners = zeros(size(training_neuron_classes,1),1);
                                for i=1:size(training_neuron_classes,1)
                                    if training_neuron_classes(i,1) < training_neuron_classes(i,2)
                                        neuron_winners(i,1) = 1;
                                    end
                                end

                                %apply test inputs to net
                                test_outputs = net(test_inputs);
                                %find where each test input got mapped to in neuron space, hence decide its class
                                test_predicted_targets = zeros(1,size(test_outputs,2));
                                for i=1:size(test_outputs,2)
                                    mapped_neuron_index = find(test_outputs(:,i) == 1);

                                    if neuron_winners(mapped_neuron_index,1) == 0
                                        test_predicted_targets(1,i) = 0; 
                                    else
                                        test_predicted_targets(1,i) = 1; 
                                    end
                                end

                                %compare to test targets
                                TP = 0;
                                TN = 0;
                                FP = 0;
                                FN = 0;

                                for j=1:size(test_predicted_targets,2)
                                    if (test_predicted_targets(1,j) == 0)
                                        if (test_targets(1,j) == 0)
                                            TN = TN + 1;
                                        else
                                            FN = FN + 1;
                                        end
                                    else
                                        if (test_targets(1,j) == 1)
                                            TP = TP + 1;
                                        else
                                            FP = FP + 1;
                                        end
                                    end
                                end

                                %statistics for this model
                                accuracy = (TP + TN)/(TP + TN + FP + FN);
                                kappa = (accuracy - 0.5) / 0.5;
                                sensitivity = TP / (TP + FN);
                                specificity = TN / (TN + FP);
                                precision = TP / (TP + FP);

                                elapsed = toc;

                                %SAVE ALL THE STATISTICS AND PARAMETER VALUES
                                statistics_headers = {'model_id' 'epochs' 'learningRate' 'dimensions' 'neighbourhoodSize' 'topologyFunc' 'TP' 'TN' 'FP' 'FN' 'accuracy' 'kappa' 'sensitivity' 'specificity' 'precision', 'time'};        

                                statistics = [statistics; (iteration_number+1) epochTrials(1,epoch_index) learningRateTrials(1,lr_index) som_dimension ns topology_index TP TN FP FN accuracy kappa sensitivity specificity precision elapsed];


                                %print out the iteration number
                                if total_iterations > 100
                                    %for more than 100 iterations, print out after every 1% of
                                    %iterations completed
                                    if ( mod(iteration_number,floor((total_iterations/100))) == 0 )
                                        disp([num2str(iteration_number),' iterations of ',num2str(total_iterations),' complete']);
                                    end
                                else
                                    disp([num2str(iteration_number),' iterations of ',num2str(total_iterations),' complete']);
                                end

                                %increment the iteration number
                                iteration_number = iteration_number + 1;

                            end
                        end
                    end
                    
                end
                
            end
        
        end
    end
    
    statistics = statistics(2:end,:); %remove initial row of all zeros

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% OUTPUT RESULTS TO CSV %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    filename = strcat(char(datasets(1,dataset_index)),'-RESULTS.csv');

    % check dimensions of data and header
    [drow dcol] = size(statistics);
    [hrow hcol] = size(statistics_headers);
    if hcol ~= dcol
        error( 'header not of same length as data (columns)' )
    end

    % open file
    outid = fopen(filename, 'w+');

    % write header
    for idx = 1:hcol
        fprintf(outid, '%s', statistics_headers{idx});
        if idx ~= hcol
            fprintf(outid, ',');
        else
            fprintf(outid, '\n' );
        end
    end

    % close file
    fclose(outid);

    % write data
    dlmwrite (filename, statistics, '-append' );
    
    statistics = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
end
