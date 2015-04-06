clc;
clear;

%%%
%%% Iteration through each of the datasets with different parameter variations
%%%
datasets = {'adultData-unaggregated-preprocessed.csv', 'adultData-aggregated-preprocessed.csv', 'adultData-PCA-preprocessed.csv'}; 

trainingSizePercentage = [0.60 0.65 0.70];
validationSizePercentage = [0.15 0.15 0.15];

startingLearningRate = 0.15;
lrStepSize = 0.15;
maxLearningRate = 0.31;
learningRate = startingLearningRate;

startingMomentum = 0.0;
momentumStepSize = 0.30; 
maxMomentum = 0.9;
momentum = startingMomentum;

startingNumberEpochs = 300;
epochStepSize = 400;
maxNumberEpochs = 700;
numberEpochs = startingNumberEpochs;

startingHiddenNeurons = 20;
maxHiddenNeurons = 40;
hiddenNeuronStepSize = 20;
hiddenNeurons = startingHiddenNeurons;

startingReularization = 0.1;
maxReularization = 0.9;
regularizationStepSize = 0.4;
regularization = startingReularization;

total_iterations = size(datasets,2)*size(trainingSizePercentage,2)*(((maxHiddenNeurons + hiddenNeuronStepSize - startingHiddenNeurons)/hiddenNeuronStepSize))*(((maxNumberEpochs + epochStepSize - numberEpochs)/epochStepSize))*(((maxMomentum + momentumStepSize - momentum)/momentumStepSize))*(((maxLearningRate + lrStepSize - learningRate)/lrStepSize))*(((maxReularization + regularizationStepSize - regularization)/regularizationStepSize));
iteration_number = 0;

% Will build up statistics of each model produced by different parameter variation defined above
statistics_headers = {'model_id' 'hidden_neurons' 'trainPct' 'regularization' 'learning_rate' 'momentum' 'numberEpochs' 'mseTrain' 'mseValidation' 'mseTest' 'train_accuracy' 'val_accuracy' 'test_accuracy' 'TP' 'TN' 'FP' 'FN' 'accuracy' 'kappa' 'sensitivity' 'specificity' 'precision' 'time'};        
statistics = zeros(1,size(statistics_headers,2));

for dataset_index=1:size(datasets,2)
    
    x = csvread(char(datasets(1,dataset_index)),1);
    t = x(:,end);
    x = x(:,1:(end-1));
    inputs = x';
    targets = t';
    numb_datapoints = size(inputs,2);
    total_instances = size(targets,2);
    
    for trainingSizePercentage_index=1:size(trainingSizePercentage,2)
        while ((maxHiddenNeurons - hiddenNeurons)  > -0.000001)
            while ( (maxLearningRate - learningRate)  > -0.000001)
                 while ((maxMomentum - momentum)  > -0.000001)
                    while ((maxNumberEpochs - numberEpochs) > -0.000001)
                        while ((maxReularization - regularization) > -0.000001)
                            tic;
                            % Create a Fitting Network
                            hiddenLayerSize = hiddenNeurons;
                            net = fitnet(hiddenLayerSize);

                            % Choose Input and Output Pre/Post-Processing Functions
                            % For a list of all processing functions type: help nnprocess
                            net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
                            net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

                            % Setup Division of Data for Training, Validation, Testing
                            % For a list of all data division functions type: help nndivide
                            %net.divideFcn = 'dividerand';  % Divide data randomly
                            %net.divideMode = 'sample';  % Divide up every sample
                            %net.divideParam.trainRatio = 65/100;
                            %net.divideParam.valRatio = 15/100;
                            %net.divideParam.testRatio = 20/100;
                            
                            %define the indices to split train,val,test based on percentage parameters 
                            trainPct = trainingSizePercentage(1,trainingSizePercentage_index);
                            trainIndexEnd = floor(trainPct*numb_datapoints);
                            validationPct = validationSizePercentage(1,trainingSizePercentage_index);
                            validationIndexStart = trainIndexEnd + 1;
                            validationIndexEnd  = validationIndexStart + floor(validationPct*numb_datapoints);
                            testIndexStart = validationIndexEnd + 1;
                            
                            %divide up train,val,test sets
                            net.divideFcn = 'divideind';
                            net.divideParam.trainInd = 1:trainIndexEnd;
                            net.divideParam.valInd = validationIndexStart:validationIndexEnd;
                            net.divideParam.testInd = testIndexStart:numb_datapoints;
                            

                            % For help on training function 'trainlm' type: help trainlm
                            % For a list of all training functions type: help nntrain
                            net.trainFcn = 'traingdm';  % Levenberg-Marquardt

                            % set parameters for this model variation
                            net.trainParam.epochs = numberEpochs;
                            net.trainParam.lr = learningRate;
                            net.trainParam.showWindow = 1;
                            net.trainParam.max_fail = 10;
                            net.trainParam.mc = momentum;

                            % Choose a Performance Function
                            % For a list of all performance functions type: help nnperformance
                            net.performFcn = 'msereg';  % Regularised Mean squared error
                            net.performParam.regularization = regularization;

                            % Choose Plot Functions
                            % For a list of all plot functions type: help nnplot
                            %net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                            %  'plotregression', 'plotfit'};

                            % Train the Network
                            [net,tr] = train(net,inputs,targets);

                            % Test the Network
                            outputs = net(inputs);
                            errors = gsubtract(targets,outputs);
                            performance = perform(net,targets,outputs);
                            time = toc;
                            % Recalculate Training, Validation and Test Performance
                            trainTargets = targets .* tr.trainMask{1};
                            valTargets = targets  .* tr.valMask{1};
                            testTargets = targets  .* tr.testMask{1};
                            trainPerformance = perform(net,trainTargets,outputs);
                            valPerformance = perform(net,valTargets,outputs);
                            testPerformance = perform(net,testTargets,outputs);

                            % View the Network
                            %view(net);

                            % Plots
                            % Uncomment these lines to enable various plots.
                            %figure, plotperform(tr);
                            %figure, plottrainstate(tr);
                            %figure, plotfit(net,inputs,targets)
                            %figure, plotregression(targets,outputs)
                            %figure, ploterrhist(errors)
                            %figure, plotconfusion(targets,outputs);
                            %figure, plotroc(targets,outputs);
                            
                            %confusion matrx train,val,test
                            [c,cm,ind,per] = confusion(trainTargets,outputs);
                            train_TP = cm(2,2);
                            train_TN = cm(1,1);
                            train_FP = cm(1,2);
                            train_FN = cm(2,1);
                            train_accuracy = (train_TP + train_TN)/(train_TP + train_TN + train_FP + train_FN);
                            
                            [c,cm,ind,per] = confusion(valTargets,outputs);
                            val_TP = cm(2,2);
                            val_TN = cm(1,1);
                            val_FP = cm(1,2);
                            val_FN = cm(2,1);
                            val_accuracy = (val_TP + val_TN)/(val_TP + val_TN + val_FP + val_FN);
                            
                            [c,cm,ind,per] = confusion(testTargets,outputs);
                            test_TP = cm(2,2);
                            test_TN = cm(1,1);
                            test_FP = cm(1,2);
                            test_FN = cm(2,1);
                            test_accuracy = (test_TP + test_TN)/(test_TP + test_TN + test_FP + test_FN);

                            %confusion matrix for all instances
                            [c,cm,ind,per] = confusion(targets,outputs);
                            

                            %for neural network classifier
                            classifier_TP = cm(2,2);
                            classifier_TN = cm(1,1);
                            classifier_FP = cm(1,2);
                            classifier_FN = cm(2,1);
                            classifier_accuracy = (classifier_TP + classifier_TN)/(classifier_TP + classifier_TN + classifier_FP + classifier_FN);
                            observed_aggreement = classifier_TP + classifier_TN;

                            %for random classifier
                            p = 0.5;
                            numb_pos_instances = sum(targets==1);
                            numb_neg_instances = sum(targets==0);
                            TP = p*numb_pos_instances;
                            FP = (1-p)*numb_pos_instances;
                            TN = p*numb_neg_instances;
                            FN = (1-p)*numb_neg_instances;
                            random_accuracy = (TP + TN)/(TP + TN + FP + FN);

                            %alternative random classifier calculation
                            expected_pos_agreement = ((classifier_TP + classifier_FP)/total_instances) * (numb_pos_instances/total_instances);
                            expected_neg_agreement = ((classifier_TN + classifier_FN)/total_instances) * (numb_neg_instances/total_instances);
                            expected_accuracy = expected_pos_agreement + expected_neg_agreement;

                            %calculate kappa statistic (using 0.5 chance)
                            kappa = (classifier_accuracy - random_accuracy) / (1 - random_accuracy);

                            %other statistics
                            sensitivity = classifier_TP / (classifier_TP + classifier_FN);
                            specificity = classifier_TN / (classifier_TN + classifier_FP);
                            precision = classifier_TP / (classifier_TP + classifier_FP);

                            %SAVE ALL THE STATISTICS AND PARAMETER VALUES
                            statistics = [statistics; (iteration_number+1) hiddenNeurons trainPct regularization learningRate momentum numberEpochs trainPerformance valPerformance testPerformance train_accuracy val_accuracy test_accuracy classifier_TP classifier_TN classifier_FP classifier_FN classifier_accuracy kappa sensitivity specificity precision time];
                            
                            
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
                        
                            regularization = regularization + regularizationStepSize;
                        end
                        
                        
                        regularization = startingReularization;
                        numberEpochs = numberEpochs + epochStepSize;
                    end %number of epochs
                    
                    regularization = startingReularization;
                    numberEpochs = startingNumberEpochs;
                    momentum = momentum + momentumStepSize;
                 end %momentum
                
                 regularization = startingReularization;
                 numberEpochs = startingNumberEpochs;
                 momentum = startingMomentum;
                 learningRate = learningRate + lrStepSize;
            end %learning rate
            
            regularization = startingReularization;
            numberEpochs = startingNumberEpochs;
            momentum = startingMomentum;
            learningRate = startingLearningRate;
            hiddenNeurons = hiddenNeurons + hiddenNeuronStepSize;
        end %hidden neuron loop
        
        regularization = startingReularization;
        numberEpochs = startingNumberEpochs;
        momentum = startingMomentum;
        learningRate = startingLearningRate;
        hiddenNeurons = startingHiddenNeurons;
    end
    
    regularization = startingReularization;
    numberEpochs = startingNumberEpochs;
    momentum = startingMomentum;
    learningRate = startingLearningRate;
    hiddenNeurons = startingHiddenNeurons;
    
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
    
    statistics = zeros(1,size(statistics_headers,2));
    
end


