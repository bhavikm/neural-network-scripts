# neural-network-scripts
Two scripts are provided that build upon the generated scripts from Matlab for Self-Organizing Map (SOM) learning and Supervised Neural Networks.  

The scripts allow you to easily learn and test different Neural Network model variations for your given pre-processed datasets (assumes rows in dataset are randomized and target). Just define the names of the datasets in the top of the script and the model parameter variations you wish to step through. An output CSV of results for each model variation will be generated, including performance statistics.   

Example pre-processed datasets are provided (as well the original dataset) that the scripts were run on. Details of the pre-processing and a full report I wrote (with a partner) at Uni based on the example datasets and scripts provided can be found at: http://bhavikm.org/pdfs/neural-networks-assignment.pdf  

SOM
===
SOM_neural_network.m

"A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map." See more at: http://en.wikipedia.org/wiki/Self-organizing_map

 Output performance statistics:
   * True Positive
   * True Negative
   * False Positive
   * False Negative
   * Accuracy
   * Kappa
   * Sensitivity
   * Specificity
   * Precision
   * Learning time

Multi-Layer Perceptron Neural Network
================
supervised_neural_network.m

"A multilayer perceptron (MLP) is a feedforward artificial neural network model that maps sets of input data onto a set of appropriate outputs. A MLP consists of multiple layers of nodes in a directed graph, with each layer fully connected to the next one. Except for the input nodes, each node is a neuron (or processing element) with a nonlinear activation function." See more at: http://en.wikipedia.org/wiki/Multilayer_perceptron

 Output performance statistics:
   * Training/Test/Validation Accuracy
   * True Positive
   * True Negative
   * False Positive
   * False Negative
   * Accuracy
   * Kappa
   * Sensitivity
   * Specificity
   * Precision
   * Learning time

