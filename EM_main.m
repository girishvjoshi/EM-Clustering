clc
clear all
%{
===========================================================================
Auhtor: Girish Joshi
Date:   07/06/2018
Code:   Expectation-Maximization Clustering
===========================================================================
%}
% Synthetic Data to Verify the Code

% Generate the data
% Cluster 1
true_mu1 = [2 3];
true_sigma1 = [1 1.5; 1.5 3];
rng default  % For reproducibility
R1 = mvnrnd(true_mu1,true_sigma1,100);
% Cluster 2
true_mu2 = [-3 0.5];
true_sigma2 = [1 0.1; 0.1 1];
rng default  % For reproducibility
R2 = mvnrnd(true_mu2,true_sigma2,100);
% Cluster 3
true_mu3 = [0 -3];
true_sigma3 = [0.5 0.1; 0.1 0.5];
rng default  % For reproducibility
R3 = mvnrnd(true_mu3,true_sigma3,100);

% Data Set
cluster_data = [R1;R2;R3]; % This Variable can be loaded with User Data

% Select number of centers you wish to classify the data into
Centers = 3;

% Create EM Object
classifier = EM(Centers,cluster_data);

%Initialize the Variables
step = 1;
mean_error = Inf;

% Rus the EM-Alorithm till the Error in Approximated mean
% of clusters Converges

while mean_error > 1e-4
    % Expectation Step
    cluster_prob = classifier.Expectation();
    % Maximization Step
    mean_error(step) = classifier.Maximization(cluster_prob);
    step = step+1;
end

% Plot the EM-Clustering convergence Performance
figure(2)
plot(mean_error,'LineWidth',2)
xlabel('Steps')
ylabel('||\mu_N - \mu_{N-1}||_2')
title('Convergence in Cluster Means')
grid on



