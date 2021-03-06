classdef EM < handle
    %{
===========================================================================
Auhtor: Girish Joshi
Date:   07/06/2018
Code:   Class File for Expectation-Maximization Clustering.
        % For Ellipsod Plotting the following Code is used
        % https://www.mathworks.com/matlabcentral/fileexchange/16543-plot-gaussian-ellipsoid    
===========================================================================
    %}
    properties (Access = 'private')
        cluster_count = [];
        cluster_data = [];
        cluster_mean = [];
        cluster_mean_prev = [];
        cluster_sigma = [];
        cluster_n = [];
        uniform_prior = [];
        panel = [];
        cluster_plot = [];
        cluster_mean_plot = [];
        cluster_ellipse = [];
        color = [];
        shape = [];
    end
    
    methods (Access = 'public')
        
        function obj = EM(cluster_count,cluster_data)
            obj.cluster_count = cluster_count;
            obj.cluster_data = cluster_data;
            obj.cluster_n = size(obj.cluster_data);
            obj.cluster_mean = 6*rand(obj.cluster_count,obj.cluster_n(1,2))-3;
            obj.cluster_mean_prev = zeros(obj.cluster_count,obj.cluster_n(1,2));
            for i = 1:cluster_count
                obj.cluster_sigma(:,:,i) = 2*rand*eye(obj.cluster_n(1,2))-1;
            end
            % The Code uses uniform Prior on the clusters
            obj.uniform_prior = 1/cluster_count;
            % Initialize the Plotting of the Data
            obj.initSim;
        end
        
        function cluster_probs = Expectation(obj)
            
            % Generate Cluster Probabilities
            likelihood_prob = obj.Likelihood();
            
            % Evaluate the Posterior Cluster Probability
            for j = 1:obj.cluster_n(1)
                evidence = 0;
                for i=1:obj.cluster_count
                    evidence = evidence + likelihood_prob(j,i).*obj.uniform_prior;
                end
                for i=1:obj.cluster_count
                    cluster_probs(j,i) = (likelihood_prob(j,i).*obj.uniform_prior)./evidence;
                end
            end
        end
        
        function error = Maximization(obj,cluster_probs)
            % Generate Update Cluster Mean and Variance
            Updated_mean = cluster_probs'*obj.cluster_data;
            for i=1:obj.cluster_count
                obj.cluster_mean(i,:) = Updated_mean(i,:)./(sum(cluster_probs(:,i)));
                clusters{i} = [];
            end
            % Classify the Data points depending on their higher Likelihood
            for i = 1:obj.cluster_n(1)
                [~, arg_max] = max(cluster_probs(i,:));
                clusters{arg_max}(i,:) = obj.cluster_data(i,:);
            end
            
            % Update the Covariance Matrix
            for i=1:obj.cluster_count
                if ~isempty(clusters{i})
                    obj.cluster_sigma(:,:,i) = cov(clusters{i}-obj.cluster_mean(i,:));
                end
            end
            error = norm(obj.cluster_mean-obj.cluster_mean_prev);
            obj.cluster_mean_prev = obj.cluster_mean;
            obj.plotClusters(clusters,obj.cluster_mean,obj.cluster_sigma)
            
        end
    end
    methods (Access = 'private')
        % Initialize the data plotting
        function initSim(obj)
            
            obj.panel = figure;
            obj.panel.Position = [100 100 600 600];
            obj.panel.Color = [1 1 1];
            hold on
            obj.color = {'r','b','k','g','y',[.5 .6 .7],[.8 .2 .6]}; % Cell array of colros.
            obj.shape = {'o','+','x','*','d','s','p'};
            for ii=1:obj.cluster_count
                obj.cluster_plot{ii} = scatter(0,0,50,obj.color{ii},obj.shape{ii});
                obj.cluster_mean_plot{ii} = scatter(0,0,100,obj.color{ii},'filled');
                obj.cluster_ellipse{ii} = plot_gaussian_ellipsoid(obj.cluster_mean(ii,:), 3*obj.cluster_sigma(:,:,ii),obj.color{ii});
            end
            axPlot = obj.cluster_plot{1}.Parent;
            axPlot.XTick = [];
            axPlot.YTick = [];
            axPlot.Visible = 'off';
            axPlot.Position = [0.35 0.4 0.3 0.3];
            axPlot.Clipping = 'off';
            axis equal
            axis([-1 3 0 5]);
            hold off
        end
        % Update the Plot
        function plotClusters(obj,C, mean, sigma)
            
            for ii =1:obj.cluster_count
                delete(obj.cluster_ellipse{ii})
                if ~isempty(C{ii})
                    set(obj.cluster_plot{ii},'XData',C{ii}(:,1));
                    set(obj.cluster_plot{ii},'YData',C{ii}(:,2));
                end
                set(obj.cluster_mean_plot{ii},'XData',mean(ii,1))
                set(obj.cluster_mean_plot{ii},'YData',mean(ii,2))
                obj.cluster_ellipse{ii} = plot_gaussian_ellipsoid(mean(ii,:), 3*sigma(:,:,ii));
                pause(0.1)
                
            end
            drawnow;
            
        end
        % Calculate the Liklihood of the data point belonging to cluster C: P(x_i|C)
        function prob = Likelihood(obj)
            
            for j = 1:obj.cluster_count
                for i=1:obj.cluster_n(1,1)
                    prob(i,j) = obj.Normal_pdf(obj.cluster_data(i,:),obj.cluster_mean(j,:),obj.cluster_sigma(:,:,j));
                end
            end
            
        end
        % Gaussian Likelihood
        function prob = Normal_pdf(obj,x,mu,sigma)
            prob = (1/sqrt(2*pi*det(sigma)))*exp(-0.5*(x-mu)*(inv(sigma))*(x-mu)');
        end
        
    end
end

