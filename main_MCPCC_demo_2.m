clc; clear;close all;
% Load sample database
load('chessboard.mat');

options.ct = lower ( 'mcpcc' );   % select method, "pcc, epcc, mcpcc"
options.sn = upper ( 'L2' );      % select norm type, "L1 or L2"
options.C = abs( 600 );             % penalty coefficient for svm margine training, "a positive scalar"
options.number_centers = int8( 85 );% number of centers," an integer above 1"
%(option for mcpcc not for pcc or epcc)

%calculate conic center as mean of positive samples
pos  =  (train_labels == 1);
options.centers =  mean(train_features(pos,:));

%calculate centers for mcpcc
if strcmp(options.ct,'mcpcc')
    if options.number_centers<1
        error('an integer above 1 is expected for number of centers')
    end
    
    % Find some centers in the database using k-means
    [~,centers_] = kmeans(train_features,(options.number_centers-1));
    
    % Add positive samples' center to the center list
    options.centers = [ options.centers; centers_ ];
end

% Train model
model = train_linear_polyhedral_classifier_k(train_features, train_labels, options);

% Test model
Xtest_k=arrange_sample_k(test_features, options);
[plabels,score] = predict(model,Xtest_k);
%averagePrecision = evaluateDetectionPrecision(score,test_labels);

% Dislay result
accuracy = sum(plabels==test_labels) / length(test_labels);
S = [ options.ct ' ' options.sn ' accuracy is ' num2str(accuracy) ];
disp ( S );

% Show visual plot
plot_graph(test_features,test_labels, model, options)

%% Functions used in this demo
function model_ = train_linear_polyhedral_classifier_k( X_, y_ , options)
%This function accomplishes training with data given and constructs a model
%of the classifier

%arrange data for training
algortihm  =  [ options.ct options.sn ] ;
mu_ = options.centers;
switch algortihm
    case 'pccL1',   X_ = [(X_-mu_) vecnorm( X_ - mu_, 1, 2 ) ];
    case 'pccL2',   X_ = [(X_-mu_) vecnorm( X_ - mu_, 2, 2 ) ];
    case 'epccL1',  X_ = [(X_-mu_) abs(  X_ - mu_) ];
    case 'epccL2'
        Xa = X_-mu_;
        X_ = [Xa (Xa.*Xa)];
    case {'mcpccL1','mcpccL2', 'mcpcc'}
        n = length(y_);
        k = size(mu_,1);
        Xa = zeros(k,n);
        for i = 1:k
            switch options.sn
                case 'L1', Xa(i,:)  =  vecnorm( X_-mu_(i,:),1,2);   %L1 norm
                case 'L2', Xa(i,:)  =  vecnorm( X_-mu_(i,:),2,2)  ;   %L2 Norm
            end
        end
        X_ = [(X_-mu_(i,:)) Xa' ];
    otherwise, error('Check classifier type');
        
end

%start training
model_  =  fitcsvm(X_,y_, 'BoxConstraint',options.C);
end

function XX = arrange_sample_k( X_, options)
%This function accomplishes arrangement of data before test prosedures and
%outputs new data

%arrange data for testing pcc,epcc,mcpcc
algortihm  =  [ options.ct options.sn] ;
mu_= options.centers;
switch algortihm
    case 'pccL1', XX = [(X_-mu_) vecnorm( X_ - mu_, 1, 2) ];
    case 'pccL2', XX = [(X_-mu_) vecnorm( X_ - mu_, 2, 2) ];
    case 'epccL1', XX = [(X_-mu_) abs( X_ - mu_) ];
    case 'epccL2'
        Xa = X_-mu_;
        XX = [Xa (Xa.*Xa)];
    case {'mcpccL1','mcpccL2', 'mcpcc'}
        n = size(X_,1);
        k = size(mu_,1);
        Xa = zeros(k,n);
        for i = 1:k
            switch options.sn
                case 'L1', Xa(i,:)  =  vecnorm( X_-mu_(i,:),1,2);   %L1 norm
                case 'L2', Xa(i,:)  =  vecnorm( X_-mu_(i,:),2,2);   %L2 Norm
            end
        end
        XX = [(X_-mu_(i,:))  Xa' ];
    otherwise, error('Check classifier type');
end
end

function plot_graph( Xtest_, ytest_, model,options)
%this function plots data, positive acceptance region, and centers

%prepare grid  area for plot
d = 0.02;
[x1Grid,x2Grid] = meshgrid( min(Xtest_(:,1)):d:max(Xtest_(:,1)),...
    min(Xtest_(:,2)):d:max(Xtest_(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

%arrange grid data for testing
xGrid_k = arrange_sample_k(xGrid,options);

%get scores of grid points
[~,score] = predict( model, xGrid_k );
%averagePrecision = evaluateDetectionPrecision(score,test_labels);

figure
hold on
%plot decision area
contourf(x1Grid,x2Grid,reshape(score(:,2),size(x1Grid)),[0 0],'k');

%show positive samples
ll=(ytest_==1);
plot(Xtest_(ll,1),Xtest_(ll,2),'.');
%show negative samples
plot(Xtest_(~ll,1),Xtest_(~ll,2),'.');

%show centers in mcpcc method
if strcmp(options.ct,'mcpcc')
    % show centers in mcpcc
    plot(options.centers(:,1),options.centers(:,2),'.k','MarkerSize',20);
end
end

