clc
clear all

%% Read the data from the csv file
csv_file='C:\Kiran\Semesters\AI Project\Project Code\data3.csv';
[x_raw, d_raw]=getdata1(csv_file);
[rows,cols]=size(x_raw);
%% Split the data into two parts
%Training part
%Validation part
[train_x_raw,train_d_raw,valid_x_raw,valid_d_raw]=divide1(x_raw,d_raw);

%% Set the initial state of random number
% so that the results will be the same when this script is executed multiple times.
rng('default')
rng(0)
d=200;
E=zeros(d,1);
A=zeros(d,1);
TPR=zeros(d,1);
PRECISON=zeros(d,1);
FPR=zeros(d,1);
OPT_Mean=zeros(d,1);


for i=1:d
%% MLP dynamics
% Maximum iterations
maxN=1000;
% Learning rate
eta=0.0001*i;
% Momentum
alpha=0.1;

%%
% R: # of instances
% I: # of units in the input layer
% J: # of units in the hidden layer
% K: # of units in the output layer
[R, I] = size(train_x_raw);
J=I+1;
K = size(train_d_raw,2);

% Matrix of weights between input layer and hidden layer
net.v_ij=rand(I+1,J);
% Matrix of weights between hidden layer and output layer
net.w_jk=rand(J+1,K);


%% Standardization of training data
% Standardization convert the raw data to dataset that has range of [0,1],
% Make sure you understand the usage of "repmat" function
%train_x_avg=mean(train_x_raw);
%train_x_shift = train_x_raw - repmat(train_x_avg, R, 1);

%train_x_std = std(train_x_shift);
%train_x = train_x_shift ./ repmat(train_x_std, R, 1);
train_x=train_x_raw;
% No need to standardize the expected output
train_d = train_d_raw;

E_r = zeros(maxN,1);

%%

for n=1:maxN
    r=randi(floor(0.3*R));
    x_r_i = [train_x(r,:),1]';
    a_r_j = net.v_ij' * x_r_i;
    % z_r_j_raw =a_r_j;
    z_r_j_raw =1./(1+(exp(-a_r_j))); %modified code sigmoid function
    
    z_r_j = [z_r_j_raw;1];
    a_r_k = net.w_jk' * z_r_j;
    % y_r = a_r_k;
    y_r = 1./(1+(exp(-a_r_k))); %modified code sigmoid function
    
    diff_out_r_n = train_d(r,:)'-y_r;
    
    E_r(n)=1/2*sqrt(sum(diff_out_r_n.^2));
    
    %fprintf('The %dth iteration, the %dth sample, squared error %f\n', ...
    %   n,r, E_r(n));
    
    DELTA_k = diff_out_r_n;
    diff_w_jk = eta*z_r_j*DELTA_k';
    
    delta_j = (net.w_jk(1:J,:)*DELTA_k);
    diff_v_ij = eta*x_r_i*delta_j';
    
    if n==1
        net.v_ij = net.v_ij + diff_v_ij;
        net.w_jk = net.w_jk + diff_w_jk;
    else
        net.v_ij = net.v_ij + (1-alpha)*diff_v_ij + alpha * prev_diff_v_ij;
        net.w_jk = net.w_jk + (1-alpha)*diff_w_jk + alpha * prev_diff_w_jk;
    end
    
    prev_diff_v_ij = diff_v_ij;
    prev_diff_w_jk = diff_w_jk;
end
%% Apply the trained NN model on the validation data to see how precise
% the model is.

[R, ~] = size(valid_x_raw);

%% Standardization of validation data using mean and std of training data
%Make sure you understand the usage of "repmat" function
%valid_x_shift = valid_x_raw - repmat(train_x_avg, R, 1);

%valid_x_std = std(valid_x_shift);
%valid_x = valid_x_shift ./ repmat(train_x_std, R, 1);
valid_x=valid_x_raw;
% No need to standardize the expected output
valid_d = valid_d_raw;

x = [valid_x, ones(R, 1)];
aj = x * net.v_ij;
%z_raw = aj;
z_raw =1./(1+(exp(-aj))); %modified code sigmoid function

z = [z_raw, ones(R, 1)];
ak = z * net.w_jk;
%y = ak;
y = 1./(1+(exp(-ak))); %modified code sigmoid function
limit=5;
Mean_y=meany(y,limit);
a=zeros(limit,1);
for k=1:limit
    y(y<Mean_y(k,1)) = 0;
    y(y>=Mean_y(k,1)) = 1;
    a(k,1)=sum(y==1)/R*100;
    if a(k,1)>=max(a)
        accuracy_rate=a(k,1);
        meany1=Mean_y(k,1);
    end
end
E(i,1)=eta;
A(i,1)=accuracy_rate;
OPT_Mean(i,1)=meany1;
%% confusion matrix
confusion_mat = [ valid_d, y ];

TP = sum( confusion_mat(:,1)==1 & confusion_mat(:,2)==1 );
TN = sum( confusion_mat(:,1)==0 & confusion_mat(:,2)==0 );
FN = sum( confusion_mat(:,1)==1 & confusion_mat(:,2)==0 );
FP = sum( confusion_mat(:,1)==0 & confusion_mat(:,2)==1 );

PRECISON(i,1)= 100 * TP/ (TP+FP);
TPR(i,1)= 100 *  TP/(TP+FN); %RECALL
FPR(i,1)= 100 *  FP/(TN+FP);
end

figure(2);
ax1 = subplot(2,2,1);
plot(ax1,E,A,'b-','LineWidth',0.1);
ylabel('Accuracy')
xlabel('Learning rate')
title('TEST1')
hold('on');
ax2 = subplot(2,2,2);
plot(ax2,OPT_Mean,A,'b-','LineWidth',0.1);
xlabel('Threshold')
ylabel('Accuracy')
title('TEST2');
hold('on');
for i=1:d
    if A(i,1)==max(A)
        Max_A=A(i,1);
        Opt_E=E(i,1);
        Opt_M=OPT_Mean(i,1);
        plot(ax1,Opt_E,Max_A,'r*');
        plot(ax2,Opt_M,Max_A,'r*');
        %break;
    end
end
fprintf('\nAccuracy : %f \tLearning Rate : %f \t Threshold: %f',Max_A,Opt_E,Opt_M);
ax3 = subplot(2,2,3);
plot(ax3,E,PRECISON,'b-','LineWidth',0.1);
ylabel('Precison/Positive Predicative value')
xlabel('Learning rate')
title('TEST3')
hold('on')
for i=1:d
    if PRECISON(i,1)==max(PRECISON)
        Max_P=PRECISON(i,1);
        Opt_E=E(i,1);
        Opt_M=OPT_Mean(i,1);
        plot(ax3,Opt_E,Max_P,'r*');
        fprintf('\nPPV/Precision : %f \tLearning Rate : %f \t  Threshold: %f',Max_P,Opt_E,Opt_M)
        %break;
    end
end
ax4 = subplot(2,2,4);
plot(ax4,E,TPR,'b-','LineWidth',0.1);
ylabel('True Positive Rate/Recall')
xlabel('Learning rate')
title('TEST4')
hold('on')
for i=1:d
    if TPR(i,1)==max(TPR)
        Max_TPR=TPR(i,1);
        Opt_E=E(i,1);
        Opt_M=OPT_Mean(i,1);
        plot(ax4,Opt_E,Max_TPR,'r*');
        fprintf('\nTPR/Recall : %f \t Learning Rate : %f \t Threshold: %f \n',Max_TPR,Opt_E,Opt_M)
        %break;
    end    
end 
