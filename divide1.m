function [train_x,train_d,valid_x,valid_d]=divide(data,label)
% Split the data into two portions: training data and validation data
% Here we pick 30 male, 30 female (totally 60) for model training

% Separate male and female
m_data=data(label==1,:);
f_data=data(label==0,:);
NUM_M=length(m_data);
NUM_F=length(f_data);


TRAIN_NUM_M=floor(0.3*NUM_M);
TRAIN_NUM_F=floor(0.3*NUM_F);

%% Set the initial state of random number
% so that the results will be the same when this
% script is executed multiple times.
rng('default')
rng(0)

% Randomly permutate the male records and separate training and validation
r=randperm(NUM_M);
train_x(1:TRAIN_NUM_M,:)=m_data(r(1:TRAIN_NUM_M),:);
valid_x(1:NUM_M-TRAIN_NUM_M,:)= m_data(r(TRAIN_NUM_M+1:NUM_M),:);

%% Set the initial state of random number
% so that the results will be the same when this
% script is executed multiple times.
rng('default')
rng(0)


% Randomly permutate the female records and separate training and
% validation
r=randperm(NUM_F);
train_x(TRAIN_NUM_M+1:TRAIN_NUM_M+TRAIN_NUM_F,:)= ...
    f_data(r(1:TRAIN_NUM_F),:);
valid_x(NUM_M-TRAIN_NUM_M+1:NUM_M-TRAIN_NUM_M+NUM_F-TRAIN_NUM_F,:)=...
    f_data(r(TRAIN_NUM_F+1:NUM_F),:);


train_d=zeros(TRAIN_NUM_M+TRAIN_NUM_F,1);
train_d(1:TRAIN_NUM_M)=1;

valid_d=zeros(NUM_M+NUM_F-TRAIN_NUM_M-TRAIN_NUM_F,1);
valid_d(1:NUM_M-TRAIN_NUM_M)=1; %it should be TRAIN_NUM_F