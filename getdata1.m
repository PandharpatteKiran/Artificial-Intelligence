function [data,label]=getdata(csvfile)

% Skip the first row which is the header
raw_data = csvread(csvfile,0,0);
[rows,cols]=size(raw_data);
% The height and weight are the variables/features/attributes of each instance of data
data = raw_data(:,1:cols-1);

% The sex column is the target/label that the supervised learning istrying to learn to match
label = raw_data(:,cols);

end