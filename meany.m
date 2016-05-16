function [output]=meany(Y,limit)
output=zeros(limit,1);
for k=1:limit
n=floor(1/k*length(Y));
sum1=0;
for i=1:n
    sum1=sum1+Y(i,1);
end
output(k,1)=sum1/n;
end
end