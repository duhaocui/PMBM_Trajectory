function [r,x,P,lambdau,xu,Pu] = Recycling(r,x,P,lambdau,xu,Pu)

% Recycle Bernoulli components with small existence probability
recycle_threshold = 0.1;
ss = r < recycle_threshold;

% Allocate memory
nu = length(lambdau);
nr = length(find(ss==1));
lambdau(end+nr) = 0;
xu(:,end+nr) = 0;

% Recycle
lambdau(nu+1:end) = r(ss);
xu(:,nu+1:end) = x(:,ss);
Pu = cat(3,Pu,P(:,:,ss));

% Pruning
ss = ~ss;
r = r(ss);
x = x(:,ss);
P = P(:,:,ss);

end

