function [r,x,P] = Pruning(r,x,P)
% Prune Bernoulli components with small existence probability

prune_threshold = 1e-3;
ss = r > prune_threshold;
r = r(ss);
x = x(:,ss);
P = P(:,:,ss);

end

