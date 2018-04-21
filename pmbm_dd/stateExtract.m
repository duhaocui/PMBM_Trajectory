function est = stateExtract(trajectoryEst)
% Extract target states

% MAP cardinality estimate
r = extractfield(trajectoryEst,'r');
r(r==1) = 1-eps;                    % remove numerical error
ss = false(size(r));
pcard = prod(1-r)*poly(-r./(1-r));
[~,n] = max(pcard);
[~,o] = sort(-r);
n = n - 1;
ss(o(1:n)) = true;
trajectoryEst = trajectoryEst(ss);
len = length(trajectoryEst);
est = zeros(4,len);
for i = 1:len
    est(:,i) = trajectoryEst(i).x;
end

end

