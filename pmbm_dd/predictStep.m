function [trajectoryMBM,unknownPPP] = predictStep(trajectoryMBM,unknownPPP,model)
%PREDICT: PREDICT MULTI-BERNOULLI AND POISSON COMPONENTS

% Get multi-Bernoulli prediction parameters from model
F = model.F;
Q = model.Q;
Ps = model.Ps;

% Get birth parameters from model
lambdab = model.lambdab;
nb = length(lambdab);
xb = model.xb;
Pb = model.Pb;
lambdab_threshold = model.threshold;

lambdau = unknownPPP.lambdau;
xu = unknownPPP.xu;
Pu = unknownPPP.Pu;

% Interpret length of inputs
n = length(trajectoryMBM);
nu = length(lambdau);

% Implement prediction algorithm

% Predict existing tracks (single target hypotheses)
for i = 1:n
    if trajectoryMBM(i).r~=0
        trajectoryMBM(i).r = Ps*trajectoryMBM(i).r;
        trajectoryMBM(i).x = F*trajectoryMBM(i).x;
        trajectoryMBM(i).P = F*trajectoryMBM(i).P*F' + Q;
    end
end

% Predict existing PPP intensity
for k = 1:nu
    lambdau(k) = Ps*lambdau(k);
    xu(:,k) = F*xu(:,k);
    Pu(:,:,k) = F*Pu(:,:,k)*F' + Q;
end

% Incorporate birth intensity into PPP

% Allocate memory
lambdau(end+nb) = 0;
xu(:,end+nb) = 0;
Pu(:,:,end+nb) = 0;
for k = 1:nb
    lambdau(nu+k) = lambdab(k);
    xu(:,nu+k) = xb(:,k);
    Pu(:,:,nu+k) = Pb(:,:,k);
end

% Not shown in paper--truncate low weight components
ss = lambdau > lambdab_threshold;
unknownPPP.lambdau = lambdau(ss);
unknownPPP.xu = xu(:,ss);
unknownPPP.Pu = Pu(:,:,ss);

