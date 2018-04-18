function [unknownPPP,trajectoryUpdMBM,trajectoryNewMBM] = ...
    updateStep2(unknownPPP,trajectoryMBM,model,z,t)
%UPDATE: CONSTRUCT COMPONENTS OF DISTRIBUTION UPDATED WITH MEASUREMENT

% Extract parameters from model
Pd = model.Pd;
H = model.H;
R = model.R;
lambda_fa = model.lambda_fa;

lambdau = unknownPPP.lambdau;
xu = unknownPPP.xu;
Pu = unknownPPP.Pu;

% Interpret sizes from inputs
% If it is a hypothesis with zero existence probability, no update
% e.g. n_invalid, n_valid, nupd = n_invalid + n_valid*(m+1);
[~,nu] = size(xu);
[~,m] = size(z);
n = length(trajectoryMBM);  % number of single target hypotheses to be updated
if n > 0
    nupd = length(find(extractfield(trajectoryMBM,'r')~=0))*m + n; 
else
    nupd = 0;
end

% Allocate memory for existing tracks (single target hypotheses)
trajectoryUpdMBM = struct('r',0,'x',zeros(4,1),'P',zeros(4,4),'l',[],'c',0,'a',[]);
trajectoryUpdMBM = repmat(trajectoryUpdMBM,nupd,1);

% Keep record ancestor information, i.e., the index of the single target 
% hypothesis being updated, can be used in N-scan pruning

% Update existing tracks
iupd = 0;   % initiate updating index
insideGating = true(nupd,1);
usedMeas = false(m,1);
for i = 1:n
    % first determine whether the single target hypothesis has valid
    % existence probability or not
    if trajectoryMBM(i).r == 0 % this case corresponds to non-existence single target hypothesis
        iupd = iupd+1;
        trajectoryUpdMBM(iupd).c = 0;
        trajectoryUpdMBM(iupd).r = 0;
        trajectoryUpdMBM(iupd).x = zeros(4,1);
        trajectoryUpdMBM(iupd).P = zeros(4,4,1);
        trajectoryUpdMBM(iupd).l = [trajectoryMBM(i).l;[t,0]];
        trajectoryUpdMBM(iupd).a = trajectoryMBM(i).a;
    else
        % Create missed detection hypothesis
        iupd = iupd+1;
        temp = 1-trajectoryMBM(i).r+trajectoryMBM(i).r*(1-Pd);
        trajectoryUpdMBM(iupd).c = trajectoryMBM(i).c-log(temp);
        trajectoryUpdMBM(iupd).r = trajectoryMBM(i).r*(1-Pd)/temp;
        trajectoryUpdMBM(iupd).x = trajectoryMBM(i).x;
        trajectoryUpdMBM(iupd).P = trajectoryMBM(i).P;
        % If it is missed detection, add label 0
        trajectoryUpdMBM(iupd).l = [trajectoryMBM(i).l;[t,0]];
        trajectoryUpdMBM(iupd).a = trajectoryMBM(i).a;
        
        if trajectoryMBM(i).r >= 1e-4
        % Create hypotheses with measurement updates
        S = H*trajectoryMBM(i).P*H' + R;
        sqrt_det2piS = sqrt(det(2*pi*S));
        K = trajectoryMBM(i).P*H'/S;
        Pplus = trajectoryMBM(i).P - K*H*trajectoryMBM(i).P;
        for j = 1:m
            iupd = iupd+1;
            v = z(:,j) - H*trajectoryMBM(i).x;
            temp = exp(-0.5*v'/S*v)/sqrt_det2piS; % temp < 0.0026
            if temp < 0.0005 % 3-sigma gating
                insideGating(iupd) = false;
            else
                usedMeas(j) = true;
                trajectoryUpdMBM(iupd).c = trajectoryMBM(i).c-log(trajectoryMBM(i).r*Pd*temp);
                trajectoryUpdMBM(iupd).r = 1;
                trajectoryUpdMBM(iupd).x = trajectoryMBM(i).x + K*v;
                trajectoryUpdMBM(iupd).P = Pplus;
                % Otherwise, add the index of the measurement at current scan
                trajectoryUpdMBM(iupd).l = [trajectoryMBM(i).l;[t,j]];
                trajectoryUpdMBM(iupd).a = trajectoryMBM(i).a;
            end
        end
        else
            insideGating(iupd+1:iupd+m) = false;
            iupd = iupd + m;
        end
    end
end

unusedMeas = ~usedMeas;


% Prune single target hypothesis with really small likelihood
idx_keep = insideGating;
trajectoryUpdMBM = trajectoryUpdMBM(idx_keep);

% Allocate memory for new tracks, each new track contains two single target
% hypothese
trajectoryNewMBM = struct('r',0,'x',zeros(4,1),'P',zeros(4,1),'l',[],'c',0,'a',[]);
trajectoryNewMBM = repmat(trajectoryNewMBM,2*m,1);

% Allocate temporary working for new tracks
Sk = zeros(2,2,nu);
Kk = zeros(4,2,nu);
Pk = zeros(4,4,nu);
ck = zeros(nu,1);
sqrt_det2piSk = zeros(nu,1);
yk = zeros(4,nu);

% Create a new track for each measurement by updating PPP with measurement
for k = 1:nu
    Sk(:,:,k) = H*Pu(:,:,k)*H' + R;
    sqrt_det2piSk(k) = sqrt(det(2*pi*Sk(:,:,k)));
    Kk(:,:,k) = Pu(:,:,k)*H'/Sk(:,:,k);
    Pk(:,:,k) = Pu(:,:,k) - Kk(:,:,k)*H*Pu(:,:,k);
end
for j = 1:m
    for k = 1:nu
        v = z(:,j) - H*xu(:,k);
        ck(k) = lambdau(k)*Pd*exp(-0.5*v'/Sk(:,:,k)*v)/sqrt_det2piSk(k);
        yk(:,k) = xu(:,k) + Kk(:,:,k)*v;
    end
    C = sum(ck);
    % first single target hypothesis for measurement associated to previous
    % track, second for new track
    trajectoryNewMBM(2*j-1).c = 0;
    trajectoryNewMBM(2*j).c = -log(C + lambda_fa);
    trajectoryNewMBM(2*j).r = C/(C + lambda_fa);
    ck = ck/C;
    trajectoryNewMBM(2*j).x = yk*ck;
    trajectoryNewMBM(2*j).P = zeros(4,4);
    for k = 1:nu
        v = trajectoryNewMBM(2*j).x - yk(:,k);
        trajectoryNewMBM(2*j).P = trajectoryNewMBM(2*j).P + ck(k)*(Pk(:,:,k) + v*v');
    end
    % for trajectory purpose
    trajectoryNewMBM(2*j-1).l = [t,0];    % add 0, if there is no new target
    trajectoryNewMBM(2*j).l = [t,j];      % otherwise, add measurement index
    if nupd==0
        trajectoryNewMBM(2*j-1).a = j;
        trajectoryNewMBM(2*j).a = j;
    else  
        trajectoryNewMBM(2*j-1).a = trajectoryUpdMBM(end).a+j;
        trajectoryNewMBM(2*j).a = trajectoryUpdMBM(end).a+j;
    end
end

idx_remain = true(2*m,1);
for i = 1:m
    if trajectoryNewMBM(2*i).r < model.threshold && unusedMeas(i)==true
        idx_remain(2*i-1:2*i) = false;
    end
end
trajectoryNewMBM = trajectoryNewMBM(idx_remain);

% Update (i.e., thin) intensity of unknown targets
unknownPPP.lambdau = (1-Pd)*lambdau;