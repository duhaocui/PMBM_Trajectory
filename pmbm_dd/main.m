% clc;clear
dbstop if error

Pd = 0.9;
lfai = 10;
slideWindow = 3;

numMonteCarlo = 100;
K = 101; % time steps in total
% Set simulation parameters
numtruth = 6; % number of targets
% simcasenum = 1; % simulation case 1 or 2
simcasenum = 2;
% covariance used for mid-point initialisation
Pmid = 1e-6*eye(4);

% GOSPA metric
gospa_vals = zeros(K,4,numMonteCarlo);
gospa_c = 20;
gospa_p = 1;
gospa_alpha = 2;

est = cell(numMonteCarlo,1);

% Generate truth
for trial = 1:numMonteCarlo
    [model,measlog,xlog] = gentruth(Pd,lfai,numtruth,Pmid,simcasenum,slideWindow);
    % Multi-Bernoulli representation
    trajectory = struct([]);
    n = 0;
    r = zeros(n,1);
    x = zeros(4,n);
    P = zeros(4,4,n);
    l = cell(n,1);  % store trajectory
    c = zeros(n,1); % store the cost of each single target hypothesis
    a = zeros(n,1); % store track index

    % Unknown target PPP parameters
    unknownPPP.lambdau = model.lambdab;
    unknownPPP.xu = model.xb;
    unknownPPP.Pu = model.Pb;
    
    % Loop through time
    xest = cell(K,1);
    tra = cell(K,1);
    trackIndex = cell(K,1);
    
    for t = 1:K 

        % Predict all single target hypotheses of previous scan
        [r,x,P,unknownPPP] = predictStep(r,x,P,unknownPPP,model);
        
        % Update all predicted single target hypotheses of previous scan
        [unknownPPP,rupd,xupd,Pupd,lupd,cupd,rnew,xnew,Pnew,lnew,cnew,aupd,anew] = ...
            updateStep(unknownPPP,r,x,P,l,c,a,model,measlog{t},t);
        
        % multi-scan data association
        [r_hat,x_hat,P_hat,l_hat,a_hat,r,x,P,l,c,a] = ...
            dataAssoc(rupd,xupd,Pupd,lupd,cupd,aupd,rnew,xnew,Pnew,lnew,cnew,anew,model);
        
        % Target state extraction
        xest{t} = stateExtract(r_hat,x_hat);
       
        % Performance evaluation using GOSPA metric
        gospa_vals(t,:,trial) = gospa_dist(get_comps(xlog{t},[1 3]),...
            get_comps(xest{t},[1 3]),gospa_c,gospa_p,gospa_alpha);
        
        [trial,t]
    end

    est{trial} = xest;
end

pmbm_dd.estimation = est;
pmbm_dd.gospa_vals = gospa_vals;

save(strcat('pmbm_dd',num2str(100*Pd),num2str(lfai),num2str(slideWindow)),'pmbm_dd');

averGospa = mean(gospa_vals,3);
mean(averGospa)
