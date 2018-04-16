function [r_hat,x_hat,aout,cout,rout,xout,Pout,lout] = dataAssoc...
    (aupd,cupd,rupd,xupd,Pupd,lupd,cnew,rnew,xnew,Pnew,lnew,anew,model)

Hpre = length(cupd);        % num of single target hypotheses updating pre-existing tracks
Hnew = length(cnew);        % num of single target hypotheses updating new tracks
H = Hpre+Hnew;              % total num of single target hypotheses
mcur = Hnew/2;              % num of measurements in current scan

% If there is no pre-existing track
if Hpre == 0
    r_hat = rnew;
    x_hat = xnew;
    rout = rnew;
    xout = xnew;
    Pout = Pnew;
    lout = lnew;
    cout = cnew;
    aout = anew;
    return;
end

% n: num of pre-existing tracks; num of new tracks = num of meas at current scan
unique_a = unique(aupd,'stable');
npre = length(unique_a);
% number of single target hypotheses in pre-exist track i, i belongs {1,...,npre}
ns = zeros(npre,1);
for i = 1:npre
    ns(i) = length(find(aupd==unique_a(i)));
end

% calculate the length of trajectory of each single target hypothesis in
% pre-existing tracks, used for determine the num of scans should be used
tralen = cellfun(@(x) size(x,1),lupd);

% cost of single target hypotheses
c = [cupd;cnew];
c = c - min(c(:));

% construct binary indicator matrix for constraint (1): each track should
% only be used once, each measurement received in current scan creates a
% new track; this constraint is neccessary for the implementation of dual
% decomposition, since sliding window is used.
A0 = zeros(npre+mcur,H);
% for each pre-existing track
idx = 0;
for i = 1:npre
    A0(i,idx+1:idx+ns(i)) = 1;
    idx = idx+ns(i);
end
% for each new track
for i = 1:mcur
    A0(npre+i,idx+1:idx+2) = 1;
    idx = idx+2;
end

maxtralen = max(tralen); % maxtralen-scan data association 
tratemp = cell(maxtralen,1);

% construct binary indicator matrix for constraint (2): each measurement in
% each scan should only be used once
% target trajectory of the current scan
tratemp{1} = cellfun(@(x) x(end,2),lupd);
tcur = [tratemp{1};cellfun(@(x) x(1,2),lnew)];
At = false(mcur,H);
for i = 1:mcur
    At(i,tcur==i) = true;
end

% constriant equality
b0 = ones(npre+mcur,1);
bt = ones(mcur,1);

slideWindow = model.slideWindow; % apply sliding window
maxtralen(maxtralen>slideWindow) = slideWindow;

A = cell(maxtralen,1);
b = cell(maxtralen,1);

A{1} = At;
b{1} = bt;
% Htl(1) = H;

for tl = 2:maxtralen
    % get number of single target hypotheses existed at current-tl+1 scan
    idx = find(tralen>=tl);
    Htemp = length(idx);
    % target trajectory of the current-tl+1 scan, tracks from left to
    % right, old to new
    tratemp{tl} = cellfun(@(x) x(end-tl+1,2),lupd(idx));
    % num of measurements in current-tl+1 scan, do not count missed detection and
    % non-exist new track
    measUnique = unique(tratemp{tl}(tratemp{tl}~=0),'stable');
    mtemp = length(measUnique);
    
    Atemp = false(mtemp,Htemp); % current-tl+1 scan
    for i = 1:mtemp
        Atemp(i,tratemp{tl}==measUnique(i)) = true;
    end
    Atemp = cat(2,Atemp,false(mtemp,H-Htemp));
    btemp = ones(mtemp,1);
    
    A{tl} = Atemp;
    b{tl} = btemp;
    
    At = cat(1,At,Atemp);
    bt = cat(1,bt,btemp);
end

Amatrix = [A0;At];
sparseAmatrix = sparse(Amatrix);

options = optimoptions('intlinprog','Display','off');
% u = intlinprog(c,1:length(c),[],[],sparse(Amatrix),[b0;bt],zeros(length(c),1),ones(length(c),1),[],options);
% u = round(u);

sparseMatrix = cell(maxtralen,1);
for tl = 1:maxtralen
    sparseMatrix{tl} = sparse([A0;A{tl}]);
end

% pre-calculated parameters
measindices = cell(maxtralen,1);
idx_miss = cell(maxtralen,1);
is_preCalculated = cell(maxtralen,1);
tratl = cell(maxtralen,npre);
missornull = cell(maxtralen,npre);

for tl = 1:maxtralen
    num_meas = size(A{tl},1);
    measindices{tl} = cell(num_meas,npre);
    idx_miss{tl} = cell(num_meas,npre);
    for j = 1:num_meas
        idx = 0;
        for i = 1:npre
            % find single target hypotheses in track i that use this
            % measurement
            is_preCalculated{tl}{j,i} = find(A{tl}(j,idx+1:idx+ns(i))==true);
            if ~isempty(is_preCalculated{tl}{j,i})
                measindices{tl}{j,i} = tratemp{tl}(idx+1:idx+ns(i));
                idx_miss{tl}{j,i} = find(measindices{tl}{j,i}==0);
            end
            idx = idx+ns(i);
        end
    end
end

for tl = 1:maxtralen
    idx = 0;
    for i = 1:npre
        if  size(lupd{idx+1},1) >= tl+1
            tratl{tl}{i} = tratemp{tl}(idx+1:idx+ns(i));
            missornull{tl}{i} = find(tratl{tl}{i}==0);
        end
        idx = idx+ns(i);
    end
end

%%
% if 1
% dual decomposition, solution is a binary indicator vector, decides which
% single target hypotheses are included in the "best" global hypotheses.

% subproblem t: min(c/t+\deltat)*u, s.t. [A0;At]*u = [b0;bt];

% Larange multiplier \delta is initialised with 0
delta = zeros(H,maxtralen);
% subproblem solutions
u_hat = false(H,maxtralen);

% initialise maximum num of iteration
numIteration = 0;
maxIteration = 1e1;
numRepetition = 0;
% store the best feasible primal cost obtained so far (upper bound)
bestPrimalCost = inf;
uprimal = false(H,1);

while (numIteration<maxIteration&&numRepetition<1)
    % get suboptimal solution for each subproblem
    subDualCost = zeros(maxtralen,1);
    for tl = 1:maxtralen
        method = 2;
        switch method
            case 1
                % implementation using branch and bound, the gap needs
                % to be adjusted
                [u_hat(:,tl),subDualCost(tl)] = intlinprog((c/maxtralen+delta(:,tl)),1:length(c),[],[],...
                    sparse([A0;A{tl}]),[b0;b{tl}],zeros(length(c),1),ones(length(c),1),[],options);
            case 2
                % implementation using auction
                c_hat = c/maxtralen+delta(:,tl);
                c_temp = min(c_hat);
                c_hat = c_hat - c_temp;
                % get number of measurements at scan tl
                num_meas = size(A{tl},1);
                % construct track to measurement assignment matrix at scan tl
                cost = ones(npre+mcur,num_meas)*inf;
                % store assignment index of the single target hypothesis with the
                % minimum cost in each track
                idxCost = zeros(npre+mcur,num_meas);
                for j = 1:num_meas
                    idx = 0;
                    for i = 1:npre
                        % find single target hypotheses in track i that use this
                        % measurement
                        if ~isempty(is_preCalculated{tl}{j,i})
                            % if found, find the single target hypothesis with the
                            % minimum cost, and record its index
                            [cost(i,j),idxmin] = min(c_hat(idx+is_preCalculated{tl}{j,i}));
                            if ~isempty(idx_miss{tl}{j,i})
                                cost(i,j) = cost(i,j) - min(c_hat(idx+idx_miss{tl}{j,i}));
                            end
                            idxCost(i,j) = idx+is_preCalculated{tl}{j,i}(idxmin);
                        end
                        idx = idx+ns(i);
                    end
                    for i = npre+1:npre+mcur
                        is = find(A{tl}(j,idx+1:idx+2)==true);
                        if ~isempty(is)
                            [cost(i,j),idxmin] = min(c_hat(idx+is));
                            cost(i,j) = cost(i,j) - min(c_hat(idx+find(A{tl}(j,idx+1:idx+2)==false),1));
                            idxCost(i,j) = idx+is(idxmin);
                        end
                        idx = idx+2;
                    end
                end
                % find the most likely assignment
                costInput = [cost inf*ones(npre+mcur,npre+mcur-num_meas)];
                costInput = costInput-min(costInput(:));
                [assignments,~] = assignmentoptimal(costInput);
                assignments(assignments>num_meas) = 0;
                indicator = false(H,1);
                for i = 1:npre+mcur
                    if assignments(i)>0
                        indicator(idxCost(i,assignments(i))) = true;
                    end
                end
                % if a track has no measurement assigned to it, choose the single
                % target hypotheses to be non-exist or miss if the track exists
                % before scan N-tl, (if exists after scan N-tl, ofcourse, no measurement assigned)
                utemp = false(H,1);
                utemp(indicator) = true;
                
                idx = 0;
                for i = 1:npre
                    if ~any(utemp(idx+1:idx+ns(i)))
                        if size(lupd{idx+1},1) >= tl+1
                            [~,idxtratl] = min(c_hat(missornull{tl}{i}+idx));
                            utemp(idx+missornull{tl}{i}(idxtratl)) = true;
                        else
                            [~,idxmin] = min(c_hat(idx+1:idx+ns(i)));
                            utemp(idx+idxmin) = true;
                        end
                    end
                    idx = idx+ns(i);
                end
                for i = 1:mcur
                    if ~any(utemp(idx+1:idx+2))
                        [~,idxmin] = min(c_hat(idx+1:idx+2));
                        utemp(idx+idxmin) = true;
                    end
                    idx = idx+2;
                end
                u_hat(:,tl) = utemp;
                c_hat = c_hat + c_temp;
                subDualCost(tl) = c_hat'*u_hat(:,tl);
                
        end
    end
    u_hat = logical(u_hat);
    
    u_hat_mean = sum(u_hat,2)/maxtralen;
    
    % All the subproblem solutions are equal means we have found the
    % optimal solution
    if isempty(find(u_hat_mean~=1&u_hat_mean~=0,1))
        uprimal = u_hat(:,1);
        break;
    end
    
    % second calculate dual cost
    dualCosthat = sum(subDualCost);
    
    % find a feasible primal solution using branch&bound
    idx_selectedHypo = u_hat_mean==1;
    idx_unselectedHypo = ~idx_selectedHypo;
    idx_uncertainMeasTracks = sum(Amatrix(:,idx_selectedHypo),2)~=1;
    % if we are certain about the track, remove the single target
    % hypotheses of it
    for i = 1:length(idx_uncertainMeasTracks)
        if idx_uncertainMeasTracks(i) == false
            idx_unselectedHypo(Amatrix(i,:)==1) = false;
        end
    end
    
    A_uncertain = Amatrix(idx_uncertainMeasTracks,idx_unselectedHypo);
    b_uncertain = ones(size(A_uncertain,1),1);
    c_uncertain = c(idx_unselectedHypo);
    
    sparseA_uncertain = sparse(A_uncertain);
    
    params.outputflag = 0;          % Silence gurobi
    params.method     = 1;          % Use dual simplex method
    model_gurobi.A = sparseA_uncertain;
    model_gurobi.obj = c_uncertain;
    model_gurobi.sense = '=';
    model_gurobi.rhs = b_uncertain;
    model_gurobi.vtype = 'B';
    model_gurobi.Q = sparse(size(A_uncertain,2),size(A_uncertain,2));
    result = gurobi(model_gurobi, params);
    try
        uprimal_uncertain = result.x;
    catch
        uprimal_uncertain = [];
    end

    % When there is no feasible solution found, one should use greedy
    % method to find a feasible solution, if no solution found, set the
    % hypothesis to null-hypothesis
    
    if isempty(uprimal_uncertain) % No feasible solution found
        uprimalhat = intlinprog([],1:length(c),[],[],...
            sparseAmatrix,[b0;bt],zeros(length(c),1),ones(length(c),1),[],options);
        uprimalhat = round(uprimalhat);
    else
        uprimalhat = idx_selectedHypo;
        uprimalhat(idx_unselectedHypo) = logical(uprimal_uncertain);
    end
    
    conflicts = Amatrix*uprimalhat-[b0;bt];
    if any(conflicts)
        uprimalhat = intlinprog([],1:length(c),[],[],...
            sparseAmatrix,[b0;bt],zeros(length(c),1),ones(length(c),1),[],options);
        uprimalhat = round(uprimalhat);
    end
    
    bestPrimalCosthat = c'*uprimalhat;
    
    if bestPrimalCosthat < bestPrimalCost
        bestPrimalCost = bestPrimalCosthat;
        uprimal = uprimalhat;
        numRepetition = 0;
    else
        numRepetition = numRepetition + 1;
    end
    
    gap = (bestPrimalCost - dualCosthat)/bestPrimalCost;
    if gap < 0.05
        break;
    end
    
    % calculate step size used in subgradient methods
    % third calculate subgradient
    g = u_hat - u_hat_mean;
    
    % fourth calculate step size used in subgradient method
    stepSize = (bestPrimalCost - dualCosthat)/(norm(g)^2);
    
    % update Lagrange multiplier
    delta = delta + stepSize*g;
    
    numIteration = numIteration+1;
end

u = uprimal;

%%

% single target hypotheses in the ML global association hypotheses updating
% pre-existing tracks
I = u(1:Hpre)==1;
r_hat = rupd(I);
x_hat = xupd(:,I);
l = lupd(I);
c = cupd(I);

% single target hypotheses in the ML global association hypotheses
% updating new tracks
Inew = u(Hpre+1:H)==1;
r_hat = [r_hat;rnew(Inew)];
x_hat = [x_hat xnew(:,Inew)];

% N-scan pruning
idx = 0;
idx_remain = false(Hpre,1);
nc = 1; % prune null-hypothesis with length no less than nc
for i = 1:size(l,1)
    if (size(l{i},1)>=nc && ns(i)==1 && isequal(l{i}(end-nc+1:end,2),zeros(nc,1)))
        % prune null-hypothesis
    else
        if size(l{i},1)>=slideWindow
            traCompared = l{i}(1:end-slideWindow+1,2);
            for j = idx+1:idx+ns(i)
                if isequal(lupd{j}(1:end-slideWindow+1,2),traCompared)
                    idx_remain(j) = true;
                end
            end
        else
            idx_remain(idx+1:idx+ns(i)) = true;
        end
    end
    idx = idx+ns(i);
end

rupd = rupd(idx_remain);
xupd = xupd(:,idx_remain);
Pupd = Pupd(:,:,idx_remain);
lupd = lupd(idx_remain);
cupd = cupd(idx_remain);
aupd = aupd(idx_remain);

% Find single target hypotheses with existence probability smaller than a
Aupd = At(:,1:Hpre);
Aupd = Aupd(:,idx_remain);
idx_smallExistenceProb = rupd<model.threshold&rupd>0;
idx_measSmallExistenceProb = sum(Aupd(:,idx_smallExistenceProb),2)>=1;
idx_highExistenceProb = rupd>=model.threshold;
idx_measHighExistenceProb = sum(Aupd(:,idx_highExistenceProb),2)>=1;
indicator = (idx_measSmallExistenceProb-idx_measHighExistenceProb)==1;
len = size(Aupd,2);
idx_remain = true(len,1);
idx_remain(sum(Aupd.*indicator)>0) = false;

% idx_remain = ~idx_smallExistenceProb;
rupd = rupd(idx_remain);
xupd = xupd(:,idx_remain);
Pupd = Pupd(:,:,idx_remain);
lupd = lupd(idx_remain);
cupd = cupd(idx_remain);
aupd = aupd(idx_remain);

% if a track contains too many single target hypothesis, crop it, we should
% prune single target hypothesis with score larger than the one in the ML
% estimation to keep the total number with in a largest allowed threshold.
% Aupd = Aupd(:,idx_remain);
unique_a = unique(aupd,'stable');
npre = length(unique_a);
% number of single target hypotheses in pre-exist track i, i belongs {1,...,npre}
ns = zeros(npre,1);
for i = 1:npre
    ns(i) = length(find(aupd==unique_a(i)));
end
maxNumHypoperTrack = (slideWindow-1)*10;
idx = 0;
idxtobePrune = false(length(cupd),1);
for i = 1:npre
    if ns(i) > maxNumHypoperTrack
        [~,idxSort] = sort(cupd(idx+1:idx+ns(i)));
        idxtobePrune(idxSort(maxNumHypoperTrack+1:end)+idx) = true;
        idxtobePrune(find(cupd(idx+1:idx+ns(i))==c(i))+idx) = false;
    end
    idx = idx + ns(i);
end

% for each branch at time scan tau-1, at least keep one leaf
idx = 0;
for i = 1:npre
%     trajectory_one_step_back = cellfun(@(x) x(end-1),lupd(idx+1:idx+ns(i)));
    trajectory_one_step_back = tratemp{2}(idx+1:idx+ns(i));
    uniqueTra = unique(trajectory_one_step_back,'stable');
    for j = 1:length(uniqueTra)
        temp = ~idxtobePrune(idx+1:idx+ns(i));
        tempc = cupd(idx+1:idx+ns(i));
        if ~any(temp(trajectory_one_step_back==uniqueTra(j)))
            [~,idxmin] = min(tempc);
            idxtobePrune(idx+idxmin) = false;
        end
    end
    idxtobePrune(idx+find(trajectory_one_step_back==0)) = false;
    idx = idx + ns(i);
end

idx_remain = ~idxtobePrune;
rupd = rupd(idx_remain);
xupd = xupd(:,idx_remain);
Pupd = Pupd(:,:,idx_remain);
lupd = lupd(idx_remain);
cupd = cupd(idx_remain);
aupd = aupd(idx_remain);

%
idx_smallExistenceProb = rnew<model.threshold&rnew>0;
idx_measSmallExistenceProb = cell2mat(lnew(idx_smallExistenceProb));
if ~isempty(idx_measSmallExistenceProb)
    idx_measSmallExistenceProb = idx_measSmallExistenceProb(:,2);
end
idx_meas = cellfun(@(x) x(end,2),lupd);
idx_measPrune = ~ismember(idx_measSmallExistenceProb,idx_meas);
measPrune = idx_measSmallExistenceProb(idx_measPrune);
temp = cell2mat(lnew);
idx_remain = ~ismember(temp(:,2),measPrune);
lnew = lnew(idx_remain);
anew = anew(idx_remain);
cnew = cnew(idx_remain);
rnew = rnew(idx_remain);
xnew = xnew(:,idx_remain);
Pnew = Pnew(:,:,idx_remain);

rout = [rupd;rnew];
xout = [xupd xnew];
Pout = cat(3,Pupd,Pnew);
lout = cat(1,lupd,lnew);
cout = [cupd;cnew];
aout = [aupd;anew];

end