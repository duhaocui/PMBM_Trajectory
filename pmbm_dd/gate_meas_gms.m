function [z_gate,z_out]= gate_meas_gms(z,model,m,P)

if isempty(m)
    z_gate = z;
    z_out = zeros(2,0);
else
    zlength = size(z,2);
    valid_idx = false(zlength,1);
    plength = size(m,2);
    for j=1:plength
        Sj= model.R + model.H*P(:,:,j)*model.H';
        nu= z- model.H*repmat(m(:,j),[1 zlength]);
        dist= sum((inv(chol(Sj))'*nu).^2);
        valid_idx(dist < model.gamma) = true;
    end
    z_gate = z(:,valid_idx);
    z_out = z(:,~valid_idx);
end

