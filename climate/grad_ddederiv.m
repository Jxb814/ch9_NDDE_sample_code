function [grads_theta,grads_tau,loss] = grad_ddederiv(tt_all,tt_int,dltheta,dltau,input_x,input_u,targets_dx,lm,st,w)
loss = 0;
for i = 1:length(targets_dx)
    x = reshape(interp1(tt_all,input_x{i}',tt_int)',size(input_x{1},1),[]);
    xdelay = [];
    for k = 1:length(dltau)
        xdelay = [xdelay; reshape(interp1(tt_all,input_x{i}',tt_int-dltau(k))',size(input_x{1},1),[])];
    end
    u = @(t) interp1(tt_int,input_u{i},t);
    dldx = ddeModel(tt_int,x,xdelay,dltheta,u);
    if strcmp(lm,'L1')
        if strcmp(w,'average')
%             newloss = sum(sum(abs(dldx-targets_dx{i}),1))/size(dldx,1)/size(dldx,2);
            newloss = l1loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'weighted')
            newloss = l1loss(dldx,targets_dx{i},repmat(1./st,size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        end
    elseif strcmp(lm,'L2')
        if strcmp(w,'average')
            newloss = l2loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'weighted')
            newloss = l2loss(dldx,targets_dx{i},repmat(1./st,size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        end
    elseif strcmp(lm,'L15')
        if strcmp(w,'average')
            newloss = sum(sum(sqrt(abs(dldx-targets_dx{i}).^3),1))/size(dldx,1)/size(dldx,2);
        elseif strcmp(w,'weighted')
            newloss = sum(sum(repmat(1./st,size(dldx,1),1).*sqrt(abs(dldx-targets_dx{i}).^3),1))/size(dldx,1)/size(dldx,2);
        end
    elseif strcmp(lm,'L3')
        if strcmp(w,'average')
            newloss = sum(sum(abs(dldx-targets_dx{i}).^3,1))/size(dldx,1)/size(dldx,2);
        elseif strcmp(w,'weighted')
            newloss = sum(sum(repmat(1./st,size(dldx,1),1).*abs(dldx-targets_dx{i}).^3,1))/size(dldx,1)/size(dldx,2);
        end
    elseif strcmp(lm,'L4')
        if strcmp(w,'average')
            newloss = sum(sum((dldx-targets_dx{i}).^4,1))/size(dldx,1)/size(dldx,2);
        elseif strcmp(w,'weighted')
            newloss = sum(sum(repmat(1./st,size(dldx,1),1).*(dldx-targets_dx{i}).^4,1))/size(dldx,1)/size(dldx,2);
        end
    elseif strcmp(lm,'L6')
        if strcmp(w,'average')
            newloss = sum(sum((dldx-targets_dx{i}).^6,1))/size(dldx,1)/size(dldx,2);
        elseif strcmp(w,'weighted')
            newloss = sum(sum(repmat(1./st,size(dldx,1),1).*(dldx-targets_dx{i}).^6,1))/size(dldx,1)/size(dldx,2);
        end

    end
    loss = loss + newloss;
end
loss = loss/length(targets_dx);
grads_theta = dlgradient(loss,dltheta);
grads_tau = dlgradient(loss,dltau); 
end