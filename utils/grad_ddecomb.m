function [grads_theta,grads_tau,loss] = grad_ddecomb(tt_all,tt_int,tt,t_simu,dltheta,dltau,input_x,targets_dx,dlhist_vec,hist_t,targets,lm,st1,st2,w)
loss1 = 0;
for i = 1:length(input_x)
    x = reshape(interp1(tt_all,input_x{i}',tt_int)',size(input_x{1},1),[]);
    xdelay = [];
    for k = 1:length(dltau)
        xdelay = [xdelay; reshape(interp1(tt_all,input_x{i}',tt_int-dltau(k))',size(input_x{1},1),[])];
    end
    dldx = ddeModel(tt_int,x,xdelay,dltheta);

    if strcmp(lm,'L1')
        if strcmp(w,'weighted')
            loss1 = loss1+l1loss(dldx,targets_dx{i},repmat(1./sqrt(st2),size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'average')
            loss1 = loss1+l1loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        end
    
    elseif strcmp(lm,'L2')
       if strcmp(w,'weighted')
            loss1 = loss1+l2loss(dldx,targets_dx{i},repmat(1./sqrt(st2),size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'average')
            loss1 = loss1+l2loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        end
    end
end
loss1 = loss1/length(input_x);

loss2 = 0;
for i = 1:length(dlhist_vec)
    dlhist = @(t)interp1(hist_t,dlhist_vec{i}',t)';
    dlx = dl_mdde_ab4(@(t,x,xdelay,theta)ddeModel(t,x,xdelay,theta),dltheta,dltau,dlhist,t_simu);
    gap = round((tt(2)-tt(1))/(t_simu(2)-t_simu(1)));
    dlx = dlx(:,1:gap:end);

    if strcmp(lm,'L1')
%         newloss = l1loss(dlx(:,2:end),targets{i}(:,2:end),'NormalizationFactor','all-elements','DataFormat','CT');
        newloss = l1loss(dlx(:,2:end),targets{i}(:,2:end),repmat(1./(1:size(dlx,2)-1),size(dlx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        if strcmp(w,'weighted')
            newloss = 1/sqrt(st1(mod(i-1,length(st1))+1))*newloss;
        elseif strcmp(w,'average')

        end
    
    elseif strcmp(lm,'L2')
%         newloss = l2loss(dlx(:,2:end),targets{i}(:,2:end),'NormalizationFactor','all-elements','DataFormat','CT');
        newloss = l2loss(dlx(:,2:end),targets{i}(:,2:end),repmat(1./(1:size(dlx,2)-1),size(dlx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        if strcmp(w,'weighted')
            newloss = 1/sqrt(st1(mod(i-1,length(st1))+1))*newloss;
        elseif strcmp(w,'average')
        end
   end

    loss2 = loss2 + newloss;
end
loss2 = loss2/length(dlhist_vec);
loss = loss1+loss2;
grads_theta = dlgradient(loss1,dltheta);
grads_tau = dlgradient(loss1,dltau);

end