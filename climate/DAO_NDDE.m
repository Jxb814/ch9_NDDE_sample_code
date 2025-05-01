%% DAO model
clear
clc
close all
addpath('..\utils');
plottingPreferences()
%%
sys = 'DAO';
a = 2.02;
b = 3.03;
c = 2.6377;
du = 2.0;
dl = -0.4;
k = 2.0;
truetau = [0.0958 0.4792];
tau_max = ceil(2*max(truetau));
u = @(t)cos(2*pi*t);
A = @(h) (h>=0)*du*tanh(k/du*h)+(h<0)*dl*tanh(k/dl*h);
trueModel = @(t,x,xdelay) a*A(xdelay(1))-b*A(xdelay(2))+c*u(t);

T_tr = 30;
T_ts = 5;
Tst = -tau_max; 
tend = Tst+T_tr+T_ts; 
dt = 0.05;
dt_simu = 0.01;

tr_c =  1;
clean_data = cell(1,length(tr_c));
xdata = cell(1,length(tr_c));
xdata_all= cell(1,length(tr_c));
targets_all = cell(1,length(tr_c));
targets_dx =cell(1,length(tr_c));
udata = cell(1,length(tr_c));

hist_time = Tst:dt:Tst+tau_max;
tt_all = Tst:dt:tend;      % matching xdata_all
tt_target = Tst+tau_max:dt:tend;
t_deriv = Tst+tau_max:dt:Tst+T_tr;
for i = 1:length(tr_c)
    histvec_int = tr_c(i);
    hist_int = @(t)histvec_int;
    sol = dde23(trueModel,truetau,hist_int,[0 tend]);
    clean_data_t = 0:0.01:tend;
    clean_data{i}  = deval(sol,clean_data_t);
    clean_data{i}  = [repmat(histvec_int,1,round(tau_max/0.01)) clean_data{i}];       
    clean_data_t = Tst:0.01:tend;

    xdata{i}  = deval(sol,0:dt:tend);
    xdata_all{i} = [repmat(histvec_int,1,round(tau_max/dt)) xdata{i}];
    targets_all{i} = xdata{i};   % no history
    targets_dx{i} = ctrDiff(targets_all{i},tt_target);
    udata{i} = u(tt_target);
end
%%
figure(1)
set(gcf,'Position',[100 100 600 200])
hold on;
for i = 1:length(xdata)
    plot(clean_data_t,clean_data{i}(:))
    scatter(tt_all,xdata_all{i}(:),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
end
plot([Tst+tau_max Tst+tau_max],[min(clean_data{i}(:)) max(clean_data{i}(:))],'k--')
plot([Tst Tst],[min(clean_data{i}(:)) max(clean_data{i}(:))],'k-')
plot([Tst+T_tr Tst+T_tr],[min(clean_data{i}(:)) max(clean_data{i}(:))],'k-')
hold off;
box on;
title('Climate Dynamics')
xlim([Tst tend])
ylim([min(clean_data{i}(:)) max(clean_data{i}(:))])
ylabel('$x(t)$')
xlabel('$t$')

%% learning 
% lossfun='simu';
% lm = 'L2';  % loss measure, which norm to use
% w = 'average';
lossfun='deriv';
lm = 'L2'; 
w = 'weighted';
plotFrequency = 1000;
if strcmp(lossfun,'deriv')
    numIter = 20000;
    learnRate = 0.01;
    ratio = 0.1;
    batchSize = 200; 
    steps = 1;
elseif strcmp(lossfun,'simu')
    numIter = 10000;
    learnRate = 0.01;
    ratio = 1;
    batchSize = 20;
    steps = 10;
end
gap = round(dt/dt_simu); 

hiddenSize = 20;
nx = 1;
nu = 1;
% tau = dlarray([0.0958 0.4792]);
% tau = dlarray([0.3 0.6 0.8]);
tau = dlarray([0.3 0.8]);
nd = length(tau);

NDDE = struct;
NDDE.fc1 = struct;
sz = [hiddenSize nx*nd+nu];
NDDE.fc1.Weights = initializeGlorot(sz);
NDDE.fc1.Bias    = initializeZeros([sz(1) 1]);

NDDE.fc2 = struct;
sz = [hiddenSize hiddenSize];
NDDE.fc2.Weights = initializeGlorot(sz);
NDDE.fc2.Bias    = initializeZeros([sz(1) 1]);

NDDE.fc3 = struct;
sz = [nx hiddenSize];
NDDE.fc3.Weights = initializeGlorot(sz);

%%

start = tic;
Loss = zeros(1,numIter);
Tau_tr = zeros(length(tau),numIter);

gradDecay = 0.9;
sqGradDecay = 0.999;
aveGrad   = [];
aveSqGrad = [];
aveGrad_tau = [];
aveSqGrad_tau = [];
nsets = length(xdata);
nn = 0;
for iter = 1:numIter
    if strcmp(lossfun,'simu')
        hist_v = cell(1,nsets*batchSize);
        dlhist_v = cell(1,nsets*batchSize);
        targets_tr= cell(1,nsets*batchSize);
        udata_tr = cell(1,nsets*batchSize);
        tt = 0:dt:dt*steps;       % the sampling time
        t_simu = 0:(dt/gap):dt*steps; % simulation time step
        hist_t = -tau_max:dt:0;
        st = randperm(length(t_deriv)-steps,batchSize);
        for kk = 1:nsets
            for batch = 1:batchSize
                hist_v{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch):st(batch)+length(hist_t)-1);
                dlhist_v{(kk-1)*batchSize+batch} = dlarray(hist_v{(kk-1)*batchSize+batch});
                targets_tr{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch)+length(hist_t)-1:...
                    st(batch)+length(hist_t)-1+steps);
                udata_tr{(kk-1)*batchSize+batch} = udata{kk}(st(batch):st(batch)+steps);
            end
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddesimu,tt,t_simu,NDDE,tau,udata_tr,dlhist_v,hist_t,targets_tr,lm,st,w);
    elseif strcmp(lossfun,'deriv')
        targets_dx_tr = cell(1,nsets);
        udata_tr = cell(1,nsets*batchSize);
        st = randperm(length(t_deriv),batchSize);
        t_batch = t_deriv(st);
        for kk = 1:nsets
            targets_dx_tr{kk} = targets_dx{kk}(st);
            udata_tr{kk} = udata{kk}(st);
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddederiv,tt_all,t_batch,NDDE,tau,xdata_all,udata_tr,targets_dx_tr,lm,st,w);
    end
    
    Tau_tr(:,iter)=extractdata(tau');
    currentLoss = double(extractdata(loss));
    Loss(iter) = currentLoss;
    if Loss(iter)<=min(Loss(1:iter))
        NDDE_best = NDDE;
        tau_best = tau;
        nn = 0;
    elseif iter>10 && Loss(iter)>mean(Loss(iter-10:iter))
        nn = nn+1;
    end

    if mod(iter,plotFrequency) == 0  || iter == 1
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
%%      
        figure(2)
        set(gcf,'Units','normalized','Position',[0 0 1 0.9])
        clf
        subplot(2,3,1)
        semilogy(1:iter, Loss(1:iter),'b')
        xlabel('Iteration')
        xlim([1,iter+1])
        ylabel('Loss')
        title(['min loss = ',num2str(min(Loss(1:iter)))])
%         title("Training loss")

        subplot(2,3,4)
        hold on
        for k = 1:nd
            plot(1:iter, Tau_tr(k,1:iter),'b','LineWidth',2)
        end
        plot(1:iter+1, truetau.*ones(length(truetau),iter+1)', 'k--',LineWidth=2)
        ylim([0,tau_max])
        xlim([1,iter+1])
        xlabel('Iteration')
        ylabel('$\tau$')
        hold off
        box on
        title("Iter = "+iter+", Elapsed: "+string(D))
%         title("Learned delay")
       

        subplot(2,3,2)
        % simulation
        W1 = double(extractdata(NDDE.fc1.Weights));
        W2 = double(extractdata(NDDE.fc2.Weights));
        W3 = double(extractdata(NDDE.fc3.Weights));
        b1 = double(extractdata(NDDE.fc1.Bias));
        b2 = double(extractdata(NDDE.fc2.Bias));
        NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[reshape(xdelay,[],1);u(t)]+b1)+b2);

        % predict the whole traj
        sol1 = dde23(@(t,x,xdelay)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target);
        tt_target_fine=Tst+tau_max:0.01:tend;
        y = deval(sol1,tt_target_fine);
        hold on
        plot(tt_target_fine,y,'b','LineWidth',1)
        plot(hist_time,xdata_all{1}(:,1:length(hist_time)),'b')
        scatter(tt_target,targets_all{1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time,xdata_all{1}(:,1:length(hist_time)),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[min(xdata_all{1}) max(xdata_all{1})],'k--', 'LineWidth',1)
        xlim([hist_time(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        ylim([min(xdata_all{1}) max(xdata_all{1})])
        title('Network simulation')

        subplot(2,3,5)
        hold on
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine-tau(k))',nx,[])];
        end
        dx = ddeModel(tt_target_fine,x,xdelay,NDDE,u);
        plot(tt_target_fine,extractdata(dx),'b')
        scatter(tt_target,targets_dx{1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[min(targets_dx{1}) max(targets_dx{1})],'k--', 'LineWidth',1)
        xlim([hist_time(1) tend])
        ylim([min(targets_dx{1}) max(targets_dx{1})])
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on

        subplot(2,3,3)
        % predict only for testing
        hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
        tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
        sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);
        tt_target_ts_fine = Tst+T_tr:0.01:Tst+T_tr+T_ts;
        y = deval(sol1,tt_target_ts_fine );
        hold on
        plot(tt_target_ts_fine ,y,'b','LineWidth',1)
        plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
        scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[min(xdata_all{1}) max(xdata_all{1})],'k--', 'LineWidth',1)
        xlim([hist_time_ts(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        ylim([min(xdata_all{1}) max(xdata_all{1})])
        title('Network simulation')

        subplot(2,3,6)
        hold on
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
        end
        dx = ddeModel(tt_target_ts_fine,x,xdelay,NDDE,u);
        plot(tt_target_ts_fine,extractdata(dx),'b')
        scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,"filled",'MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[min(targets_dx{1}) max(targets_dx{1})],'k--', 'LineWidth',1)
        xlim([hist_time_ts(1) tend])
        ylim([min(targets_dx{1}) max(targets_dx{1})])
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on
    end
    [NDDE,aveGrad,aveSqGrad] = adamupdate(NDDE,grads_NDDE,aveGrad,aveSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    [tau,aveGrad_tau,aveSqGrad_tau] = adamupdate(tau,grads_tau,aveGrad_tau,aveSqGrad_tau,iter,...
        ratio*learnRate,gradDecay,sqGradDecay);
    tau = min(max(0.00001,tau),tau_max-0.00001);
end
%%
if numIter == 0
    iter=0;
end

NDDE = NDDE_best;
tau = tau_best;
W1 = double(extractdata(NDDE.fc1.Weights));
W2 = double(extractdata(NDDE.fc2.Weights));
W3 = double(extractdata(NDDE.fc3.Weights));
b1 = double(extractdata(NDDE.fc1.Bias));
b2 = double(extractdata(NDDE.fc2.Bias));
NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[reshape(xdelay,[],1);u(t)]+b1)+b2);

figure(5)
set(gcf,'Position',[100 100 300 200])
semilogy(1:iter, Loss(1:iter))
hold on
[~,min_id] = min(Loss(1:iter));
semilogy([min_id min_id],[min(Loss(1:iter)) max(Loss(1:iter))],'k:')
hold off
xlim([1,iter+1])
xlabel('Iteration')
ylabel('Loss')
ylim([min(Loss(1:iter)) max(Loss(1:iter))])
yticks([0.00001 0.0001 0.001 0.01 0.1])

figure(6)
set(gcf,'Position',[100 100 300 200])
hold on
for k = 1:nd
    plot(1:iter, Tau_tr(k,1:iter),LineWidth=2)
    
end
plot(1:iter+1, truetau'.*ones(length(truetau),iter+1), 'k--',LineWidth=2)
plot(1:iter+1, zeros(1,iter+1),'k--',LineWidth=2)
plot([min_id min_id],[0 tau_max],'k:')
ylim([0,tau_max])
xlim([1,iter+1])
xlabel('Iteration')
ylabel('Delays')
hold off
box on
%%
figure(9)
set(gcf,'Position',[100 100 300 200])
% predict only for testing
hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
    @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);
tt_target_ts_fine = Tst+T_tr:0.01:Tst+T_tr+T_ts;
y = deval(sol1,tt_target_ts_fine);

hold on
plot(tt_target_ts_fine ,y,'b','LineWidth',1)
plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[min(xdata_all{1}) max(xdata_all{1})],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
xlabel('$t$')
ylabel('$x$')
hold off
box on
ylim([min(xdata_all{1}) max(xdata_all{1})])
title('Network simulation')

figure(11)
set(gcf,'Position',[100 100 300 200])
hold on
x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
xdelay = [];
for k = 1:length(tau)
    xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
end
dx = ddeModel(tt_target_ts_fine,x,xdelay,NDDE,u);
plot(tt_target_ts_fine,extractdata(dx),'b')
scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[min(targets_dx{1}) max(targets_dx{1})],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
ylim([min(targets_dx{1}(:,end-length(tt_target_ts)+1:end)) max(targets_dx{1}(:,end-length(tt_target_ts)+1:end))])
xlabel('$t$')
ylabel('$\dot{x}$')
title('Network direct prediction')
hold off
box on

%% get simulation error on testing data
y_test = deval(sol1,tt_target_ts);
L_simu=sum((y_test-targets_all{1}(:,end-length(tt_target_ts)+1:end)).^2)/length(tt_target_ts);
x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts)',nx,[]);
xdelay = [];
for k = 1:length(tau)
    xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts-tau(k))',nx,[])];
end
dy_test = ddeModel(tt_target_ts,x,xdelay,NDDE,u);
L_deriv=sum((dy_test-targets_dx{1}(:,end-length(tt_target_ts)+1:end)).^2)/length(tt_target_ts);
RMSE_deriv = rmse(dy_test,targets_dx{1}(:,end-length(tt_target_ts)+1:end))
RMSE_simu = rmse(y_test,targets_all{1}(:,end-length(tt_target_ts)+1:end))
tau

% %%
% filename = strcat('DAO\',sys,'_',lossfun,lm,'loss',num2str(steps),'step_batch',num2str(batchSize),'iter',num2str(iter),...
%     '_L2N',num2str(hiddenSize),'_lr',strrep(num2str(learnRate),'.',''),'ratio',strrep(num2str(ratio),'.',''),w,...
%     '_dt',strrep(num2str(dt),'.',''),'_dtsimu',strrep(num2str(dt_simu),'.',''),'_nd',num2str(length(tau)));
% save(filename)