%% Rossler NDDE
clear
clc
close all
addpath('..\utils');
plottingPreferences()
%%
model = 'logistic';

sys = 'logistic_addhist_clean';
truetau = [1];
tau_max = 1.5 * max(truetau);

r = 1.8;
K = 10;
trueModel = @(t, x, xdelay) r*x*(1-xdelay/K);
% 
% TR_C =cell(10,1);
% for i = 1:length(TR_C)
%     TR_C{i} = [2*rand-1; 2*rand-1; 2*rand];
% end

TR_C{1} = 1;
Tst = -tau_max;  


T_tr = 30;    % this includes the initial history!
T_ts = 5;
dt = 0.05;
tend = Tst+T_tr+T_ts; 
numSamples = round((tend-Tst)/dt); % including testing and history

% lossfun='simu'; 
lossfun='deriv';
% lossfun='comb'; 
lm = 'L2';  % loss measure, which norm to use
w = 'weighted';
% w = 'average';
if strcmp(lossfun,'deriv')
    numIter = 8000;
    plotFrequency = 1000;
    learnRate = 0.01;
    learnRate_tau = 0.001;
    batchSize = 200; 
    steps = 1;
    dt_simu = dt/1;
elseif strcmp(lossfun,'simu')
    numIter = 2000;
    plotFrequency = 10;
    learnRate = 0.01;
    learnRate_tau = 0.01;
    batchSize = 10;
    steps = 20;
    dt_simu = dt/5; % Smaller dt for simulation accuracy
    gap = round(dt / dt_simu); % Simulation time step = dt/gap
elseif strcmp(lossfun,'comb')
    numIter = 2000;
    plotFrequency = 10;
    learnRate = 0.01;
    learnRate_tau = 0.1;
    batchSize = 1;
    batchSize2 = 555;
    steps = 555; 
    dt_simu = dt/1; % Smaller dt for simulation accuracy
    gap = round(dt/dt_simu); % Simulation time step = dt/gap
end

clean_data = cell(1,1);
xdata = cell(1,1);
xdata_all= cell(1,1);
targets_all = cell(1,1);
targets_dx =cell(1,1);

hist_time = Tst:dt:Tst+tau_max;
tt_all = Tst:dt:tend;      % matching xdata_all
tt_target = Tst+tau_max:dt:tend;
t_deriv = Tst+tau_max:dt:Tst+T_tr;   % represent the trainng data

for i = 1:length(TR_C)
    histvec_int = TR_C{i};
    hist_int = @(t) histvec_int*cos(t);
    sol = dde23(trueModel, truetau, hist_int, 0:dt:tend);
    clean_data_t = 0:dt:tend;
    clean_data{i} = deval(sol, clean_data_t);
    % Add history
    % clean_data{i} = [repmat(histvec_int, 1, num_hist_points), clean_data{i}];
    clean_data{i} = [hist_int(Tst:dt:-dt), clean_data{i}];
    clean_data_t = Tst:dt:tend;

    xdata{i} = deval(sol, 0:dt:tend);
    % xdata_all{i} = [repmat(histvec_int, 1, num_hist_points), xdata{i}];
    xdata_all{i} = [hist_int(Tst:dt:-dt), xdata{i}];
    targets_all{i} = xdata{i};
    % targets_dx{i} = ctrDiff(xdata_all{i}, tt_target);
    targets_dx{i} = ctrDiff(targets_all{i}, tt_target);
end

figure(1)
set(gcf,'Position',[100 100 500 220])
hold on;
for i = 1:length(xdata)
    plot(clean_data_t,clean_data{i}(:),'b-')
    scatter(tt_all,xdata_all{i}(:),5,'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
end
plot([Tst+tau_max Tst+tau_max],[0 max(clean_data{i})+1],'k--')
plot([Tst Tst],[0 max(clean_data{i})+1],'k-')
plot([Tst+T_tr Tst+T_tr],[0 max(clean_data{i})+1],'k-')
box on;
title('Delay Logistic Equation')
xlim([Tst tend])
ylim([min(clean_data{i}) max(clean_data{i})+1])
ylabel('$x(t)$')
xlabel('$t$')


figure(10)
set(gcf,'Position',[100 100 300 250])
hold on;
for i = 1:length(xdata)
    plot(clean_data{i}(1:round((T_tr-truetau)/dt)),clean_data{i}(round(truetau/dt)+1:round((T_tr)/dt)),'b-')
    scatter(xdata_all{i}(1:round((T_tr-truetau)/dt)),xdata_all{i}(round(truetau/dt)+1:round((T_tr)/dt)),10,...
        'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
end
box on;
axis equal
xlabel('$x(t-\tau)$')
ylabel('$x(t)$')
% xlim([0.2 1.4])
% ylim([0.2 1.4])
% yticks([0.2 0.8 1.4])
% xticks([0.2 0.8 1.4])

%% learning 
hiddenSize = 10;  % 2 may also work
nx = 1; % 1 state
% tau = dlarray(tau_max*rand(1));
tau = dlarray(0.5);
nd = length(tau);

NDDE = struct;
NDDE.fc1 = struct;
sz = [hiddenSize nx*(1+nd)];
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
        tt = 0:dt:dt*steps;       % the sampling time
        t_simu = 0:(dt/gap):dt*steps; % simulation time step
        hist_t = -tau_max:dt:0;
        st = randperm(length(t_deriv)-steps,batchSize);
        for kk = 1:nsets
            for batch = 1:batchSize
                hist_v{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch):st(batch)+length(hist_t)-1);
                dlhist_v{(kk-1)*batchSize+batch} = dlarray(hist_v{(kk-1)*batchSize+batch});
                targets_tr{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch)...
                    +length(hist_t)-1:st(batch)+length(hist_t)-1+steps);
            end
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddesimu,tt,t_simu,NDDE,tau,dlhist_v,hist_t,targets_tr,lm,st,w);
        % % Display gradient norms
        % % disp(['Gradients Norm for Tau: ', num2str(norm(extractdata(grads_tau)))]);
        % if mod(iter, plotFrequency) == 0 || iter == 1
        %     fprintf('Iteration %d: Loss = %.6f, Tau = [%.4f, %.4f]\n', iter, loss, extractdata(tau(1)), extractdata(tau(2)));
        %     % Plot gradients norms
        %     gradNormTheta = norm(extractdata(grads_NDDE.fc1.Weights)) + norm(extractdata(grads_NDDE.fc2.Weights)) + norm(extractdata(grads_NDDE.fc3.Weights));
        %     gradNormTau = norm(extractdata(grads_tau));
        %     fprintf('Gradient Norms: Theta = %.6f, Tau = %.6f\n', gradNormTheta, gradNormTau);
        % end

    elseif strcmp(lossfun,'deriv')
        targets_dx_tr = cell(1,nsets);
        st = randperm(length(t_deriv),batchSize);
        t_batch = t_deriv(st);

        for kk = 1:nsets
            % targets_dx_tr{kk} = targets_dx{kk}(st);
            targets_dx_tr{kk} = targets_dx{kk}(:, st);
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddederiv,tt_all,t_batch,NDDE,tau,xdata_all,targets_dx_tr,lm,st,w);

        % % Display gradient norms
        % % disp(['Gradients Norm for Tau: ', num2str(norm(extractdata(grads_tau)))]);
        % if mod(iter, plotFrequency) == 0 || iter == 1
        %     fprintf('Iteration %d: Loss = %.6f, Tau = [%.4f, %.4f]\n', iter, Loss(iter), extractdata(tau(1)), extractdata(tau(2)));
        % 
        %     % Plot gradients norms
        %     gradNormTheta = norm(extractdata(grads_NDDE.fc1.Weights)) + norm(extractdata(grads_NDDE.fc2.Weights)) + norm(extractdata(grads_NDDE.fc3.Weights));
        %     gradNormTau = norm(extractdata(grads_tau));
        %     fprintf('Gradient Norms: Theta = %.6f, Tau = %.6f\n', gradNormTheta, gradNormTau);
        % end
    elseif strcmp(lossfun,'comb')
        hist_v = cell(1,nsets*batchSize);
        dlhist_v = cell(1,nsets*batchSize);
        targets= cell(1,nsets*batchSize);
        tt = 0:dt:dt*steps;
        t_simu = 0:(dt/gap):dt*steps; % simulation time step
        hist_t = -tau_max:dt:0;
        st1 = randperm(length(t_deriv)-steps,batchSize);
        for kk = 1:nsets
            for batch = 1:batchSize
                hist_v{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st1(batch):st1(batch)+length(hist_t)-1);
                dlhist_v{(kk-1)*batchSize+batch} = dlarray(hist_v{(kk-1)*batchSize+batch});
                targets{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st1(batch)...
                    +length(hist_t)-1:st1(batch)+length(hist_t)-1+steps);
            end
        end

        targets_dx_tr = cell(1,nsets);
        st2 = randperm(length(t_deriv),batchSize2);
        t_batch = t_deriv(st2);
        for kk = 1:nsets
            targets_dx_tr{kk} = targets_dx{kk}(:,st2);
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddecomb,tt_all,t_batch,...
            tt,t_simu,NDDE,tau,xdata_all,targets_dx_tr,...
            dlhist_v,hist_t,targets,lm,st1,st2,w);
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

    % Inside your training loop
    if mod(iter,plotFrequency) == 0  || iter == 1 % nn == 0 % mod(iter,plotFrequency) == 0  || iter == 1
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
%%
        figure(2)
        set(gcf,'Position',[150 150 1000 300])
        clf
        subplot(2,3,1)
        semilogy(1:iter, Loss(1:iter),'b')
        xlabel('Iteration')
        xlim([1,iter+1])
        ylabel('Loss')
        title("Training loss")

        subplot(2,3,4)
        hold on
        for k = 1:nd
            plot(1:iter, Tau_tr(k,1:iter),'b','LineWidth',2)
        end
        % plot(1:iter+1, truetau*ones(1,iter+1), 'k--',LineWidth=2)
        % Plot each delay value in `truetau`
        for k = 1:length(truetau)
            plot(1:iter+1, truetau(k) * ones(1, iter+1), 'k--', 'LineWidth', 2);
        end

        ylim([0,tau_max])
        xlim([1,iter+1])
        xlabel('Iteration')
        ylabel('$\tau$')
        hold off
        box on
        title("Learned delay")

        subplot(2,3,2)
        % simulation
        W1 = double(extractdata(NDDE.fc1.Weights));
        W2 = double(extractdata(NDDE.fc2.Weights));
        W3 = double(extractdata(NDDE.fc3.Weights));
        b1 = double(extractdata(NDDE.fc1.Bias));
        b2 = double(extractdata(NDDE.fc2.Bias));
        NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[x;reshape(xdelay,[],1)]+b1)+b2);
        tt_target_fine=Tst+tau_max:dt:tend;
        % tt_target_fine = Tst : 0.01 : tend;  % Start from 0

        % predict the whole traj
        sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target);
        y = deval(sol1,tt_target_fine);
        % y = dl_mdde_ab4(@(t,x,xdelay,theta)ddeModel(t,x,xdelay,theta),NDDE,tau,@(t)interp1(tt_all,xdata_all{1}',t)',tt_target_fine);

        
        hold on
        plot(tt_target_fine,y,'b','LineWidth',1)
        plot(hist_time,xdata_all{1}(:,1:length(hist_time)),'b')
        scatter(tt_target,targets_all{1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time,xdata_all{1}(:,1:length(hist_time)),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        yLimits = ylim;
        plot([Tst+T_tr Tst+T_tr],yLimits,'k--', 'LineWidth',1)
        xlim([hist_time(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        % ylim([0 1.5])
        title('Network simulation')

        subplot(2,3,5)
        hold on
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine-tau(k))',nx,[])];
        end
        dx = ddeModel(tt_target_fine,x,xdelay,NDDE);
        plot(tt_target_fine,extractdata(dx),'b')
        scatter(tt_target,targets_dx{1,1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst + T_tr, Tst + T_tr], ylim, 'k--', 'LineWidth', 1)

        xlim([hist_time(1) tend])
        % ylim([-2 2])
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on
        subplot(2,3,3)
        % predict only for testing
        hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
        tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
        tt_target_ts_fine = Tst+T_tr:dt:Tst+T_tr+T_ts;
        sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);       
        y = deval(sol1,tt_target_ts_fine);
        % y = dl_mdde_ab4(@(t,x,xdelay,theta)ddeModel(t,x,xdelay,theta),NDDE,tau,@(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts_fine);

        hold on
        plot(tt_target_ts_fine ,y,'b','LineWidth',1)
        plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
        scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        %         % Scatter plot of the first state
        % scatter(tt_target, targets_dx{1}(1, :), 5, 'filled', 'MarkerEdgeColor', "#D95319", 'MarkerFaceColor', "#D95319");
        yLimits = ylim;
        plot([Tst+T_tr Tst+T_tr],yLimits,'k--', 'LineWidth',1)
        % plot([Tst+T_tr Tst+T_tr],[0 1.5],'k--', 'LineWidth',1)
        % xlim([hist_time_ts(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        % ylim([0 1.5])
        title('Network simulation')

        subplot(2,3,6)
        hold on
        % only testing
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
        % x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            % xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts-tau(k))',nx,[])];
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
        end

        % dxcheck=extractdata(dx);
        % dxtragetcheck=targets_dx{1};
        % return
        % dx = ddeModel(tt_target_ts ,x,xdelay,NDDE);
        dx = ddeModel(tt_target_ts_fine ,x,xdelay,NDDE);

        % plot(tt_target_ts,extractdata(dx),'b')
        plot(tt_target_ts_fine,extractdata(dx),'b')
        % scatter(tt_target_ts,targets_dx{1,3}(:,end-length(tt_target_ts)+1:end),5,"filled",'MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,"filled",'MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[-2 2],'k--', 'LineWidth',1)
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on
%%
    end

    [NDDE,aveGrad,aveSqGrad] = adamupdate(NDDE,grads_NDDE,aveGrad,aveSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    [tau,aveGrad_tau,aveSqGrad_tau] = adamupdate(tau,grads_tau,aveGrad_tau,aveSqGrad_tau,iter,...
        learnRate_tau,gradDecay,sqGradDecay);
    tau = min(max(0.00001,tau),tau_max-0.00001);
end

%% plot loss and delay path
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
NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[x;reshape(xdelay,[],1)]+b1)+b2);

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
for k = 1:length(truetau)
    plot(1:iter+1, truetau(k) * ones(1, iter+1), 'k--', 'LineWidth', 2);
end
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
hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
    @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);
tt_target_ts_fine = Tst+T_tr:dt:Tst+T_tr+T_ts;
y = deval(sol1,tt_target_ts_fine );
hold on
plot(tt_target_ts_fine ,y,'r','LineWidth',1)
plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[0 max(clean_data{i})+1],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
ylim([min(clean_data{i}) max(clean_data{i})+1])
xlabel('$t$')
ylabel('$x$')
hold off
box on
title('Network simulation')


figure(11)
set(gcf,'Position',[100 100 300 200])
hold on
x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
xdelay = [];
for k = 1:length(tau)
    xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
end
dx = ddeModel(tt_target_ts_fine ,x,xdelay,NDDE);
plot(tt_target_ts_fine,extractdata(dx),'r')
scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,...
    "filled",'MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[min(targets_dx{i})-1 max(targets_dx{i})+1],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
ylim([min(targets_dx{i}) max(targets_dx{i})+1])
xlabel('$t$')
ylabel('$\dot{x}$')
title('Network direct prediction')
hold off
box on

%% get error on testing data
y_test = deval(sol1,tt_target_ts);
L_simu=sum((y_test-targets_all{1}(:,end-length(tt_target_ts)+1:end)).^2)/length(tt_target_ts);
x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts)',nx,[]);
xdelay = [];
for k = 1:length(tau)
    xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts-tau(k))',nx,[])];
end
dy_test = ddeModel(tt_target_ts,x,xdelay,NDDE);
L_deriv=sum((dy_test-targets_dx{1}(:,end-length(tt_target_ts)+1:end)).^2)/length(tt_target_ts);
RMSE_deriv = rmse(dy_test,targets_dx{1}(:,end-length(tt_target_ts)+1:end))
RMSE_simu = rmse(y_test,targets_all{1}(:,end-length(tt_target_ts)+1:end))



