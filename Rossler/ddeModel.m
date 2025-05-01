function dx = ddeModel(t, x, xdelay, theta)
nx = 3;  % for Rossler
if size(theta.fc1.Weights,2)-nx == 2
    % xdelay = xdelay(1,:);
    xdelay = xdelay([1 4],:);
elseif size(theta.fc1.Weights,2)-nx == 1
    xdelay = xdelay(1,:);
end
input = [x; xdelay];  % Size: [dim, N]
dx = theta.fc3.Weights * tanh(theta.fc2.Weights * tanh(theta.fc1.Weights * input + theta.fc1.Bias) + theta.fc2.Bias);

end
