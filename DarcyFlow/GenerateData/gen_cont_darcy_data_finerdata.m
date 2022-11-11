clear; close all; clc

% define parameters
K1 = 256;
K2 = 64;
N = 1;

% define parameters for random field a and F
alpha = 2;
tau = 3;
F = ones(K1,K1);

% define solution grid
[X1,Y1] = meshgrid(1/(2*K1):1/K1:(2*K1-1)/(2*K1),1/(2*K1):1/K1:(2*K1-1)/(2*K1));

% define measurement grid
[X2,Y2] = meshgrid(0:1/(K2-1):1,0:1/(K2-1):1);

% define arrays to store results
coeff = zeros(N,K1,K1);
sol = zeros(N,K2,K2);

for j=1:N
    tic
    % generate log-normal input
    a = exp(gaussrnd(alpha,tau,K1));
    u = solve_gwf(a,F);
    %figure; contour(X1,Y1,u); hold on; plot(X2,Y2,'.r','MarkerSize',10)
    u = interp2(X1,Y1,u,X2,Y2,'spline');
    % save results
    coeff(j,:,:) = a;
    sol(j,:,:) = u;
    disp(j);
    toc
end

% reduce dimension of sol
sol = sol(:,4:8:end,4:8:end);

% take log - gaussian field
x = log(coeff);
y = sol;

% save final
save('darcy_data_finerdata','x','y','-v7.3')
