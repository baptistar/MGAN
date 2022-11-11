clear; close all; clc

% define parameters
K1 = 256;
N = 1000;
alpha = 2;
tau = 3;

% generate fields
coeff = zeros(N,K1,K1);
for j=1:N
    tic
    coeff(j,:,:) = gaussrnd(alpha,tau,K1);
end

% plot mean and variance
coeff_r = reshape(exp(coeff),N,K1*K1);
figure;
imagesc(reshape(mean(coeff_r,1),K1,K1)); colorbar;
figure;
imagesc(reshape(var(coeff_r,[],1),K1,K1)); colorbar;