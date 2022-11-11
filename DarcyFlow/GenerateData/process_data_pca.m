clear; close all; clc

% define latent dimensions
latent_dim = [12,25,50,100,200];

% load dataset
load('darcy_data.mat','coeff','sol')

% reduce dimension of sol
y = sol(:,4:8:end,4:8:end);

% convert to gaussian field
x = log(coeff);

% remove column mean
x_flat = reshape(x, size(x,1), size(x,2)*size(x,3));
x_mean = mean(x_flat,1);
x_flat = (x_flat - x_mean);

% project coeff inputs
[~, S_vals, x_svec] = svds(x_flat, max(latent_dim));

% project data
for i=1:length(latent_dim)

    % extract sigular vectors
    ldim = latent_dim(i);
    x_svecr  = x_svec(:,1:ldim);
    x_svalsr = diag(S_vals(1:ldim,1:ldim));

    % compute coeff_score
    x_score = x_flat * x_svecr;

    % save results
    save(['darcy_data_noiseless_latentdim' num2str(ldim) '.mat'], ...
        'x_score','y','x_svecr','x_svalsr','x_mean');

end
