%% PCA
clear;restoredefaultpath;clear;
tic;
%% load data set
run('./simulation_config_pca.m');
load(data_set_path); % data4cv
%%
slurm_job_id = randi([0 2^31],1,1);
jobs_to_run = jobs; %comes from config file
%% Do PCA 
% Each column of COEFF contains coefficients for one principal component.
[coeff, ~, ~, ~, ~, mu] = pca(X_train); % (D x R) = Rows of X_train correspond to observations and columns to variables. 
for task_id=1:jobs_to_run
    get_best_trained_PCA_model(slurm_job_id, task_id, coeff, mu)
end
time_passed = toc;
run('load_paths_multilayer.m');
iterations = 1;
[secs, minutes, hours, ~] = time_elapsed(iterations, time_passed )
beep;
disp('DONE getting all PCA errors')