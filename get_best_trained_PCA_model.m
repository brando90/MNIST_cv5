function [] = get_best_trained_PCA_model(slurm_job_id, task_id, coeff, mu)
if nargin == 3
    error('need to provide both arguments pca_coeff and mu for PCA model');
end
if nargin == 4
    disp('got pca_coeff and mu as an argument')
    U = coeff;
end
vname=@(x) inputname(1);
restoredefaultpath
slurm_job_id
task_id
run('./simulation_config_pca.m');
run('load_paths_multilayer.m');
%% load configs
current_simulation_config = sprintf( './changing_params/%s%s', cp_folder, 'simulation_config_pca.m' )
run(current_simulation_config);
changing_params_for_current_task = sprintf( sprintf('./changing_params/%s%s',cp_folder,cp_param_files_names), task_id )
run(changing_params_for_current_task);
%% load data set
load(data_set_path);
[N_train, D] = size(X_train) % (N_train x D)
[N_test, D] = size(X_test) % (N_test x D)
%% preparing models to train/test for mdl_iterator
K = center;
%% Learn PCA
iterations = 1; % silly variable, but for completeness here it is
tic;
if load_pca
    load(pca_mdl_loc)
end
X_train = X_train'; % (D x N) = (N_train x D)'
X_test = X_test'; % (D x N) = (N_test x D)'
U = U(:,1:K); % (D x K) = K pca's of dimension D
X_tilde_train = (U * U' * X_train); % (D x N_train)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
X_tilde_test = (U * U' * X_test); % (D x N_test)= (D x N)' = ( (D x K) x (K x D) x (D x N) )'
%% Errors of models
train_error_PCA = (1/N_train)*norm( X_tilde_train - X_train ,'fro')^2
test_error_PCA = (1/N_test)*norm( X_tilde_test - X_test ,'fro')^2
%pca_mdl = pca_mdl.gather();
%% save everything/write errors during iterations
error_iterations_file_name = sprintf('test_error_vs_iterations%d',task_id);
path_error_iterations = sprintf('%s%s',results_path,error_iterations_file_name)
save(path_error_iterations, vname(iterations), vname(train_error_PCA),vname(test_error_PCA), vname(center), vname(coeff), vname(U), vname(mu) );
%% write results to file
result_file_name = sprintf('results_om_id%d.m',task_id);
result_path_file = sprintf('%s%s',results_path,result_file_name)
[fileID,~] = fopen(result_path_file, 'w')
fprintf(fileID, 'task_id=%d;\ncenter=%d;\ntest_error_PCA=%d;\ntrain_error_PCA=%d;', task_id,center,test_error_PCA,train_error_PCA);
time_passed = toc;
%% write time elapsed to file
[secs, minutes, hours, ~] = time_elapsed(iterations, time_passed )
time_file_name = sprintf('time_duration_om_id%d.m',task_id);
path_file = sprintf('%s%s',results_path,time_file_name);
fileID = fopen(path_file, 'w')
fprintf(fileID, 'task_id=%d;\nsecs=%d;\nminutes=%d;\nhours=%d;\niterations=%d;\ncenter=%d;\ndata_set= ''%s'' ;', task_id,secs,minutes,hours,iterations,center,data_set_path);
disp('DONE');
disp('DONE training model')
end