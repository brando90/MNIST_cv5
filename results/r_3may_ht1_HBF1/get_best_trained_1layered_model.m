function [] = get_best_trained_1layered_model(slurm_job_id, task_id)
restoredefaultpath
slurm_job_id
task_id
%% load the path to configs for this simulation
run('./simulation_config.m');
run('load_paths_multilayer.m');
%% load most of the configs for this model
current_simulation_config = sprintf( './changing_params/%s%s', cp_folder, 'simulation_config.m' )
run(current_simulation_config);
%% load the number of centers for this model
changing_params_for_current_task = sprintf( sprintf('./changing_params/%s%s',cp_folder,cp_param_files_names), task_id )
run(changing_params_for_current_task);
%% load data set
data_set_path = sprintf('../../hbf_research_data/%s',data_set_file_name);
load(data_set_path); % data4cv
if data_normalized
    error('TODO');
end
switch task
    case 'autoencoder'
        Y_train = X_train;
        Y_test = X_test;
    case 'regression'
        %leave data set as is.
    case 'classification'
        %leave data set as is.
    otherwise
        error('task not defined');
end
%% rand seed
rand_seed = get_rand_seed( slurm_job_id, task_id)
rng(rand_seed); %rand_gen.Seed
%% model
gauss_func = @(S) exp(S);
dGauss_ds = @(A) A;

relu_func = @(A) max(0,A);
dRelu_ds = @(A) A > 0;

sigmoid_func = @(A) sigmf(A, [1, 0]);
dSigmoid_ds = @(A) A .* (1 - A);

tanh_func = @(A) tanh(A);
dTanh_ds = @(A) 1 - A.^2;

Identity = @(A) A;
dIdentity_ds = @(A) ones(size(A));
%% preapre model
h_mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
kernel_mdl = struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
switch train_func_name
    case 'learn_HBF1_SGD'
        Act = gauss_func;
        dAct_ds = dGauss_ds;
    case 'learn_HReLu_SGD'
        Act = relu_func;
        dAct_ds = dRelu_ds;
    case 'learn_HSig_SGD'
        Act = sigmoid_func;
        dAct_ds = dSigmoid_ds;
    case 'learn_HTanh_SGD'
        Act = tanh_func;
        dAct_ds = dTanh_ds;
    otherwise
        disp('OTHERWISE');
        error('The train function you gave: %s does not exist', train_func_name);
end
for l =1:L-1
    kernel_mdl(l).Act = Act;
    kernel_mdl(l).dAct_ds = dAct_ds;
    kernel_mdl(l).beta = gau_precision;
    kernel_mdl(l).lambda = lambda;
    h_mdl(l).Act = Act;
    h_mdl(l).dAct_ds = dAct_ds;
    h_mdl(l).beta = gau_precision;
    h_mdl(l).lambda = lambda;
end
kernel_mdl(1).F = @F;
h_mdl(1).F = @F;
switch F_func_name
    case 'F_NO_activation_final_layer'
        h_mdl(L).Act = Identity;
        h_mdl(L).dAct_ds = dIdentity_ds;
        kernel_mdl(L).Act = Identity;
        kernel_mdl(L).dAct_ds = dIdentity_ds;
        kernel_mdl(l).beta = gau_precision;
        kernel_mdl(l).lambda = lambda;
    case 'F_activation_final_layer'
        h_mdl(L).Act = Act;
        h_mdl(L).dAct_ds = dAct_ds;
        kernel_mdl(L).Act = Identity;
        kernel_mdl(L).dAct_ds = dIdentity_ds;
        h_mdl(l).beta = gau_precision;
        h_mdl(l).lambda = lambda;     
end
tic;
if gpu_on
    X_train = gpuArray(X_train);
    Y_train = gpuArray(Y_train);
    X_test = gpuArray(X_test);
    Y_test = gpuArray(Y_test);
end
K = center;
D_out = size(Y_train,2);
%% statistics of data
y_std = std(Y_train,0,2); % (D_out x 1) unbiased std of coordinate/var/feature
y_mean = mean(Y_train,2); % (D_out x 1) mean of coordinate/var/feature
y_std = repmat( y_std', [K,1]); % (K x D_out) for c = (K x D_out)
y_mean = repmat( y_mean', [K,1]); % (K x D_out) for c = (K x D_out)  
%% get errors of all initilization models
mdl_error_train_all_inits = zeros(nb_inits,1); % (nb_inits x 1)
mdl_error_test_all_inits = zeros(nb_inits,1); % (nb_inits x 1)
error_train_all_iterations = zeros(nb_inits,nb_iterations+1); % (nb_inits x nb_iterations)
error_test_all_iterations = zeros(nb_inits,nb_iterations+1); % (nb_inits x nb_iterations)
all_kernel_models = cell([nb_inits,1]);
all_h_mdl_models = cell([nb_inits,1]);
for init_index=1:nb_inits
    fprintf('initialization_index: %s\n', init_index)
    %% t_initilization
    fprintf('t_initilization: %s\n', t_initilization);
    switch t_initilization
    case 't_random_data_points'
        % selects K rows selected from X_train
        t_init = datasample(X_train, K, 'Replace', false)'; % (D x K)
        b_init_1 = normrnd(0,epsilon_t,[1,K]);
        b_init_2 = normrnd(0,epsilon_t,[1,D_out]);
    case 't_zeros_plus_eps'
        t_init = normrnd(0,epsilon_t,[D,K]); % (D x K)
        b_init_1 = normrnd(0,epsilon_t,[1,K]);
        b_init_2 = normrnd(0,epsilon_t,[1,D_out]);
    case 't_random_data_points_treat_offset_special'
        error('TODO');
    otherwise
        error('The t wieghts init that you gave is invalid var t_initilization: %s', t_initilization);
    end
    %% put in GPU
    c_init = (1 + 1)*rand(K,D_out);
    if gpu_on
        c_init = gpuArray(c_init);
        t_init = gpuArray(t_init);
        b_init_1 = gpuArray(b_init_1);
        b_init_2 = gpuArray(b_init_2);
    end
    %% train kernel model
    kernel_mdl(1).b = b_init_1;
    kernel_mdl(1).W = t_init;
    kernel_mdl(2).b = b_init_2;
    kernel_mdl(2).W = c_init; 
    switch train_func_name
        case 'learn_HBF1_SGD'
            fp = kernel_mdl(1).F(kernel_mdl, X_train); %centers are fixed
            Kern = fp(1).A; % (K x D) = (N x K)' x (N x D)
            %kernel_mdl.c = Kern \ Y_train;
            kernel_mdl(2).W = Kern \ Y_train;
            c_init = kernel_mdl(2).W; % (K x D)
        case 'learn_HModel_SGD'
            fp = kernel_mdl(1).F(kernel_mdl, X_train); %centers are fixed
            Kern = fp(1).A; % (K x D) = (N x K)' x (N x D)
            kernel_mdl(2).W = Kern \ Y_train';
            c_init = kernel_mdl(2).W; % (K x D)
            
        otherwise
            disp('OTHERWISE');
            error('The train function you gave: %s does not exist', train_func_name);
    end
    %% c_initilization H model
    fprintf('c_initilization: %s\n', c_initilization);
    switch c_initilization
    case 'c_kernel_mdl_as_initilization'
        % Get c = K\Y (soln to K*c = Y )
        switch train_func_name
            case 'learn_HBF1_SGD'
                c_init = kernel_mdl(2).W; % (K x D)
            case 'learn_HModel_SGD'
                c_init = kernel_mdl(2).W; % (K x D)
            otherwise
                disp('OTHERWISE');
                error('The train function you gave: %s does not exist', train_func_name);
         end
    case 'c_normal_zeros_plus_eps'
        c_init = normrnd(0,epsilon_c,[K,D_out]); % (K x D_out)
    case 'c_normal_ymean_ystd'
        c_init = normrnd(y_mean,y_std,[K,D_out]); % (K x D_out)
    case 'c_uniform_random_centered_ymean_std_ystd'
        c_init = (y_std + y_std) .* rand(K,D_out) + y_mean;
    case 'c_uniform_random_centered_ymean_std_min_max_y'
        %c_init = repmat(max_y - min_y,[K,1]) .* rand(K,D_out) + repmat(y_mean,[K,1]);
        error('TODO');
    case 'c_hard_coded_c_init'
        c_init = (1 + 1)*rand(K,D_out) - 1;
    otherwise
        disp('OTHERWISE')
    end
    if c_init_normalized && ~strcmp( train_func_name, 'c_kernel_mdl_as_initilization') 
        %IF its c_init is suppose to be normalized AND its NOT a kernel c_init
        c_init = normc(c_init); %NOTE: doesn't make sense to normalize the kernel c weights
    end
    %% train H mdl
    switch train_func_name
    case 'learn_HBF1_SGD'
         h_mdl(1).W = t_init;
         h_mdl(1).b = zeros([1,K]);
         h_mdl(2).W = c_init;
         h_mdl(1).b = zeros([1,D_out]);
         for l=1:L
             h_mdl(l).beta = gau_precision;
             h_mdl(l).lambda = lambda;
         end
         [ h_mdl, iteration_errors_train, iteration_errors_test ] = multilayer_learn_HBF_MiniBatchSGD( X_train, Y_train, h_mdl, nb_iterations, batchsize, X_test,Y_test, step_size_params, sgd_errors);
    case 'learn_HReLu_SGD'  
        b_init_1 = normrnd(0,epsilon_t,[1,K]);
        b_init_2 = normrnd(0,epsilon_t,[1,D_out]);
        h_mdl(1).W = t_init;
        h_mdl(1).b = b_init_1;
        h_mdl(2).W = c_init;
        h_mdl(1).b = b_init_2;
        for l=1:L
            h_mdl(l).beta = gau_precision;
            h_mdl(l).lambda = lambda;
        end
        [ h_mdl, iteration_errors_train, iteration_errors_test ] = multilayer_learn_HModel_explicit_b_MiniBatchSGD( X_train,Y_train, h_mdl, nb_iterations,batchsize, X_test,Y_test, step_size_params, sgd_errors );
        %[ h_mdl, iteration_errors_train, iteration_errors_test ] = learn_HModel_MiniBatchSGD( X_train, Y_train, mdl, nb_iterations,visualize, X_test,Y_test, eta_c, eta_t, sgd_errors);
    otherwise
       error('The train function you gave: %s does not exist', train_func_name);
    end
    %% Collect model information
    mdl_error_train_all_inits(init_index) = compute_Hf_sq_error(X_train, Y_train, h_mdl );
    mdl_error_test_all_inits(init_index) = compute_Hf_sq_error(X_test, Y_test, h_mdl );
    all_kernel_models{init_index} = kernel_mdl;
    all_h_mdl_models{init_index} = h_mdl;
    error_train_all_iterations(init_index, :) = iteration_errors_train;
    error_test_all_iterations(init_index, :) = iteration_errors_test;
end
time_passed = toc;
%% Errors of models Gather from GPU
%struct('W', cell(1,L),'b', cell(1,L),'F', cell(1,L), 'Act',cell(1,L),'dAct_ds',cell(1,L),'lambda', cell(1,L), 'beta', cell(1,L));
for i=1:nb_inits
    kernel_mdl = all_kernel_models{i};
    h_mdl = all_h_mdl_models{i};
    for l=1:L
        kernel_mdl(l).W = gather(kernel_mdl(l).W);
        kernel_mdl(l).b = gather(kernel_mdl(l).b);
        kernel_mdl(l).lambda = gather(kernel_mdl(l).lambda);
        kernel_mdl(l).beta = gather(kernel_mdl(l).beta);
        h_mdl(l).W = gather(h_mdl(l).W);
        h_mdl(l).b = gather(h_mdl(l).b);
        h_mdl(l).lambda = gather(h_mdl(l).lambda);
        h_mdl(l).beta = gather(h_mdl(l).beta);
    end
end
%% save everything/write errors during iterations
[s,git_hash_string_mnist_cv4] = system('git -C . rev-parse HEAD')
[s,git_hash_string_hbf_research_data] = system('git -C ../../hbf_research_data rev-parse HEAD')
[s,git_hash_string_hbf_research_ml_model_library] = system('git -C ../../hbf_research_ml_model_library rev-parse HEAD')
vname=@(x) inputname(1);

file_name_mdl_error_train_test_all_inits = sprintf('error_train_test_all_inits%d',task_id);
path_mdl_error_test_train_all_inits  = sprintf('%s%s',results_path,file_name_mdl_error_train_test_all_inits)
save(path_mdl_error_test_train_all_inits, vname(mdl_error_train_all_inits), vname(mdl_error_test_all_inits) );

file_name_mdl_error_iterations = sprintf('mdl_error_iterations%d',task_id);
path_mdl_error_iterations = sprintf('%s%s',results_path,file_name_mdl_error_iterations)
save(path_mdl_error_iterations, vname(error_train_all_iterations), vname(error_test_all_iterations) );

file_name_mdl_all_mdls_inits = sprintf('all_mdls_inits%d',task_id);
path_mdl_all_mdls_inits = sprintf('%s%s',results_path,file_name_mdl_all_mdls_inits)
save(path_mdl_all_mdls_inits, vname(all_kernel_models), vname(all_h_mdl_models), vname(rand_seed), vname(nb_iterations), vname(center));

file_name_git_info = sprintf('git_info%d',task_id);
path_git_info = sprintf('%s%s',results_path,file_name_git_info)
save(path_git_info, vname(git_hash_string_mnist_cv4), vname(git_hash_string_hbf_research_data), vname(git_hash_string_hbf_research_ml_model_library) );
%% save my own code
my_self = 'get_best_trained_1layered_model.m';
source = sprintf('./%s', my_self);
destination = sprintf('%s', results_path)
copyfile(source, destination);
%% write time elapsed to file
[secs, minutes, hours, ~] = time_elapsed(nb_iterations, time_passed )
time_file_name = sprintf('time_duration_om_id%d.m',task_id);
path_file = sprintf('%s%s',results_path,time_file_name);
fileID = fopen(path_file, 'w')
fprintf(fileID, 'task_id=%d;\nsecs=%d;\nminutes=%d;\nhours=%d;\nnb_iterations=%d;\ncenter=%d;\ndata_set= ''%s'' ;', task_id,secs,minutes,hours,nb_iterations,center,data_set_path);
disp('DONE');
disp('DONE training model')
end