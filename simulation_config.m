gau_precision = 0.005
nb_inits = 1
nb_iterations = int64(5) % nb_iterations
batchsize = 64
L = 2; %nb_layers
% train_func_name = 'learn_RBF_SGD'
% mdl_func_name = 'RBF'
lambda = 0
step_size_params =  struct('eta_c', cell(1), 'eta_t', cell(1), ...
    'AdaGrad', cell(1), 'Momentum', cell(1), ...
    'Decaying', cell(1), 'step_size', cell(1) );
step_size_params.eta_c = 0.001;
step_size_params.eta_t = 0.001;
step_size_params.eta_beta = 0.001;

step_size_params.AdaGrad = 0;
step_size_params.Momentum = 0;

step_size_params.Decaying = 1;
step_size_params.step_size = 0.01;
%% collect SGD iteration errors
sgd_errors = 1
%% locations
F_func_name = 'F_NO_activation_final_layer'
%F_func_name = 'F_activation_final_layer'
train_func_name = 'learn_HBF1_SGD'
mdl_func_name = 'HBF1'

cp_folder = 'cp_13may_ht1_HBF1/'
cp_param_files_names = 'cp_13may_ht1_HBF1_%d.m'
results_path = './results/r_3may_ht1_HBF1/'

% train_func_name = 'learn_HSig_SGD'
% mdl_func_name = 'HSig'
% cp_folder = 'cp_29mar_ht2_HSig/'
% cp_param_files_names = 'cp_29mar_ht2_HSig_%d.m'
% results_path = './results/r_29mar_ht2_HSig/'

% train_func_name = 'learn_HReLu_SGD'
% mdl_func_name = 'HReLu'
% cp_folder = 'cp_29mar_ht2_HReLu/'
% cp_param_files_names = 'cp_29mar_ht2_HReLu_%d.m'
% results_path = './results/r_29mar_ht2_HReLu/'
%% jobs
jobs = 2
start_centers = 10
end_centers = 250
%% data
task = 'autoencoder';
data_set_file_name = 'data_MNIST_original_minist_60k_10k_split_train_test'
data_normalized = 0
%% GPU
gpu_on = 0
%% t_initilization
epsilon_t = 0.01;
t_initilization = 't_random_data_points' %datasample(X_train', K, 'Replace', false)';

%epsilon_t = 0.01;
%t_initilization = 't_zeros_plus_eps' %normrnd(0,epsilon,[K,D]);

%epsilon_t = 0.1;
%t_initilization = 't_random_data_points_treat_offset_special' %-normrnd(epsilon*t_mean - epsilon*t_std, t_std);
%% c_initilization
c_initilization = 'c_kernel_mdl_as_initilization' % c_init = kernel_mdl.c;

%epsilon_c = 0.01;
%c_initilization = 'c_normal_zeros_plus_eps' %normrnd(0,epsilon,[K,D_out]);

%c_initilization = 'c_normal_ymean_ystd' %normrnd(y_mean,y_std,[K,D_out]);
%c_initilization = 'c_uniform_random_centered_ymean_std_ystd' %(y_std + y_std) .* rand(K,D_out) + y_mean;
%c_initilization = 'c_uniform_random_centered_ymean_std_min_max_y' %TODO
%c_initilization = 'c_hard_coded_c_init' %(1 + 1)*rand(K,D_out) - 1
%% normalized
c_init_normalized = 0