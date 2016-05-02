gau_precision = 0.005
num_inits = 1
nb_iterations = int64(5)
batchsize = 64
% train_func_name = 'learn_RBF_SGD'
% mdl_func_name = 'RBF'
lambda = 0
eta_c = 0.001
eta_t = 0.001
eta_beta = 0.001
visualize = 0
sgd_errors = 1
%% locations
F_func_name = 'F_NO_activation_final_layer'
%F_func_name = 'F_activation_final_layer'
train_func_name = 'learn_HBF1_SGD'
mdl_func_name = 'HBF1'
cp_folder = 'cp_2apr_ht1_HBF1/'
cp_param_files_names = 'cp_2apr_ht1_HBF1_%d.m'
results_path = './results/r_2apr_ht1_HBF1/'

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
data_set_path = '../../hbf_research_data/data_MNIST_0.7_0.15_0.15_49000_10500_10500.mat'
data_normalized = 0
%% GPU
gpu_on = 0
%% t_initilization
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