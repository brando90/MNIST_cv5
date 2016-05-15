%% PCA
train_func_name = 'PCA' %DOESN'T actually do anything, not set to anything, load_paths will crash
%% locations
cp_folder = 'cp_14may_ht1_pca/'
cp_param_files_names = 'cp_14may_ht1_pca_%d.m'
results_path = './results/r_14may_ht1_pca/'
%% jobs
jobs = 5
start_centers = 10
end_centers = 250
%% PCA model location
load_pca = 0
pca_mdl_loc = ''
%% data
data_set_path = '../../hbf_research_data/data_MNIST_original_minist_60k_10k_split_train_test'
data_normalized = 0
%% GPU
gpu_on = 0