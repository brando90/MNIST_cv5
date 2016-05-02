%% PCA
train_func_name = 'PCA' %DOESN'T actually do anything, not set to anything, load_paths will crash
%% locations
cp_folder = 'cp_30mar_hj1_pca/'
cp_param_files_names = 'cp_30mar_hj1_pca_%d.m'
results_path = './results/r_30mar_hj1_pca/'
%% jobs
jobs = 5
start_centers = 10
end_centers = 250
%% PCA model location
load_pca = 0
pca_mdl_loc = ''
%% data
data_set_path = '../../hbf_research_data/data_MNIST_0.7_0.15_0.15_49000_10500_10500.mat'
data_normalized = 0
%% GPU
gpu_on = 0