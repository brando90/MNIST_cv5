clear;
%%
num_models = 5; % <-- change
results_file_name_prefix_PCA = '../results/r_30mar_hj1_pca/'; % <-- change
results_file_name_prefix_HSig = '../results/r_30mar_omj3_HSig/'; % <-- change
results_file_name_prefix_HReLu = '../results/r_30mar_omj2_HReLu/';% <-- change 
results_file_name_prefix_HBF = '../results/r_1apr_omt5_HBF1/';% <-- change
results_file_name_prefix = 'results';
%PCA
[ ~, list_train_errors_PCA, list_test_errors_PCA ] = collect_errors_vs_centers(results_file_name_prefix_PCA, results_file_name_prefix, num_models, 'train_error_PCA', 'test_error_PCA');
% HSig
[ ~, list_train_errors_Sig, list_test_errors_Sig ] = collect_errors_vs_centers(results_file_name_prefix_HSig, results_file_name_prefix, num_models, 'train_error_kernel_mdl', 'test_error_kernel_mdl');
[ ~, list_train_errors_HSig, list_test_errors_HSig ] = collect_errors_vs_centers(results_file_name_prefix_HSig, results_file_name_prefix, num_models, 'train_error_H_mdl', 'test_error_H_mdl');
% HReLu
[ ~, list_train_errors_ReLu, list_test_errors_ReLu ] = collect_errors_vs_centers(results_file_name_prefix_HReLu, results_file_name_prefix, num_models, 'train_error_kernel_mdl', 'test_error_kernel_mdl');
[ centers, list_train_errors_HReLu, list_test_errors_HReLu ] = collect_errors_vs_centers(results_file_name_prefix_HReLu, results_file_name_prefix, num_models, 'train_error_H_mdl', 'test_error_H_mdl');
% HBF
[ ~, list_train_errors_RBF, list_test_errors_RBF ] = collect_errors_vs_centers(results_file_name_prefix_HBF, results_file_name_prefix, num_models, 'train_error_kernel_mdl', 'test_error_kernel_mdl');
[ ~, list_train_errors_HBF1, list_test_errors_HBF1 ] = collect_errors_vs_centers(results_file_name_prefix_HBF, results_file_name_prefix, num_models, 'train_error_H_mdl', 'test_error_H_mdl');
%% plot figures
%plot(centers, list_train_errors_PCA, '-ro', centers, list_test_errors_PCA, '-b*');
%legend('PCA train errors','PCA test errors');
%plot(centers, list_train_errors_PCA, '-bo', centers, list_test_errors_PCA, '-b*', centers, list_train_errors_RBF, '-go', centers, list_test_errors_RBF, '-g*', centers, list_train_errors_HBF1, '-ro', centers, list_test_errors_HBF1, '-r*');

fig1 = figure;
% plot(centers, list_train_errors_PCA, '-bo', centers, list_test_errors_PCA, '-b*', centers, list_train_errors_Sig, '-go', centers, list_test_errors_Sig, '-g*', centers, list_train_errors_HSig, '-ro', centers, list_test_errors_HSig, '-r*');
% legend('PCA train errors','PCA test errors','Sig train errors','Sig test errors', 'HSig train errors','HSig test errors');
plot(centers, list_train_errors_PCA, '-ko', centers, list_test_errors_PCA,'-k*', ...
    centers, list_train_errors_Sig, '-go', centers, list_test_errors_Sig, '-g*', ... 
    centers, list_train_errors_HSig, '-gd', centers, list_test_errors_HSig, '-gs', ...
    centers, list_train_errors_ReLu, '-bo', centers, list_test_errors_ReLu, '-b*', ...
    centers, list_train_errors_HReLu, '-co', centers, list_test_errors_HReLu, '-c*', ...
    centers, list_train_errors_RBF, '-ro', centers, list_test_errors_RBF, '-r*', ...
    centers, list_train_errors_HBF1, '-mo', centers, list_test_errors_HBF1, '-m*' ...
    );
legend('PCA train errors','PCA test errors',...
    'Sig train errors','Sig test errors', 'HSig train errors','HSig test errors', ...
    'ReLu train errors','ReLu test errors', 'HReLu train errors','HReLu test errors', ...
    'RBF train errors','RBF test errors', 'HBF1 train errors','HBF1 test errors' ...
    );

%fig2 = figure;
%plot(centers, list_train_errors_PCA, '-bo', centers, list_test_errors_PCA, '-b*', centers, list_train_errors_ReLu, '-go', centers, list_test_errors_ReLu, '-g*', centers, list_train_errors_HReLu, '-ro', centers, list_test_errors_HReLu, '-r*');
%legend('PCA train errors','PCA test errors','ReLu train errors','ReLu test errors', 'HReLu train errors','HReLu test errors');

title('Cost vs Centers');xlabel('number of centers');ylabel('euclidean error');
%% save figures
saveas(fig1, 'errors_vs_centers_pca_sig');
saveas(fig1, 'errors_vs_centers_pca_sig.jpeg');

%saveas(fig2, 'errors_vs_centers_pca_sig_relu');
%saveas(fig2, 'errors_vs_centers_sig_relu.jpeg');