function [fig] = plot_errors_vs_centers_results( path_to_results, num_models )
% path_to_results = './results/results'
files = dir( sprintf('%s%s', path_to_results, 'results*') ); %collect results
centers = zeros(1,num_models); %TODO
list_train_errors_H_mdl = zeros(1,num_models);
list_test_errors_H_mdl = zeros(1,num_models);
list_train_errors_kernel_mdl = zeros(1,num_models);
list_test_errors_kernel_mdl = zeros(1,num_models);
for file = files'
    name = file.name;
    path=sprintf('%s%s',path_to_results,name)
    run(path);
    i = task_id;
    centers(i) = center;
    list_train_errors_H_mdl(i) = train_error_H_mdl;
    list_test_errors_H_mdl(i) = test_error_H_mdl;
    list_train_errors_kernel_mdl(i) = train_error_kernel_mdl;
    list_test_errors_kernel_mdl(i) = test_error_kernel_mdl;
end
% load('../rbf_real')
% list_train_errors_real_RBF = repmat( train_error_RBF,[1,num_models]);
% list_test_errors_real_RBF = repmat( test_error_RBF,[1,num_models]);
%%
centers = centers(1,1:num_models);
list_train_errors_H_mdl = list_train_errors_H_mdl(1,1:num_models);
list_test_errors_H_mdl = list_test_errors_H_mdl(1,1:num_models);
list_train_errors_kernel_mdl = list_train_errors_kernel_mdl(1,1:num_models);
list_test_errors_kernel_mdl = list_test_errors_kernel_mdl(1,1:num_models);
% list_train_errors_real_RBF = list_train_errors_real_RBF(1,1:num_models);
% list_test_errors_real_RBF = list_test_errors_real_RBF(1,1:num_models);
%%
fig = figure
plot(centers, list_train_errors_H_mdl, '-ro', centers, list_test_errors_H_mdl, '-b*', centers, list_train_errors_kernel_mdl, '-go', centers, list_test_errors_kernel_mdl, '-c*');
%plot(centers, list_train_errors_H_mdl, '-rd', centers, list_test_errors_H_mdl, '-bs', centers, list_train_errors_kernel_mdl, '-go', centers, list_test_errors_kernel_mdl, '-c*');
%plot(centers, list_train_errors_HBF1, '-ro', centers, list_test_errors_HBF1, '-b*', centers, list_train_errors_RBF, '-go', centers, list_test_errors_RBF, '-c*', centers, list_train_errors_real_RBF, '-m*', centers, list_test_errors_real_RBF, '-k*');
%legend('HBF1 SGD mdl train errors','HBF1 SGD mdl test errors', 'RBF train errors', 'RBF test errors', 'list_test_errors_RBF', 'list_test_errors_real_RBF');
legend('H mdl SGD mdl train errors','H mdl SGD mdl test errors', 'Kernel mdl train errors', 'Kernel mdl test errors');
title('Cost vs Centers');
xlabel('number of centers')
ylabel('euclidean error')
end