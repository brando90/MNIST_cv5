clear;
workspace_file_name = 'beta_start_0_end_1_num_betas_1250_center_250'; % <- CHANGE
job_name = 'betas_21mar_j9'; % <- CHANGE
path_to_name_workspace_file = sprintf('../betas/%s/%s.mat', job_name, workspace_file_name);
load(path_to_name_workspace_file)
%% times
secs
minutes
hours
%% plot
fig = figure;
plot(betas, rbf_train_errors, '-ro', betas, rbf_test_errors, '-b*')
legend('RBF train errors','RBF test errors');

%plot(betas, rbf_cv_errors, '-ro', betas, rbf_train_errors, '-b*')
%legend('RBF train errors','RBF test errors');

title('Errors (squared) vs Precision of Gaussian/beta = \beta = 1/(2 \pi \sigma)  ');
xlabel('Precision of Gaussian/beta = \beta = 1/(2 \pi \sigma)')
ylabel('Error (squred/euclidean error)')
saveas(fig, 'beta_vs_error');
saveas(fig, 'beta_vs_error.jpeg');