clear;
%% errors vs iterations
location = 'errors_vs_iterations';
mkdir( sprintf('../%s',location) )
jobs = 5;
for job_num=1:jobs
    prefix_name = sprintf('test_error_vs_iterations%d',job_num);
    results_fold = 'r_1apr_omt5_HBF1'; %<---- CHANGE
    errors_location = sprintf('../results/%s/%s', results_fold, prefix_name);
    load(errors_location);
    %errors_train = best_train_error_H_mdl;
    %errors_test = best_test_error_H_mdl;
    errors_train = best_train_iteration_errors_H_mdl;
    errors_test = best_test_iteration_errors_H_mdl;
    fig = plot_error_vs_iterations_single_model( center, errors_train, errors_test, eta_c, eta_t)
    title( strrep(results_fold, '_', ' ') )
    %saveas(fig, sprintf( '../%s/%d.jpeg',location, center) );
    %saveas(fig, sprintf('../%s/%d',location, center) );
end
beep;