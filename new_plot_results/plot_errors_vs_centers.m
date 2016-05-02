%% errors vs centers
results_fold = 'r_4apr_omt1_HBF1'; %<---- CHANGE
path_to_results = sprintf('../results/%s/', results_fold);
num_models = 5;
fig = plot_errors_vs_centers_results( path_to_results, num_models )
title( strrep(results_fold, '_', ' ') )
saveas(fig, 'centers_vs_errors');
saveas(fig, 'centers_vs_errors.jpeg');
beep;