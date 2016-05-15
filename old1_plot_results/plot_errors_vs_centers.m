%% errors vs centers
path_to_results = '../results/r_22mar_j3/';
num_models = 5;
fig = plot_errors_vs_centers_results( path_to_results, num_models )
saveas(fig, 'errors_vs_centers');
saveas(fig, 'errors_vs_centers.jpeg');
beep;