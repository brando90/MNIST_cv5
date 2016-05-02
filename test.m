results_path = './results/r_5mar_ht1/'
%%
my_self = 'get_best_trained_hbf1_model.m';
source = sprintf('./%s', my_self);
destination = sprintf('%s', results_path)
copyfile(source, destination);
beep;
disp('Done')