disp(train_func_name)
if strcmp( train_func_name, 'learn_HBF1_SGD')
    folderName = fullfile('../../hbf_research_ml_model_library/multilayer_HBF_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
elseif strcmp( train_func_name, 'learn_HSig_SGD') || strcmp( train_func_name, 'learn_HReLu_SGD') || strcmp( train_func_name, 'learn_HSig_SGD') ||
    folderName = fullfile('../../hbf_research_ml_model_library/multilayer_HModel_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
end
folderName = fullfile('../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);