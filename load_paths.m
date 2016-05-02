disp(train_func_name)
if strcmp( train_func_name, 'learn_HBF1_SGD')
%     folderName = fullfile('../../hbf_research_ml_model_library/RBF_multivariant_regression');
%     p = genpath(folderName);
%     addpath(p);
    addpath('../../hbf_research_ml_model_library/RBF_multivariant_regression/RBF'); %only adds the RBF class
    folderName = fullfile('../../hbf_research_ml_model_library/HBF1_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
elseif strcmp( train_func_name, 'learn_RBF_SGD')
    folderName = fullfile('../../hbf_research_ml_model_library/RBF_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
elseif strcmp( train_func_name, 'learn_HSig_SGD')
    folderName = fullfile('../../hbf_research_ml_model_library/HSig_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
elseif strcmp( train_func_name, 'learn_HReLu_SGD')
    folderName = fullfile('../../hbf_research_ml_model_library/HReLu_multivariant_regression');
    p = genpath(folderName);
    addpath(p);
end
% addpath('../../../common/squared_error_risk');
% addpath('../../../common/visualize_centers')
% addpath('../../../common/cross_validation/standard_train_cv_test_validation')
% addpath('../../../common')
% addpath('../../../common/MNIST')
% addpath('../../../common/kernel_functions')
%
folderName = fullfile('../../hbf_research_ml_model_library/common');
p = genpath(folderName);
addpath(p);
