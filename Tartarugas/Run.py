import CDP.KFold_Validation as kf
from CGT.Error_Validate import save_lbp_RGB, save_lbp_YCBCR

kf.clean_Result()
#A primeira vez que rodar: b_update = True

error_file, name_test, name_pred = kf.accuracyK_Fold_SVM_RGB(save=True)#, b_update=True)
save_lbp_RGB(error_file, name_test, name_pred, 'SVM', 'RGB')

error_file, name_test, name_pred = kf.accuracyK_Fold_SVM_YCBCR(save=True)
save_lbp_YCBCR(error_file, name_test, name_pred, 'SVM', 'YCBCR')

error_file, name_test, name_pred = kf.accuracyK_Fold_KNN_RGB()
#lbp_from_rgb(error_file, name_test, name_pred)

error_file, name_test, name_pred = kf.accuracyK_Fold_KNN_YCBCR()
#lbp_from_rgb(error_file, name_test, name_pred)