import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LassoLarsIC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

CV_SET=10

def clever_print(input_string):
    print('-'*len(input_string))
    print(input_string)
    print('-'*len(input_string))

#data_procession------------------------------------
np.random.seed(9002)
df=pd.read_csv('./clean_data.csv',index_col=0)
df=pd.concat([df,pd.get_dummies(df.Poll_Rating,prefix=['poll_rating_'])],axis=1)
df.loc[np.isnan(df.PVI),'PVI']=np.mean(df[~np.isnan(df.PVI)].PVI)

dis_col=['District_ID_AK_0','District_ID_AL_1','District_ID_AL_2','District_ID_AL_3','District_ID_AL_4','District_ID_AL_5','District_ID_AL_6','District_ID_AL_7','District_ID_AL_88','District_ID_AR_1','District_ID_AR_2','District_ID_AR_3','District_ID_AR_4','District_ID_AS_0','District_ID_AZ_1','District_ID_AZ_2','District_ID_AZ_3','District_ID_AZ_4','District_ID_AZ_5','District_ID_AZ_6','District_ID_AZ_7','District_ID_AZ_8','District_ID_AZ_9','District_ID_CA_1','District_ID_CA_10','District_ID_CA_11','District_ID_CA_12','District_ID_CA_13','District_ID_CA_14','District_ID_CA_15','District_ID_CA_16','District_ID_CA_17','District_ID_CA_18','District_ID_CA_19','District_ID_CA_2','District_ID_CA_20','District_ID_CA_21','District_ID_CA_22','District_ID_CA_23','District_ID_CA_24','District_ID_CA_25','District_ID_CA_26','District_ID_CA_27','District_ID_CA_28','District_ID_CA_29','District_ID_CA_3','District_ID_CA_30','District_ID_CA_31','District_ID_CA_32','District_ID_CA_33','District_ID_CA_34','District_ID_CA_35','District_ID_CA_36','District_ID_CA_37','District_ID_CA_38','District_ID_CA_39','District_ID_CA_4','District_ID_CA_40','District_ID_CA_41','District_ID_CA_42','District_ID_CA_43','District_ID_CA_44','District_ID_CA_45','District_ID_CA_46','District_ID_CA_47','District_ID_CA_48','District_ID_CA_49','District_ID_CA_5','District_ID_CA_50','District_ID_CA_51','District_ID_CA_52','District_ID_CA_53','District_ID_CA_6','District_ID_CA_7','District_ID_CA_8','District_ID_CA_9','District_ID_CO_1','District_ID_CO_15','District_ID_CO_2','District_ID_CO_3','District_ID_CO_4','District_ID_CO_5','District_ID_CO_6','District_ID_CO_7','District_ID_CT_1','District_ID_CT_2','District_ID_CT_3','District_ID_CT_4','District_ID_CT_5','District_ID_DC_0','District_ID_DE_0','District_ID_FL_0','District_ID_FL_1','District_ID_FL_10','District_ID_FL_11','District_ID_FL_12','District_ID_FL_13','District_ID_FL_14','District_ID_FL_15','District_ID_FL_16','District_ID_FL_17','District_ID_FL_18','District_ID_FL_19','District_ID_FL_2','District_ID_FL_20','District_ID_FL_21','District_ID_FL_22','District_ID_FL_23','District_ID_FL_24','District_ID_FL_25','District_ID_FL_26','District_ID_FL_27','District_ID_FL_3','District_ID_FL_4','District_ID_FL_5','District_ID_FL_6','District_ID_FL_7','District_ID_FL_8','District_ID_FL_9','District_ID_GA_1','District_ID_GA_10','District_ID_GA_11','District_ID_GA_12','District_ID_GA_13','District_ID_GA_14','District_ID_GA_2','District_ID_GA_3','District_ID_GA_4','District_ID_GA_5','District_ID_GA_6','District_ID_GA_7','District_ID_GA_8','District_ID_GA_9','District_ID_GU_0','District_ID_HI_1','District_ID_HI_2','District_ID_IA_1','District_ID_IA_2','District_ID_IA_3','District_ID_IA_4','District_ID_IA_5','District_ID_ID_1','District_ID_ID_2','District_ID_IL_1','District_ID_IL_10','District_ID_IL_11','District_ID_IL_12','District_ID_IL_13','District_ID_IL_14','District_ID_IL_15','District_ID_IL_16','District_ID_IL_17','District_ID_IL_18','District_ID_IL_19','District_ID_IL_2','District_ID_IL_3','District_ID_IL_4','District_ID_IL_5','District_ID_IL_6','District_ID_IL_7','District_ID_IL_8','District_ID_IL_9','District_ID_IN_1','District_ID_IN_2','District_ID_IN_3','District_ID_IN_4','District_ID_IN_5','District_ID_IN_6','District_ID_IN_7','District_ID_IN_8','District_ID_IN_9','District_ID_KS_1','District_ID_KS_2','District_ID_KS_3','District_ID_KS_4','District_ID_KY_1','District_ID_KY_2','District_ID_KY_3','District_ID_KY_4','District_ID_KY_5','District_ID_KY_6','District_ID_LA_1','District_ID_LA_2','District_ID_LA_3','District_ID_LA_4','District_ID_LA_5','District_ID_LA_6','District_ID_MA_1','District_ID_MA_10','District_ID_MA_15','District_ID_MA_2','District_ID_MA_3','District_ID_MA_4','District_ID_MA_5','District_ID_MA_6','District_ID_MA_7','District_ID_MA_8','District_ID_MA_9','District_ID_MD_0','District_ID_MD_1','District_ID_MD_2','District_ID_MD_3','District_ID_MD_4','District_ID_MD_5','District_ID_MD_6','District_ID_MD_7','District_ID_MD_8','District_ID_ME_1','District_ID_ME_2','District_ID_MI_1','District_ID_MI_10','District_ID_MI_11','District_ID_MI_12','District_ID_MI_13','District_ID_MI_14','District_ID_MI_15','District_ID_MI_2','District_ID_MI_3','District_ID_MI_4','District_ID_MI_5','District_ID_MI_6','District_ID_MI_7','District_ID_MI_8','District_ID_MI_9','District_ID_MN_1','District_ID_MN_2','District_ID_MN_3','District_ID_MN_4','District_ID_MN_5','District_ID_MN_6','District_ID_MN_7','District_ID_MN_8','District_ID_MO_1','District_ID_MO_2','District_ID_MO_3','District_ID_MO_4','District_ID_MO_5','District_ID_MO_6','District_ID_MO_7','District_ID_MO_8','District_ID_MO_9','District_ID_MP_0','District_ID_MP_1','District_ID_MS_1','District_ID_MS_2','District_ID_MS_3','District_ID_MS_4','District_ID_MT_0','District_ID_NC_0','District_ID_NC_1','District_ID_NC_10','District_ID_NC_11','District_ID_NC_12','District_ID_NC_13','District_ID_NC_2','District_ID_NC_3','District_ID_NC_4','District_ID_NC_5','District_ID_NC_6','District_ID_NC_7','District_ID_NC_8','District_ID_NC_9','District_ID_ND_0','District_ID_NE_1','District_ID_NE_2','District_ID_NE_3','District_ID_NH_1','District_ID_NH_2','District_ID_NJ_1','District_ID_NJ_10','District_ID_NJ_11','District_ID_NJ_12','District_ID_NJ_13','District_ID_NJ_2','District_ID_NJ_3','District_ID_NJ_4','District_ID_NJ_5','District_ID_NJ_6','District_ID_NJ_7','District_ID_NJ_8','District_ID_NJ_9','District_ID_NM_1','District_ID_NM_2','District_ID_NM_3','District_ID_NV_1','District_ID_NV_2','District_ID_NV_3','District_ID_NV_4','District_ID_NY_1','District_ID_NY_10','District_ID_NY_11','District_ID_NY_12','District_ID_NY_13','District_ID_NY_14','District_ID_NY_15','District_ID_NY_16','District_ID_NY_17','District_ID_NY_18','District_ID_NY_19','District_ID_NY_2','District_ID_NY_20','District_ID_NY_21','District_ID_NY_22','District_ID_NY_23','District_ID_NY_24','District_ID_NY_25','District_ID_NY_26','District_ID_NY_27','District_ID_NY_28','District_ID_NY_29','District_ID_NY_3','District_ID_NY_4','District_ID_NY_5','District_ID_NY_6','District_ID_NY_7','District_ID_NY_8','District_ID_NY_9','District_ID_OH_1','District_ID_OH_10','District_ID_OH_11','District_ID_OH_12','District_ID_OH_13','District_ID_OH_14','District_ID_OH_15','District_ID_OH_16','District_ID_OH_17','District_ID_OH_18','District_ID_OH_2','District_ID_OH_3','District_ID_OH_4','District_ID_OH_5','District_ID_OH_6','District_ID_OH_7','District_ID_OH_8','District_ID_OH_9','District_ID_OK_1','District_ID_OK_2','District_ID_OK_3','District_ID_OK_4','District_ID_OK_5','District_ID_OK_68','District_ID_OR_1','District_ID_OR_2','District_ID_OR_3','District_ID_OR_4','District_ID_OR_5','District_ID_PA_0','District_ID_PA_1','District_ID_PA_10','District_ID_PA_11','District_ID_PA_12','District_ID_PA_13','District_ID_PA_14','District_ID_PA_15','District_ID_PA_16','District_ID_PA_17','District_ID_PA_18','District_ID_PA_19','District_ID_PA_2','District_ID_PA_3','District_ID_PA_4','District_ID_PA_5','District_ID_PA_53','District_ID_PA_6','District_ID_PA_7','District_ID_PA_8','District_ID_PA_9','District_ID_PR_0','District_ID_RI_1','District_ID_RI_2','District_ID_SC_1','District_ID_SC_2','District_ID_SC_3','District_ID_SC_4','District_ID_SC_5','District_ID_SC_6','District_ID_SC_7','District_ID_SD_0','District_ID_TN_1','District_ID_TN_2','District_ID_TN_3','District_ID_TN_4','District_ID_TN_5','District_ID_TN_6','District_ID_TN_7','District_ID_TN_8','District_ID_TN_9','District_ID_TX_1','District_ID_TX_10','District_ID_TX_11','District_ID_TX_12','District_ID_TX_13','District_ID_TX_14','District_ID_TX_15','District_ID_TX_16','District_ID_TX_17','District_ID_TX_18','District_ID_TX_19','District_ID_TX_2','District_ID_TX_20','District_ID_TX_21','District_ID_TX_22','District_ID_TX_23','District_ID_TX_24','District_ID_TX_25','District_ID_TX_26','District_ID_TX_27','District_ID_TX_28','District_ID_TX_29','District_ID_TX_3','District_ID_TX_30','District_ID_TX_31','District_ID_TX_32','District_ID_TX_33','District_ID_TX_34','District_ID_TX_35','District_ID_TX_36','District_ID_TX_4','District_ID_TX_5','District_ID_TX_6','District_ID_TX_7','District_ID_TX_8','District_ID_TX_9','District_ID_UT_1','District_ID_UT_2','District_ID_UT_3','District_ID_UT_4','District_ID_VA_1','District_ID_VA_10','District_ID_VA_11','District_ID_VA_2','District_ID_VA_3','District_ID_VA_4','District_ID_VA_5','District_ID_VA_6','District_ID_VA_7','District_ID_VA_8','District_ID_VA_9','District_ID_VI_0','District_ID_VT_0','District_ID_WA_1','District_ID_WA_10','District_ID_WA_2','District_ID_WA_3','District_ID_WA_4','District_ID_WA_5','District_ID_WA_6','District_ID_WA_7','District_ID_WA_8','District_ID_WA_9','District_ID_WI_1','District_ID_WI_2','District_ID_WI_3','District_ID_WI_4','District_ID_WI_5','District_ID_WI_6','District_ID_WI_7','District_ID_WI_8','District_ID_WV_1','District_ID_WV_2','District_ID_WV_3','District_ID_WY_0']
bin_col = [col for col in df if df[col].dropna().value_counts().index.isin([0,1]).all()]
non_bin_col= df.columns.difference(bin_col)
target_col=['Party_DEM','Party_OTHER','Party_REP']
drop_col=['name_commonness','Same_Party_As_President','Orig_District_ID','Orig_Gender','Orig_Party','Orig_Governor_Party']
train_columns=df.columns.difference(target_col+['class_label']+drop_col)

def give_class(row):
    if row.Party_DEM==1:
        return 1
    if row.Party_OTHER==1:
        return 3
    if row.Party_REP==1:
        return 2
df['class_label']=df.apply(lambda row: give_class(row),axis=1)

data_train_x=df[df.Year!=2018][df.Year!=2016][train_columns]
data_train_y=df[df.Year!=2018][df.Year!=2016].class_label
data_dev_x=df[df.Year==2016][train_columns]
data_dev_y=df[df.Year==2016].class_label
data_all_x=df[df.Year!=2018][train_columns]
data_all_y=df[df.Year!=2018].class_label

#logistic regression------------------------------------

# from sklearn.linear_model import LogisticRegression

# clever_print('logistic regression with no penelization')
# basic_multi_logi=LogisticRegression(max_iter=10000000,penalty='l2',C=10000000000).fit(data_train_x,data_train_y)
# print('accuracy on training')
# print(accuracy_score(data_train_y,basic_multi_logi.predict(data_train_x)))
# print('accuracy on dev')  
# print(accuracy_score(data_dev_y,basic_multi_logi.predict(data_dev_x)))

# clever_print('logistic regression CV with L1 penelization')
# to_cv_C = [0.0001,0.001,0.01,0.1,1,10]
# param_grid = {'C': to_cv_C}
# logi_l1_grid_search = GridSearchCV(LogisticRegression(max_iter=100000,penalty='l1'), param_grid, cv=CV_SET)
# logi_l1_grid_search.fit(data_all_x, data_all_y)
# print('best cv accuracy')
# print(logi_l1_grid_search.best_score_)
# print('best params')
# print(logi_l1_grid_search.best_params_)

# clever_print('logistic regression with L2 penelization, CV')
# logi_l2_grid_search = GridSearchCV(LogisticRegression(max_iter=100000,penalty='l2'), param_grid, cv=CV_SET)
# logi_l2_grid_search.fit(data_all_x, data_all_y)
# print('best cv accuracy')
# print(logi_l2_grid_search.best_score_)
# print('best params')
# print(logi_l2_grid_search.best_params_)


# #decision tree------------------------------------

# from sklearn import tree
# decision_tree=tree.DecisionTreeClassifier().fit(data_train_x,data_train_y)
# print('accuracy on training')
# print(accuracy_score(data_train_y,decision_tree.predict(data_train_x)))
# print('accuracy on dev')
# print(accuracy_score(data_dev_y,decision_tree.predict(data_dev_x)))


# clever_print('decision Tree with CV using grid search')
# from sklearn.model_selection import GridSearchCV

# to_cv_depth = range(1,100,10)
# to_cv_min_samples_split=range(2,20)
# param_grid = {'max_depth': to_cv_depth,'min_samples_split':to_cv_min_samples_split}
# decision_tree_grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=CV_SET)
# decision_tree_grid_search.fit(data_all_x, data_all_y)

# print('accuracy on training and dev')
# print(decision_tree_grid_search.best_score_)
# print('best param')
# print(decision_tree_grid_search.best_params_)


# # Random Forest------------------------------------

# from sklearn.ensemble import RandomForestClassifier
# random_forest= RandomForestClassifier().fit(data_train_x,data_train_y)
# print('accuracy on training')
# print(accuracy_score(data_train_y,random_forest.predict(data_train_x)))
# print('accuracy on dev')
# print(accuracy_score(data_dev_y,random_forest.predict(data_dev_x)))

# clever_print('random forest with CV using grid search')

# to_cv_depth = range(1,100,10)
# to_cv_min_samples_split=range(2,20)
# param_grid = {'max_depth': to_cv_depth,'min_samples_split':to_cv_min_samples_split}

# random_forest_grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=CV_SET)
# random_forest_grid_search.fit(data_all_x, data_all_y)

# print('accuracy on training and dev')
# print(random_forest_grid_search.best_score_)
# print('best param')
# print(random_forest_grid_search.best_params_)


# KNN------------------------------------


knn_scaler_all = StandardScaler().fit(data_all_x)
knn_scaler_train = StandardScaler().fit(data_train_x)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier().fit(knn_scaler_train.transform(data_train_x),data_train_y)
clever_print('KNN')
print('accuracy on training')
print(accuracy_score(data_train_y,knn.predict(knn_scaler_train.transform(data_train_x))))
print('accuracy on dev')
print(accuracy_score(data_dev_y,knn.predict(knn_scaler_train.transform(data_dev_x))))


to_cv_nei = range(1,100,5)
param_grid = {'n_neighbors':to_cv_nei}
clever_print('KNN CV')
knn_cv= GridSearchCV(KNeighborsClassifier(), param_grid, cv=CV_SET)
knn_cv.fit(knn_scaler_all.transform(data_all_x), data_all_y)

print('accuracy on training and dev')
print(knn_cv.best_score_)
print('best param')
print(knn_cv.best_params_)


input()
# SVM------------------------------------
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

svm_scaler = StandardScaler().fit(data_all_x)
svm_scaler_train=StandardScaler().fit(data_train_x)


clever_print('svm with linear kernel')
svm_linear=SVC(kernel='linear').fit(svm_scaler_train.transform(data_train_x),data_train_y)
print('accuracy on training')
print(accuracy_score(data_train_y,svm_linear.predict(svm_scaler_train.transform(data_train_x))))
print('accuracy on dev')
print(accuracy_score(data_dev_y,svm_linear.predict(svm_scaler_train.transform(data_dev_x))))

clever_print('svm with rbf kernel')
svm_rbf=SVC(kernel='rbf').fit(svm_scaler_train.transform(data_train_x),data_train_y)
print('accuracy on training')
print(accuracy_score(data_train_y,svm_rbf.predict(svm_scaler_train.transform(data_train_x))))
print('accuracy on dev')
print(accuracy_score(data_dev_y,svm_rbf.predict(svm_scaler_train.transform(data_dev_x))))


clever_print('svm with linear kernel with cv')
to_cv_c = [0.01,0.05,0.1,0.5,1,2,5,10]
param_grid = {'C':to_cv_c}
svm_linear_cv= GridSearchCV(SVC(kernel='linear'), param_grid, cv=CV_SET)
svm_linear_cv.fit(svm_scaler.transform(data_all_x), data_all_y)

print('accuracy on training and dev')
print(svm_linear_cv.best_score_)
print('best param')
print(svm_linear_cv.best_params_)

clever_print('svm with RBF kernel with cv')
svm_rbf_cv= GridSearchCV(SVC(kernel='rbf'), param_grid, cv=CV_SET)
svm_rbf_cv.fit(svm_scaler.transform(data_all_x), data_all_y)


print('accuracy on training and dev')
print(svm_rbf_cv.best_score_)
print('best param')
print(svm_rbf_cv.best_params_)


# LDA QDA------------------------------------

clever_print('LDA analysis')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis().fit(data_train_x,data_train_y)
print('accuracy on training')
print(accuracy_score(data_train_y,lda.predict(data_train_x)))
print('accuracy on dev')
print(accuracy_score(data_dev_y,lda.predict(data_dev_x)))

clever_print('QDA analysis')

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda=QuadraticDiscriminantAnalysis().fit(data_train_x,data_train_y)
print('accuracy on training')
print(accuracy_score(data_train_y,qda.predict(data_train_x)))
print('accuracy on dev')
print(accuracy_score(data_dev_y,qda.predict(data_dev_x)))

# NN(MLP)------------------------------------

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

nn_scaler = StandardScaler().fit(data_all_x)
mlp= MLPClassifier(hidden_layer_sizes=(100,50,10)).fit(nn_scaler.transform(data_train_x),data_train_y)

print('accuracy on training')
print(accuracy_score(data_train_y,mlp.predict(nn_scaler.transform(data_train_x))))
print('accuracy on dev')
print(accuracy_score(data_dev_y,mlp.predict(nn_scaler.transform(data_dev_x))))


clever_print('neural network with cv')
to_cv_learning_rate=[0.0001,0.001,0.01]
to_cv_alpha=[0.00001,0.0001,0.001,0.01,0.1]
to_cv_hidden_layer_sizes=[(100,50,10),(100,10),(50,5)]
param_grid = {'learning_rate_init':to_cv_learning_rate,'alpha':to_cv_alpha,'hidden_layer_sizes':to_cv_hidden_layer_sizes}
mlp_cv= GridSearchCV(MLPClassifier(max_iter=10000), param_grid, cv=CV_SET)
mlp_cv.fit(nn_scaler.transform(data_all_x), data_all_y)
mlp_cv.best_params_

print('accuracy on training and dev')
print(mlp_cv.best_score_)
print('best param')
print(mlp_cv.best_params_)