from scipy.stats import spearmanr
import os
import pandas as pd
# from keras.models import Model
# from tensorflow.errors import InvalidArgumentError
import nnmodel_utils
import numpy as np
import math
from sklearn.cross_decomposition import CCA

def calculate_corr(preds1,preds2,corr_type):
    corr_arr=[]
    for i in range (0,preds1.shape[0]):
        if(corr_type=='spearmanr'):
            corr_arr.append(spearmanr(preds1[i], preds2[i])[0])
    return corr_arr

def calculate_corr_columnwise(preds1,preds2,corr_type):

    corr_arr=[]
    if(preds1.shape[1]==preds2.shape[1]):#both prediction vectors have same num of columns
        num_col=preds1.shape[1]
        for i in range (0,num_col):
            if(corr_type=='spearmanr'):
                corr_arr.append(spearmanr(preds1[:,i], preds2[:,i],axis=0)[0])
    return corr_arr

def save_dataset(input_arr,dataset_name,path):
    folder_name=os.path.join(path)#,"AllPurposeData")
    if(not os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    pd.DataFrame(input_arr).to_csv(os.path.join(folder_name,dataset_name+".csv"),header=False,index=False)

#copied from ModelsDatasetProcessing-New
def load_model_data(generated_datasets_path,dataset_name):
    folder_name=os.path.join(generated_datasets_path)
    df=pd.read_csv(os.path.join(folder_name,dataset_name+".csv"),header=None,index_col=False)
    arr=df.to_numpy()
    return arr

def check_modelpair_comparability_by_input_shapes(model_q,model_c):
    modelq_input_shape=nnmodel_utils.get_model_input_shape(model_q.model)
    modelc_input_shape=nnmodel_utils.get_model_input_shape(model_c.model)
    model_q_num_inputs=nnmodel_utils.get_flat_input_shape(modelq_input_shape)
    model_c_num_inputs=nnmodel_utils.get_flat_input_shape(modelc_input_shape)
    if(model_q_num_inputs==model_c_num_inputs):
        return True
    else:
        return False

def check_modelpair_comparability_by_output_shapes(model_q,model_c):
    modelq_output_shape=model_q.model.layers[len(model_q.model.layers) - 1].output_shape
    modelc_output_shape=model_c.model.layers[len(model_c.model.layers) - 1].output_shape
    if(model_q.activation_last_layer==model_c.activation_last_layer and modelq_output_shape==modelc_output_shape):
        if (model_q.activation_last_layer=='softmax'):# only classifiers --or model_q.activation_last_layer=="linear"):
            return True
        elif (model_q.activation_last_layer=='sigmoid'):
            # if(str(modelq_output_shape)== to allow for any classifiers, not only the single label ones
            return True
    return False

def get_CCA_corrs(preds1,preds2):
    num_col=preds1.shape[1]
    ca = CCA(n_components=num_col)
    ca.fit(preds1, preds2)
    X_c, Y_c = ca.transform(preds1, preds2)
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(num_col)]
    return corrs

# compare all models with model_all:
def compare_models_by_model_query_dataset(all_models_dic, dataset_q, dataset_name,sim_metric):  # ref_dataset is mnist or fashion
    list_pairs = list()
    compared_set=set()
    num_compared = 0
    models_pred_dic=dict()
    for model_q_name,model_q in all_models_dic.items():
        for modelname, nnmodel in all_models_dic.items():
            # print(nnmodel.model_name)
            try:
                # only iterate on models with dataset generated and avoid model_all and model_fash
                if (nnmodel.model_name != model_q.model_name and not model_q.model_name+" "+nnmodel.model_name in compared_set):# and not modelname.startswith("MNIST_models"):  # only for uncontrolled experiments: exclude other query models too, those names start with MNIST_models
                    models_comparable = check_modelpair_comparability_by_input_shapes(model_q, nnmodel)
                    if (models_comparable and check_modelpair_comparability_by_output_shapes(model_q, nnmodel)):
                        print("models " + nnmodel.model_name + " and " + model_q.model_name + " are comparable in terms of input and output shape")
                        compared_set.add(model_q.model_name+" "+nnmodel.model_name)
                        compared_set.add(nnmodel.model_name+" "+model_q.model_name)
                        # compare with mnist
                        # if (ref_dataset != None):
                        #     # acc, acc_me = evaluate_model(nnmodel.model, ref_dataset, nnmodel.input_dim)
                        #     acc = -1
                        #     acc_me = -1
                        # else:
                        #     acc = -1
                        #     acc_me = -1
                        acc = -1
                        acc_me = -1
                        mp = nnmodel_utils.ModelPair(model_q, nnmodel)
                        if(model_q.model_name in models_pred_dic):
                            model_q_preds=models_pred_dic[model_q.model_name]
                        else:
                            model_q_preds = model_q.model.predict(nnmodel_utils.convert_flat_input_to_correct_input_shape(dataset_q, model_q.input_dim))
                            models_pred_dic[model_q.model_name]=model_q_preds
                        #                     print(model_q_preds)
                        if (nnmodel.model_name in models_pred_dic):
                            model_c_preds = models_pred_dic[nnmodel.model_name]
                        else:
                            model_c_preds = nnmodel.model.predict(nnmodel_utils.convert_flat_input_to_correct_input_shape(dataset_q, nnmodel.input_dim))
                            models_pred_dic[nnmodel.model_name] = model_c_preds
                        #                     print(model_c_preds)
                        mean_corr = math.nan
                        if (sim_metric == 'spearmanr'):
                            corrs = calculate_corr_columnwise(model_q_preds, model_c_preds, 'spearmanr')
                            metric=sim_metric
                            mean_corr = np.mean(corrs)
                        if(math.isnan(mean_corr) or sim_metric == 'cca'):
                            corrs = get_CCA_corrs(model_q_preds,model_c_preds)
                            print("CCA:")
                            mean_corr = np.mean(corrs)
                            metric='CCA'

                        print(str(mean_corr))
                        # print("acc by me: " + str(acc_me))

                        mp.fill_similarity_by_agreements(dataset_name + "_" + model_q.model_name+"_"+nnmodel.model_name, mean_corr, mean_corr,metric)
                        mp.query_model_dataset_acc = acc
                        mp.query_model_dataset_acc_me = acc_me
                        num_compared += 1

                        list_pairs.append(mp)
                    else:
                        print("models " + nnmodel.model_name + " " + model_q.model_name + " not comparable in terms of input or output shape")
            except:# InvalidArgumentError as e:
                # print(e)
                print("could not get predictions for " + nnmodel.model_name)
    print("total compared: " + str(num_compared))
    return list_pairs