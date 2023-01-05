#everything from similarityChekcer is in this file
from scipy.stats import spearmanr
import os
import pandas as pd
from keras.models import Model
import numpy as np

class datasetStore:
    def __init__(self, input_pred_map, inputs_list_map, input_pred_lbl_map):
        self.input_pred_map = input_pred_map
        self.inputs_list_map = inputs_list_map
        self.input_pred_lbl_map = input_pred_lbl_map


def get_model_input_shape(model_curr):
    model_curr_type = type(model_curr).__name__
    input_layers_shape_list = list()
    if (model_curr_type == "Sequential"):
        input_layers_shape_list.append(model_curr.layers[0].input_shape)
    elif (model_curr_type == "Functional"):  # functional models can have more than one input so we need to take care of this here
        for l in model_curr.layers:
            if (type(l).__name__ == "InputLayer"):
                input_layers_shape_list.append(l.input_shape)
    return input_layers_shape_list


def get_model_last_layer_activation(model_curr):
    if ('sigmoid' in str(model_curr.layers[len(model_curr.layers) - 1].activation)):
        return 'sigmoid'
    elif ('softmax' in str(model_curr.layers[len(model_curr.layers) - 1].activation)):
        return 'softmax'
    elif ('linear' in str(model_curr.layers[len(model_curr.layers) - 1].activation)):
        return 'linear'


def get_flat_input_shape(input_dim):
    num_inputs = len(input_dim)

    all_flat_input_shape = 0
    for input_d in input_dim:  # input_dim is a list in case model has more than one inputs
        flat_input_shape = 1
        #         print(type(input_d))
        cur_input_dim = input_d
        if (type(input_d).__name__ == "list"):  # if dim is in the form of a list, only get first item
            cur_input_dim = input_d[0]
        for dim in cur_input_dim:  # now traverse the tuple for each input
            if (dim != None):
                flat_input_shape = flat_input_shape * int(dim)
        all_flat_input_shape = all_flat_input_shape + flat_input_shape
    #     print("flat inpt shape is: "+str(all_flat_input_shape))
    return all_flat_input_shape

def convert_flat_input_to_correct_input_shape(X,input_shapes):#input_shapes is a list containing the input shapes for each input (if more than one input)
    list_input=list()
    prev_input_batch_size=0
    for input_d in input_shapes:#input_shapes is a list in case model has more than one inputs
        index_dim=0
        num_dim_curr_input=1
        cur_input_dim=input_d
        curr_shape_list_transformed=list()
        curr_input_batch_size=1
        if(type(input_d).__name__=="list"):#if dim is in the form of a list, only get first item
            cur_input_dim=input_d[0]
#         print(cur_input_dim)
        for dim in cur_input_dim:#now traverse the tuple for each input
            if(index_dim==0):
                curr_shape_list_transformed.append(-1)#first dimension is the batch size, we will let this to be inferred
            else:
#                 print(dim)
                curr_shape_list_transformed.append(dim)
                curr_input_batch_size=curr_input_batch_size*int(dim)
            index_dim+=1
#         print(tuple(curr_shape_list_transformed))
        curr_shape_tuple=tuple(curr_shape_list_transformed)
#         print(curr_input_batch_size)
        if(len(input_shapes)>1):#if more than one input
            curr_input=X[:, [i for i in range(prev_input_batch_size, prev_input_batch_size+curr_input_batch_size)]]
            X_reshaped=curr_input.reshape(curr_shape_tuple)
        else:
            X_reshaped=X.reshape(curr_shape_tuple)
        list_input.append(X_reshaped)
        prev_input_batch_size=curr_input_batch_size
        curr_input_batch_size=0
#     print("flat shape transformed to: "+str(list_input))
    return list_input

def create_random_input(num_sample,num_input,low_num,high_num):
    rand_list=[]
    rand_vals=np.random.uniform(low=low_num, high=high_num, size=num_input).reshape(1,num_input)
    rand_list.append(rand_vals)
    for i in range(1,num_sample):
        rand_vals=np.random.uniform(low=low_num, high=high_num, size=num_input).reshape(1,num_input)
        rand_list.append(rand_vals)
    return np.array(rand_list).reshape(num_sample,num_input)

class NNModel:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model
        self.input_dim = get_model_input_shape(model)
        self.activation_last_layer = get_model_last_layer_activation(model)
        self.data_probs_dic = dict()
        self.data_lbls_dic = dict()
        self.datasets = dict()  # only used for data generation
        self.numpy_arr_datasets_dic = dict()  # used for data loaded from disk
        self.acc_on_datasets_dic = dict()

    def __eq__(self, other):
        if not isinstance(other, Model):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.model_name == other.model_name

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.model_name)

    # only used for data generation
    def add_dataset(self, dataset_name, input_pred_map, inputs_list_map,
                    input_pred_lbl_map):  # this is used to add the dataset info and predictions of a dataset built for this model
        d = datasetStore(input_pred_map, inputs_list_map, input_pred_lbl_map)
        self.datasets[dataset_name] = d

    def add_numpy_arr_dataset(self, dataset_name, numpy_arr):  # used for data loaded from disk
        self.numpy_arr_datasets_dic[dataset_name] = numpy_arr

    def calculate_predictions(self, dataset_name,
                              dataset):  # this is used to get predictions for shared datasets (datsets that do not belong to one model)
        self.data_probs_dic[dataset_name] = self.model.predict(
            convert_flat_input_to_correct_input_shape(dataset, self.input_dim))
        if (self.activation_last_layer == 'softmax'):
            self.data_lbls_dic[dataset_name] = np.argmax(self.data_probs_dic[dataset_name], axis=-1)
        elif (self.activation_last_layer == 'sigmoid'):
            self.data_lbls_dic[dataset_name] = (self.data_probs_dic[dataset_name] > 0.5).astype('int32')

    def put_acc_on_dataset(self, datasetname, acc):
        self.acc_on_datasets_dic[datasetname] = acc

class SimilarityStore:
    dataset_name=''
    num_matching_labels=-1
    perc_agreements=-1
    probs_euc_dist=None
    probs_kl_div=None
    corrs_list=list()
    metric=''
    def __init__(self,dataset_name):
        self.dataset_name=dataset_name
    def set_agreemnents(self,perc,num_agree):
        self.perc_agreements=perc
        self.num_matching_labels=num_agree
    def set_corrs_by_dataset(self,list_corrs):
        self.corrs_list=list_corrs

class ModelPair:
    query_model_dataset_acc = -1
    query_model_dataset_acc_me = -1
    diff_acc_with_query_model = -1
    is_similar_predicted = -1  # 0 for different, 1 for similar, -1 for unceratin
    is_similar_true = -1  # 0 for different, 1 for similar, -1 for unceratin

    #     num_greater_than_dic_by_threshold=dict()
    #     true_similarity=-1
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2
        self.similarity = None

    def fill_similarity_by_agreements(self, dataset_name, perc, num_agree,metric):
        self.similarity = SimilarityStore(dataset_name)
        self.similarity.set_agreemnents(perc, num_agree)
        self.similarity.metric=metric

    def fill_similarity_by_correlations(self, dataset_name, list_corss):
        self.similarity = SimilarityStore(dataset_name)
        self.similarity.set_corrs_by_dataset(list_corss)
#     def get_num_greater_than_by_threshold(self,dataset_name):
#         self.num_greater_than_dic_by_threshold