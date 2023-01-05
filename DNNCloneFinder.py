from scipy.stats import spearmanr
import numpy as np
import utils
import nnmodel_utils
import model_load_utils
import sys
import os
from keras import models
# from os import listdir
# import tensorflow as tf

# tf.config.threading.set_inter_op_parallelism_threads(4)

models_path=sys.argv[1] #'test_data'
generated_datasets_path=sys.argv[2]
dataset_file_name=sys.argv[3]
if len(sys.argv) == 5:
    num_rows=sys.argv[4]
    if (not str.isdigit(num_rows)):
        print("Please enter a valid input for number of rows.")
        exit()
    num_rows = int(num_rows)
else:
    num_rows = 4000

processed_file_path="processed.txt"
skipped_file_path="skipped.txt"
output_main_path="correlation_results"



if (not os.path.isdir(output_main_path)):
    os.mkdir(output_main_path)

if (not os.path.isdir(generated_datasets_path)):
    os.mkdir(generated_datasets_path)

processed=set()

for f in os.listdir(output_main_path):
    if (os.path.isfile(os.path.join(output_main_path,f))):
        cluster_name=f[:-4]
        processed.add(cluster_name)

datasets_files_set=set()
for f in os.listdir(generated_datasets_path):
    if (os.path.isfile(os.path.join(generated_datasets_path,f))):
        dataset_name=f[:-4]
        datasets_files_set.add(dataset_name)

skipped_file=open(skipped_file_path, 'a')


if (os.path.isdir(models_path)):
    models_dic = dict()
    for filename in os.listdir(models_path):
        if (filename.endswith(".h5") ):
            h5_full_path = os.path.join(models_path, filename)
            # print(h5_full_path)
            m=None #to nullify the last model
            nnmodel=model_load_utils.model_loading(h5_full_path,h5_full_path,'')
            if(nnmodel!=None):
                models_dic[h5_full_path]=nnmodel
                # model_count+=1
                input_dim_flat=nnmodel_utils.get_flat_input_shape(nnmodel.input_dim)
            # print("++++++++++++++++++++++++++++++++++
            current_output_file = open(os.path.join(output_main_path, "results.txt"), 'w')#create a file for output

            if(len(models_dic)>1):
                if (os.path.isfile(os.path.join(generated_datasets_path, dataset_file_name+".csv"))):
                    random_data = utils.load_model_data(generated_datasets_path, dataset_file_name)
                else:
                    random_data = nnmodel_utils.create_random_input(num_rows, input_dim_flat, -1, 1)
                    utils.save_dataset(random_data, dataset_file_name, generated_datasets_path)
                # random_data_file_name='RandomData-' + str(models_dic[next(iter(models_dic))].input_dim)+'-'+str(num_rows) + 'Rows' #models_dic[next(iter(models_dic))]is the first item in dic
                # if(random_data_file_name_small in datasets_files_set):
                #     random_data=utils.load_model_data(generated_datasets_path,random_data_file_name_small)
                # elif(random_data_file_name_big in datasets_files_set):
                #     random_data = utils.load_model_data(generated_datasets_path, random_data_file_name_big)
                # else:
                #     random_data = nnmodel_utils.create_random_input(5000, input_dim_flat, -1, 1)
                # random_data = nnmodel_utils.create_random_input(num_rows, input_dim_flat, -1, 1)
                    # utils.save_dataset(random_data, random_data_file_name_small,generated_datasets_path)
                list_candidate_pairs = dict()
                list_candidate_pairs = utils.compare_models_by_model_query_dataset(models_dic, random_data,'random_data','spearmanr')  # ref_dataset is mnist or fashion
                print("++++++++++++++++++++++++++++++++++")
                # for model_name in models_dic.keys():
                #     print(model_name)
                #     print(models_dic[model_name].activation_last_layer)
                # folder_name = os.path.join(output_main_path,cluster)  # ,"AllPurposeData")
                # if (not os.path.isdir(folder_name)):
                #     os.mkdir(folder_name)
                # current_cluster_output_file = open(os.path.join(output_main_path,cluster+".txt"), 'w')
                for mp in list_candidate_pairs:
                    # print(mp.similarity.corrs_list)
                    current_output_file.write(str(mp.model1.model_name)+"@#@"+str(mp.model2.model_name)+"@#@"+str(mp.similarity.perc_agreements)+"@#@"+mp.similarity.metric+"\n")
                print(len(list_candidate_pairs))
                # processed.add(cluster)
                # processed_file.write(cluster+"\n")
                # print("processed "+cluster)
            current_output_file.close()
        else:
            print("skipping "+filename)
            skipped_file.write(filename+"\n")



# processed_file.close()
skipped_file.close()
