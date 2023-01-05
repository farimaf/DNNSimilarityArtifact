import os
from keras import models
from keras import layers
import nnmodel_utils
from tensorflow.errors import InvalidArgumentError

# copied from ModelsDatasetProcessing-New
def load_models(path_to_models, max_num_model, sampled_models_set, activation_last_layer):
    models_dic = dict()
    num_model_added = 0
    num_ignored_models = 0
    for size_based_folder in os.listdir(path_to_models):  # iterate through each size based folder
        size_based_folder_full_path = os.path.join(path_to_models, size_based_folder)
        if (os.path.isdir(size_based_folder_full_path)):
            for model_folder in os.listdir(
                    size_based_folder_full_path):  # now iterate through each size based folder in each cluster
                model_folder_full_path = os.path.join(size_based_folder_full_path, model_folder)
                if (os.path.isdir(model_folder_full_path)):
                    print(model_folder_full_path)
                    for filename in os.listdir(model_folder_full_path):
                        if (filename.endswith(".h5") and (num_model_added < max_num_model)):
                            h5_full_path = os.path.join(model_folder_full_path, filename)
                            model_name_for_dic = (os.path.join(size_based_folder, model_folder, filename)).replace(
                                os.sep, '__')
                            if ((sampled_models_set == None) or (
                                    (sampled_models_set != None) and (model_name_for_dic in sampled_models_set))):
                                print(h5_full_path)
                                print("model path: " + os.path.join(size_based_folder, model_folder, filename))
                                try:
                                    nn_model = model_loading(h5_full_path, model_name_for_dic, activation_last_layer)
                                    if (nn_model != None):
                                        models_dic[model_name_for_dic] = (nn_model)  # only add the model if we can evaluate it
                                        num_model_added += 1
                                except UnicodeEncodeError as e5:
                                    num_ignored_models += 1

    return models_dic, num_ignored_models


# copied from ModelsDatasetProcessing-New
def model_loading(h5_full_path, model_name_to_save, activation_last_layer):
    model_correctly_loaded = False
    print(h5_full_path)
    try:
        m = models.load_model(h5_full_path)
        if (True ):# activation_last_layer in str(m.layers[len(m.layers) - 1].activation)):
            #         try:
            nn_model = nnmodel_utils.NNModel(model_name_to_save, m)
            print(nn_model.input_dim[0])
            if(m!=None and nn_model!=None):
                model_correctly_loaded = True
            # try:
            #     # if (nn_model.input_dim[0] == (None, 784)):
            #     #     test_acc = evaluate_model(nn_model.model, 'mnist', nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(test_img,test_label_cat)
            #     # elif (nn_model.input_dim[0] == (None, 28, 28)):
            #     #     test_acc = evaluate_model(nn_model.model, 'mnist', nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(test_image_raw_shape,test_label_cat)
            #     # elif (nn_model.input_dim[0] == (None, 60)):
            #     #     #                 test_loss,test_acc=(0,0)
            #     #     test_acc = evaluate_model(nn_model.model, 'sonar', nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(X_Sonar,Y_Sonar)
            #     # elif (nn_model.input_dim[0] == [(None, 60)]):
            #     #     test_acc = evaluate_model(nn_model.model, 'sonar', nn_model.input_dim)
            #     # #                     X_sonar_compatible=SC.convert_flat_input_to_correct_input_shape(X_Sonar,nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(X_sonar_compatible,Y_Sonar)
            #     # elif (nn_model.input_dim[0] == (None, 4)):
            #     #     test_acc = evaluate_model(nn_model.model, 'iris', nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(X_iris,Y_iris_cat)
            #     # elif (nn_model.input_dim[0] == (None, 13)):
            #     #     test_acc = evaluate_model(nn_model.model, 'boston', nn_model.input_dim)
            #     # #                     X_iris_compatible=SC.convert_flat_input_to_correct_input_shape(X_iris,nn_model.input_dim)
            #     # #                     test_loss,test_acc=m.evaluate(X_iris_compatible,Y_iris_cat)
            #     # else:
            #     #     test_loss, test_acc = 0, 0
            #     test_loss, test_acc = 0, 0
            #     # print("test acc " + str(test_acc))
            #     # model_correctly_loaded = True
            # #                 models_list.append(model_name_to_save)
            # except (ValueError) as e1:#, InvalidArgumentError) as e1:
            #     try:
            #         print(e1)
            #         print("error on categorical")
            #         # if (nn_model.input_dim[0] == (None, 784)):
            #         #     test_loss, test_acc = m.evaluate(test_img, test_label)
            #         # elif (nn_model.input_dim[0] == (None, 28, 28)):
            #         #     test_loss, test_acc = m.evaluate(test_image_raw_shape, test_label)
            #         # elif (nn_model.input_dim[0] == (None, 4)):
            #         #     test_loss, test_acc = m.evaluate(X_iris, Y_iris)
            #         # elif (nn_model.input_dim[0] == [(None, 4)]):
            #         #     X_iris_compatible = SC.convert_flat_input_to_correct_input_shape(X_iris, nn_model.input_dim)
            #         #     test_loss, test_acc = m.evaluate(X_iris_compatible, Y_iris)
            #         # print("test acc " + str(test_acc))
            #         model_correctly_loaded = True
            #
            #     except:
            #         #                                             print(e2)
            #         print("error on non-categorical too")
            # except TypeError as e3:
            #     if (str(e3).startswith("cannot unpack non-iterable float object")):
            #         print("Model not Compiled with accuracy metric")
            # except RuntimeError as e4:
            #     if (str(e4).startswith("you must compile your model before training/testing.")):
            #         print("Model has not been compiled")

        print("****************************************************************")
    #     except UnicodeEncodeError as e5:
    #         num_ignored_models += 1
    except RuntimeError as e4:
        if (str(e4).startswith("you must compile your model before training/testing.")):
            print("Model has not been compiled")
    except NotImplementedError as e5:
        print("NotImplementedError")
    except ValueError as e6:
        print("ValueError")
    except AttributeError as e6:
        print(e6)
        print("AttributeError")
    if (model_correctly_loaded):
        return nn_model

    return None