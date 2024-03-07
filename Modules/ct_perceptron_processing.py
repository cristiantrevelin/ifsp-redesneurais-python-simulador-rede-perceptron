# MÓDULO RESPONSÁVEL PELA IMPORTAÇÃO DOS DADOS E PROCESSAMENTO DA REDE PERCEPTRON.

# IMPORTAÇÕES NECESSÁRIAS:__________________________________________________________________________________

import pandas
import random
import time


# EXTRAINDO ESTRUTURA DE LISTAS DO BANCO DE DADOS:__________________________________________________________
#   - [ Número de Entradas da Rede, 
#       [Lista de Pesos + Bias], 
#       [Lista de Registros (-1, Entradas, Saída)] ] :

def _get_db_weight_list(number_of_inputs):
    weights_list = []

    for i in range(0, number_of_inputs, 1):
        weights_list.append(random.random())

    return weights_list

def _get_db_inputs_list(db_ods):
    inputs_list = []

    for i in range(0, len(db_ods), 1):
        item = [-1]

        for key in db_ods.keys():
            item.append(db_ods[key][i])
        
        inputs_list.append(item)
    
    return inputs_list

def _get_db_array_struct(db_ods):
    array_struct = []

    number_of_inputs = len(db_ods.keys())
    array_struct.append(number_of_inputs)

    weights_list = _get_db_weight_list(number_of_inputs)
    array_struct.append(weights_list)
    
    inputs_list = _get_db_inputs_list(db_ods)
    random.shuffle(inputs_list)
    array_struct.append(inputs_list)

    return array_struct


# EXTRAINDO ESTRUTURA FINAL DO BANCO DE DADOS A PARTIR DA ESTRUTURA DE LISTAS:______________________________
#   - { "number of inputs": Número de Entradas da Rede
#       "weights": [Lista de Pesos + Bias],
#       "training": [Lista de Registros de Treinamento (-1, Entradas, Saída)],
#       "test": [Lista de Registros de Teste (-1, Entradas, Saída)] } :

def _get_db_dict_struct(db_array_struct, training_rate=70):
    dict_struct = {}

    dict_struct.update({"number of inputs": db_array_struct[0]})
    dict_struct.update({"weights": db_array_struct[1]})
    dict_struct.update({"training": []})
    dict_struct.update({"test": []})

    training_amount = (training_rate / 100) * len(db_array_struct[2]) 
    training_amount = round(training_amount)

    for i in range(0, len(db_array_struct[2]), 1):
        
        if i < training_amount:
            dict_struct["training"].append(db_array_struct[2][i])
        else:
            dict_struct["test"].append(db_array_struct[2][i])

    return dict_struct


# FUNÇÃO FINAL DE IMPORTAÇÃO DO BANCO DE DADOS ODS:_________________________________________________________

def import_processed_db_from_ods_xlsx(
        path, 
        sheet_name=0,
        training_rate=70
):

    db_ods = pandas.read_excel(
        path, 
        sheet_name= sheet_name, 
        engine= ("odf" if path.endswith(".ods") else "openpyxl")
    )

    db_array_struct = _get_db_array_struct(db_ods)
    db_dict_struct = _get_db_dict_struct(db_array_struct, training_rate)

    return db_dict_struct


# FUNÇÕES DE TREINAMENTO E TESTE DA REDE:___________________________________________________________________

def _multiply_lists(list1, list2):
    product = 0

    if len(list1) == len(list2):

        for i in range(0, len(list1), 1):
            product += list1[i] * list2[i]
        
    return product

def _heaviside_function(u, c=0):
    return 0 if u < c else 1

def _update_weights(
        current_weights,
        learning_rate,
        inputs,
        desired_output,
        real_output
):

    for i in range(0, len(current_weights), 1):
        current_weights[i] = current_weights[i] + learning_rate * (desired_output - real_output) * inputs[i]


# FUNÇÕES FINAIS DE TREINAMENTO E TESTE DA REDE. AMBAS RETORNAM UM DICIONÁRIO DE RELATÓRIO:_________________

# TREINAMENTO:______________________________________________________________________________________________
        
def train_perceptron_nn(
        db_dict, 
        learning_rate=0.1, 
        include_progress_data=True
):
    
    report = {
        "total sample": len(db_dict["training"]) + len(db_dict["test"]),
        "training sample": len(db_dict["training"]),
        "test sample": len(db_dict["test"]),
        "learning rate": learning_rate,
        "number of inputs": db_dict["number of inputs"],
        "initial weights": db_dict["weights"][1:],
        "initial bias": db_dict["weights"][0],

        "epochs needed": 0,
        "processing time in ns": 0,
        "final weights": [],
        "final bias": 0
    }

    if include_progress_data:
        report.update({"progress data": []})

    epochs_counter = 0
    stop_learning = False

    processing_time = time.perf_counter_ns()

    while not stop_learning:
        flawless_epoch = True

        if include_progress_data:
            epoch_data = {
                "epoch": 0, 
                "flaws": 0, 
                "hits": 0, 
                "final weights": [], 
                "final bias": 0
            }

        for sample in db_dict["training"]:

            u = _multiply_lists(db_dict["weights"], sample[:db_dict["number of inputs"]])
            y = _heaviside_function(u)

            desired_output = sample[db_dict["number of inputs"]]

            if y != desired_output:
                flawless_epoch = False

                if include_progress_data:
                    epoch_data["flaws"] += 1

                _update_weights(
                    current_weights= db_dict["weights"],
                    learning_rate= learning_rate, 
                    inputs= sample[:db_dict["number of inputs"]], 
                    desired_output= desired_output,
                    real_output= y)
                
            elif include_progress_data:
                epoch_data["hits"] += 1
        
        epochs_counter += 1

        if include_progress_data:
            epoch_data["epoch"] = epochs_counter
            epoch_data["final weights"] = db_dict["weights"][1:]
            epoch_data["final bias"] = db_dict["weights"][0]

            report["progress data"].append(epoch_data)

        if flawless_epoch:
            stop_learning = True
    
    processing_time = time.perf_counter_ns() - processing_time
    
    report["epochs needed"] = epochs_counter
    report["processing time in ns"] = processing_time
    report["final weights"] = db_dict["weights"][1:]
    report["final bias"] = db_dict["weights"][0]

    return report


# TESTE:____________________________________________________________________________________________________

def test_perceptron_nn(db_dict, include_confusion_matrix=True):
    
    report = {
        "total sample": len(db_dict["training"]) + len(db_dict["test"]),
        "training sample": len(db_dict["training"]),
        "test sample": len(db_dict["test"]),
        "hits": 0,
        "flaws": 0,
        "sample processing time in ns": 0
    }

    if include_confusion_matrix:
        report.update({
            "true positive": 0,
            "false positive": 0,
            "true negative": 0,
            "false negative": 0
        })

    first_sample = True

    for sample in db_dict["test"]:

        if first_sample:
            sample_processing_time = time.perf_counter_ns()

        u = _multiply_lists(db_dict["weights"], sample[:db_dict["number of inputs"]])
        y = _heaviside_function(u)

        desired_output = sample[db_dict["number of inputs"]]

        if y != desired_output:
            report["flaws"] += 1

            if include_confusion_matrix:
                if desired_output == 0:
                    report["false positive"] += 1
                else:
                    report["false negative"] += 1
        else:
            report["hits"] += 1

            if include_confusion_matrix:
                if desired_output == 0:
                    report["true negative"] += 1
                else:
                    report["true positive"] += 1

        if first_sample:
            sample_processing_time = time.perf_counter_ns() - sample_processing_time
            first_sample = False
    
    report["sample processing time in ns"] = sample_processing_time

    return report
