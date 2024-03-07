# MÓDULO RESPONSÁVEL PELA INTERFACE COM O USUÁRIO PARA A UTILIZAÇÃO DA REDE PERCEPTRON.
# INCLUI A LEITURA E EXIBIÇÃO DOS RELATÓRIOS.

# IMPORTAÇÕES NECESSÁRIAS:__________________________________________________________________________________

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from Modules.ct_perceptron_processing import *


# CONSTANTES GLOBAIS:_______________________________________________________________________________________

_REPORT_MAIN_TITLE_FORMATTING = '^100'
_REPORT_TITLE_FORMATTING = '-<100'
_REPORT_FIELD_FORMATTING = '<40'

_RESOURCES_DIRECTORY_PATH = "Resources\\"
_LOADING_ANIMATION_DURATION = 3


# FUNÇÕES PARA LER E APRESENTAR OS RELATÓRIOS GERADOS PELO TREINAMENTO E TESTE DA REDE:_____________________

# RELATÓRIO DE TREINAMENTO:_________________________________________________________________________________

def display_perceptron_training_report(training_report):

    training_percentage = training_report["training sample"] / training_report["total sample"] * 100
    test_percentage = training_report["test sample"] / training_report["total sample"] * 100

    print("")

    print(f"{'PERCEPTRON TRAINING REPORT':{_REPORT_MAIN_TITLE_FORMATTING}}")

    print(
        f"\n{'> INITIAL DATA: ':{_REPORT_TITLE_FORMATTING}}\n",

        f"\n{'Total Number of Samples:':{_REPORT_FIELD_FORMATTING}}{training_report["total sample"]:<10}100.0%",
        f"\n{'Number of Training Samples:':{_REPORT_FIELD_FORMATTING}}{training_report["training sample"]:<10}{training_percentage:.1f}%",
        f"\n{'Number of Test Samples:':{_REPORT_FIELD_FORMATTING}}{training_report["test sample"]:<10}{test_percentage:.1f}%",
        f"\n{'Activation Function:':{_REPORT_FIELD_FORMATTING}}Heaviside Step Function",
        f"\n{'Learning Rate:':{_REPORT_FIELD_FORMATTING}}{training_report["learning rate"]}",
        f"\n{'Number of Inputs:':{_REPORT_FIELD_FORMATTING}}{training_report["number of inputs"]}"
    )

    print("\nInitial Weights:")

    weights_counter = 1

    for weight in training_report["initial weights"]:
        print(f"{f'  W{weights_counter}:':{_REPORT_FIELD_FORMATTING}}{weight}")
        weights_counter += 1

    print(f"\n{'Initial Bias:':{_REPORT_FIELD_FORMATTING}}{training_report["initial bias"]}")

    if "progress data" in training_report:

        print(f"\n{'> PROGRESS DATA: ':{_REPORT_TITLE_FORMATTING}}")

        for epoch_data in training_report["progress data"]:
            print(
                "\nEpoch {}:".format(epoch_data["epoch"]),
                f"\n{'  Number of Hits:':{_REPORT_FIELD_FORMATTING}}{epoch_data["hits"]}",
                f"\n{'  Number of Flaws:':{_REPORT_FIELD_FORMATTING}}{epoch_data["flaws"]}"
            )

            print("\n  Final Weights:")

            weights_counter = 1

            for weight in epoch_data["final weights"]:
                print(f"{f'    W{weights_counter}:':{_REPORT_FIELD_FORMATTING}}{weight}")
                weights_counter += 1

            print(f"\n{'  Final Bias:':{_REPORT_FIELD_FORMATTING}}{epoch_data["final bias"]}")

    print(
        f"\n{'> FINAL DATA: ':{_REPORT_TITLE_FORMATTING}}\n",

        f"\n{'Number of Epochs Needed:':{_REPORT_FIELD_FORMATTING}}{training_report["epochs needed"]}",
        "\nProcessing Time:",
        f"\n{'  in nanoseconds:':{_REPORT_FIELD_FORMATTING}}{training_report["processing time in ns"]}",
        f"\n{'  in microseconds:':{_REPORT_FIELD_FORMATTING}}{training_report["processing time in ns"] / 1000}",
        f"\n{'  in milliseconds:':{_REPORT_FIELD_FORMATTING}}{training_report["processing time in ns"] / 1000000}",
        f"\n{'  in seconds:':{_REPORT_FIELD_FORMATTING}}{training_report["processing time in ns"] / 1000000000}"
    )

    print("\nFinal Weights:")

    weights_counter = 1

    for weight in training_report["final weights"]:
        print(f"{f'  W{weights_counter}:':{_REPORT_FIELD_FORMATTING}}{weight}")
        weights_counter += 1

    print(f"\n{'Final Bias:':{_REPORT_FIELD_FORMATTING}}{training_report["final bias"]}")

    print("")


# RELATÓRIO DE TESTE:_______________________________________________________________________________________

def display_perceptron_test_report(test_report):

    training_percentage = test_report["training sample"] / test_report["total sample"] * 100
    test_percentage = test_report["test sample"] / test_report["total sample"] * 100

    print("")

    print(f"{'PERCEPTRON TEST REPORT':{_REPORT_MAIN_TITLE_FORMATTING}}")

    print(
        f"\n{'> INITIAL DATA: ':{_REPORT_TITLE_FORMATTING}}\n",

        f"\n{'Total Number of Samples:':{_REPORT_FIELD_FORMATTING}}{test_report["total sample"]:<10}100.0%",
        f"\n{'Number of Training Samples:':{_REPORT_FIELD_FORMATTING}}{test_report["training sample"]:<10}{training_percentage:.1f}%",
        f"\n{'Number of Test Samples:':{_REPORT_FIELD_FORMATTING}}{test_report["test sample"]:<10}{test_percentage:.1f}%",
        f"\n{'Activation Function:':{_REPORT_FIELD_FORMATTING}}Heaviside Step Function"
    )

    print(
        f"\n{'> TESTING DATA: ':{_REPORT_TITLE_FORMATTING}}\n",

        f"\n{'Number of Hits:':{_REPORT_FIELD_FORMATTING}}{test_report["hits"]}",
        f"\n{'Number of Flaws:':{_REPORT_FIELD_FORMATTING}}{test_report["flaws"]}",
        "\nProcessing Time of a Single Sample:",
        f"\n{'  in nanoseconds:':{_REPORT_FIELD_FORMATTING}}{test_report["sample processing time in ns"]}",
        f"\n{'  in microseconds:':{_REPORT_FIELD_FORMATTING}}{test_report["sample processing time in ns"] / 1000}",
        f"\n{'  in milliseconds:':{_REPORT_FIELD_FORMATTING}}{test_report["sample processing time in ns"] / 1000000}",
        f"\n{'  in seconds:':{_REPORT_FIELD_FORMATTING}}{test_report["sample processing time in ns"] / 1000000000}"
    )

    if "true positive" in test_report:

        accuracy = (test_report["true positive"] + test_report["true negative"]) / test_report["test sample"] * 100
        recall = test_report["true positive"] / (test_report["true positive"] + test_report["false negative"]) * 100
        precision = test_report["true positive"] / (test_report["true positive"] + test_report["false positive"]) * 100
        fscore = 2 * (precision * recall / (precision + recall))


        print(
            f"\n{'> CONFUSION MATRIX: ':{_REPORT_TITLE_FORMATTING}}\n",

            "\n  Actual    - A",
            "\n  Predicted - P\n\n",

            f"{'Positive (P)':>30}{'Negative (P)':>20}",
            f"\n\n\n{'Positive (A)':<24}{test_report["true positive"]:<20}{test_report["false positive"]}",
            f"\n\n\n{'Negative (A)':<24}{test_report["false negative"]:<20}{test_report["true negative"]}"
        )
    
        print(
            "\n\nAdditional Data:",
            f"\n{'  Accuracy:':{_REPORT_FIELD_FORMATTING}}{accuracy:.2f}%",
            f"\n{'  Recall:':{_REPORT_FIELD_FORMATTING}}{recall:.2f}%",
            f"\n{'  Precision:':{_REPORT_FIELD_FORMATTING}}{precision:.2f}%",
            f"\n{'  F-score:':{_REPORT_FIELD_FORMATTING}}{fscore:.2f}%"
        )

    print("")


# FUNÇÕES PARA PLOTAR O GRÁFICO DA REDE:____________________________________________________________________

def _plot_perceptron_nn_line_graph_circle(
        ax,
        xy=(1, 1),
        radius=1, 
        color='black', 
        linewidth=None
):

    circle = Circle(
        xy= xy,
        radius= radius,
        color= color,
        linewidth= linewidth
    )

    ax.add_patch(circle)
    ax.set_aspect('equal')

def plot_perceptron_nn_line_graph(
        db_dict_sample, 
        db_dict_final_weights,
        db_dict_initial_weights=None,
        graph_title="Line Graph", 
        x_label="X1", 
        y_label="X2"
):

    x2 = db_dict_final_weights[0] / db_dict_final_weights[2]
    x1 = db_dict_final_weights[0] / db_dict_final_weights[1]

    a = (0, x2)
    b = (x1, 0)

    if db_dict_initial_weights is not None:

        x2 = db_dict_initial_weights[0] / db_dict_initial_weights[2]
        x1 = db_dict_initial_weights[0] / db_dict_initial_weights[1]

        c = (0, x2)
        d = (x1, 0)

    fig, ax = plt.subplots()

    for sample in db_dict_sample:
        _plot_perceptron_nn_line_graph_circle(
            ax= ax,
            xy= (sample[1], sample[2]),
            radius= 0.02,
            color= ('blue' if sample[3] == 1 else 'red'),
            linewidth= None
        )

    plt.axline(a, b, linewidth=2, color='black')

    if db_dict_initial_weights is not None:
        plt.axline(c, d, linewidth=2, color='green')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)

def display_perceptron_nn_line_graphs():
    plt.show()


# FUNÇÕES PARA REALIZAR A INTERFACE COM O USUÁRIO:__________________________________________________________

# OBTENÇÃO DE ENTRADAS, VALIDAÇÃO DE DADOS E ANIMAÇÕES:_____________________________________________________
    
def _input_ods_xlsx_file_name(prompt=""):
    ods_xlsx_file_name = input(prompt).strip()
    ods_xlsx_files = [ods_xlsx_file_name, ods_xlsx_file_name]

    if not ods_xlsx_file_name.endswith((".ods", ".xlsx")):
        ods_xlsx_files[0] += ".ods"
        ods_xlsx_files[1] += ".xlsx"

    return ods_xlsx_files

def _valid_ods_xlsx_file(ods_xlsx_files):
    resources_files = os.listdir(_RESOURCES_DIRECTORY_PATH)

    valid_file_name = False
    name = ""

    for file_name in ods_xlsx_files:

        if file_name in resources_files:
            valid_file_name = True
            name = file_name
            break

    return (valid_file_name, name)

def _input_number(prompt="", min=None, max=None):

    valid_input = False

    input_line = ""

    while not valid_input:
        input_line = input(prompt).strip()

        try:
            input_line = float(input_line)

        except ValueError:
            print("\t> The value must be a number. Please, try again!")

        else:
            respects_min = True
            respects_max = True
            
            if min is not None and input_line <= min:
                respects_min = False
                print(f"\t> The value must be greater than {min}. Please, try again!")

            if max is not None and input_line >= max:
                respects_max = False
                print(f"\t> The value must be lesser than {max}. Please, try again!")

            if respects_min and respects_max:
                valid_input = True
                
    return input_line

def _input_yes_no_option(prompt=""):
    
    valid_input = False

    yes_no_option = ""

    while not valid_input:
        yes_no_option = input(prompt).strip().lower()

        if yes_no_option in ("yes", "no", "y", "n"):
            valid_input = True
        else:
            print("\t> The option must be 'yes' or 'no'. Please, try again!")
    
    return yes_no_option

def _input_int_menu_option(prompt="", min=0, max=10):

    valid_option = False

    input_option = ""

    while not valid_option:
        input_option = input(prompt).strip()

        try:
            input_option = int(input_option)

        except ValueError:
            print("\t> The option must be an integer value. Please, try again!")

        else:
            if max >= input_option >= min :
                valid_option = True
            else:
                print(f"\t> The option must be between {min} and {max}. Please, try again!")
    
    return input_option

def _play_loading_animation(seconds=3):

    time_per_frame = seconds / 3

    for i in range(0, 3, 1):
        print('. ', end='', flush=True)
        time.sleep(time_per_frame)


# FUNÇÃO DE SIMULAÇÃO DA REDE PERCEPTRON:___________________________________________________________________

def _play_perceptron_nn_simulation(db_file_info=None):

    db_filenames_tuple = ()
    db_filename = ""
    db_sheet_name = ""

    learning_rate = 0
    training_rate = 0
    include_progress_data = False

    db_dict = {}

    valid_input_data = False

    while not valid_input_data:

        if db_file_info is None:
            db_filenames_tuple = _input_ods_xlsx_file_name(">> Input the database file name ('.ods' or '.xlsx'): ")
            valid_input_data, db_filename = _valid_ods_xlsx_file(db_filenames_tuple)

        else:
            db_filename = db_file_info[0]
            db_sheet_name = db_file_info[1]
            valid_input_data = True

        if not valid_input_data:
            print("\t> Invalid file name. Please, try again!")
            print("\t  OBS: Make sure your file is in the 'Resources' directory.\n")

        else:
            if db_file_info is None:
                db_sheet_name = input(">> Input the sheet name: ").strip()
                print("")

            print("NETWORK'S TRAINING DATA:\n")

            learning_rate = _input_number(">> Learning rate: ", min= 0, max= None)
            training_rate = _input_number(">> Training sample percentage: ", min= 0, max= 100)

            db_reading_error = True
    
            while db_reading_error:

                try:     
                    db_dict = import_processed_db_from_ods_xlsx(
                        path= f"{_RESOURCES_DIRECTORY_PATH}{db_filename}",
                        sheet_name= f"{db_sheet_name}",
                        training_rate= training_rate
                    )

                except ValueError:
                    valid_input_data = False

                    print("\n> Error while reading the database file. Please, verify the sheet name:")
                    print("  OBS: If the error persists, try checking out your database file formatting.")

                    db_sheet_name = input("\n>> Input the sheet name: ").strip()
                
                else:
                    db_reading_error = False
                    valid_input_data = True

    print("\nInclude training progress data on report? ('yes' or 'no')")
    include_progress_data = _input_yes_no_option(">> ")

    try:
        training_report = train_perceptron_nn(
            db_dict= db_dict,
            learning_rate= learning_rate,
            include_progress_data= (True if include_progress_data in ('yes', 'y') else False)
        )

    except:
        print("\n> ERROR RAISED ON TRAINING STAGE!")
        print("  Please, verify your database file formatting and your simulation's configurations.\n")

    else:
        try:
            test_report = test_perceptron_nn(
                db_dict= db_dict,
                include_confusion_matrix= True
            )

        except:
            print("\n> ERROR RAISED ON TEST STAGE!")
            print("  Please, verify your database file formatting and your simulation's configurations.\n")

        else:
            print("\n>> PROCESSING NETWORK ", end='')
            _play_loading_animation(_LOADING_ANIMATION_DURATION)
            
            print("\n\n")

            try:
                display_perceptron_training_report(training_report)
                print("\n")
            except:
                print("\n> ERROR RAISED ON TRAINING REPORT GENERATION!")

            try:
                display_perceptron_test_report(test_report)
            except:
                print("\n> ERROR RAISED ON TESTING REPORT GENERATION!")

            print(
                "\n\nPlot results graphs? ('yes' or 'no')",

                "\n\nOBS:",
                "\n  - The initial line is represented in GREEN.",
                "\n  - The final lines are represented in BLACK.",

                sep=''
            )

            plot_graphs = _input_yes_no_option(">> ")

            if plot_graphs in ('yes', 'y'):

                initial_weights = training_report["initial weights"].copy()
                initial_weights.insert(0, training_report["initial bias"])

                try:
                    plot_perceptron_nn_line_graph(
                        db_dict_sample= db_dict["training"],
                        db_dict_final_weights= db_dict["weights"],
                        db_dict_initial_weights= initial_weights,
                        graph_title= "Training Results Graph",
                        x_label= "X1",
                        y_label= "X2"
                    )

                    plot_perceptron_nn_line_graph(
                        db_dict_sample= db_dict["test"],
                        db_dict_final_weights= db_dict["weights"],
                        db_dict_initial_weights= None,
                        graph_title= "Testing Results Graph",
                        x_label= "X1",
                        y_label= "X2"
                    )

                    display_perceptron_nn_line_graphs()

                except:
                    print("\n> ERROR RAISED! The system could not plot the graphs.")

    return (db_filename, db_sheet_name)


# FUNÇÃO PARA EXECUTAR O SIMULADOR DA REDE, O QUAL PODE REALIZAR MÚLTIPLAS SIMULAÇÕES:______________________

def play_perceptron_nn_simulator():
    
    os.system("cls")

    simulations_counter = 1
    use_same_db = 'no'

    playing_simulator = True

    while playing_simulator:

        print(f"\n{'PERCEPTRON NEURAL NETWORK SIMULATOR':{_REPORT_MAIN_TITLE_FORMATTING}}\n")

        print(f"{f'SIMULATION {simulations_counter}: ':{_REPORT_TITLE_FORMATTING}}\n")

        if use_same_db in ('yes', 'y'):
            _play_perceptron_nn_simulation(db_file_info)
        else:
            db_file_info =  _play_perceptron_nn_simulation()

        simulations_counter += 1

        print(
            f"\n{'OPTIONS MENU: ':{_REPORT_TITLE_FORMATTING}}\n",

            "\n[1] Simulate Again (Clear the console);",
            "\n[2] Simulate Again (Keep the console);",
            
            "\n\n[0] End Simulator."
        )

        option = _input_int_menu_option("\n>> ", min= 0, max= 2)

        if option == 0:
            playing_simulator = False
        else:
            print("\nUse the same database file and sheet name? ('yes' or 'no')")
            use_same_db = _input_yes_no_option(">> ")

            if option == 1:
                os.system("cls")
    
    print("\n>> END OF SIMULATOR.\n")
