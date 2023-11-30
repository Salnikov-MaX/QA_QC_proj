#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_seismic |##############################
#########################################################################################################

import matplotlib.pyplot as plt


#########################################################################################################
######################################| ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ |#######################################
#########################################################################################################


def generate_report_test_miss_traces(test_dict:dict):
    """
    Функция формирующая визуальный отчет о проведении теста test_miss_traces

    Args:
        mask (_type_): 2D масиив с булевими значениями, где True означает пыстые сейсмические трассы
        percent_false (_type_): процент отсутствующих сейсмических трасс от общего их колличества в кубе
    """  
    mask = test_dict['specification']["wrong_values"]
    percent_false = test_dict['specification']["percent_false"]

    colors = ['red','blue']
    percent_true = 100 - percent_false
    labels = [f'Сейсмические трассы отсутствуют ({percent_true:.1f}%)',
              f'Сейсмические трассы присутствуют ({percent_false:.1f}%)']

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='seismic')

    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=labels[i]) for i in range(len(colors))]
   
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.09))
    plt.grid(ls=':', alpha=.5)
    plt.gca().invert_yaxis()
    plt.show()




