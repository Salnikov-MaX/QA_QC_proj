#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import inspect
import re

import numpy as np
from matplotlib import pyplot as plt

def find_test_methods_with_params(params, modul):
    test_methods_with_params = {}

    for method_name, method in inspect.getmembers(modul):
        if inspect.ismethod(method) and method_name.startswith("test"):
            description = inspect.getdoc(method)
            if description:
                # Используем регулярное выражение для извлечения параметров из описания метода
                matches = re.findall(r'Required data:\s*(.*)', description)

                if matches:
                    method_params = [param.strip() for param in matches[0].split(";")]
                    # Проверяем, есть ли переданные параметры в описании метода
                    if set(method_params).issubset(params) and set(method_params).issubset(params):
                        for i in range(len(method_params)):
                            try:
                                test_methods_with_params[method_params[i]].append(method_name)
                            except KeyError:
                                test_methods_with_params[method_params[i]]=[]
                                test_methods_with_params[method_params[i]].append(method_name)

    return test_methods_with_params
