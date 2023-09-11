import numpy as np

def find_missing_intervals(arr:np.array) -> list:
    """
    Функция для поиска пропущенных интервалов во времекнном ряду

    Args:
        arr (np.array): массив содержащий временной ряд 

    Returns:
        list: список котртежей с индексами обозначающими пропущенные интервалы
    """    
    is_missing = np.isnan(arr)  # Создаем булевый массив с пометками о пропущенных значениях
    split_indices = np.where(np.diff(np.concatenate(([0], is_missing))) != 0)[0] # Находим индексы разделения участков с пропущенными значениями
    split_indices = list(split_indices) + [len(arr)] 
    # Формируем список кортежей с началом и концом каждого участка с пропущенными значениями
    missing_ranges = [(split_indices[i], split_indices[i + 1]-1) for i in range(0, len(split_indices) - 1, 2)]
    return missing_ranges
    

def find_depths_with_multiple_logs(df, logs):
    """
    Функция для поиска глубин, на которых определено более чем одно значение каротажа.

    Args:
        df (pd.DataFrame): DataFrame с данными каротажей.
        logs (List[str]): Список названий каротажей для проверки.

    Returns:
        List[Tuple[float, float]]: Список интервалов глубин, на которых имеется перекрытие.
    """
    mask = df[logs].notna().sum(axis=1) > 1
    depths = df[mask].index.tolist()

    if not depths:
        return []

    intervals = []
    start_depth = depths[0]
    end_depth = depths[0]
    
    for i in range(1, len(depths)):
        if depths[i] - end_depth == df.index[1] - df.index[0]:
            end_depth = depths[i]
        else:
            intervals.append((start_depth, end_depth))
            start_depth = depths[i]
            end_depth = depths[i]

    intervals.append((start_depth, end_depth))
    return intervals



#########################################################################################################
######################################| ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ |#######################################
#########################################################################################################

def plot_all_logs_with_overlap(df, intervals, logs):
    """
    Функция для визуализации каротажей и подсветки интервалов, на которых определено более чем одно значение.

    Args:
        df (pd.DataFrame): DataFrame с данными каротажей.
        intervals (List[Tuple[float, float]]): Список интервалов глубин для подсветки.
        logs (List[str]): Список названий каротажей для отображения.

    Returns:
        None: Функция выводит график, но ничего не возвращает.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(logs), figsize=(2*len(logs), 16))

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, log in enumerate(logs):
        axes[i].plot(df[log], df.index, label=log, color=colors[i])
        axes[i].set_title(log)
        axes[i].invert_yaxis()
        axes[i].set_xlabel("Value")
        if i > 0:  # Отображаем метки глубины только для первого каротажа
            axes[i].set_yticks([])
        else:
            axes[i].set_ylabel("Dept")
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

        for start, end in intervals:
            axes[i].axhline(y=start, color='black', linestyle='-.')
            axes[i].axhline(y=end, color='black', linestyle='-.')
            axes[i].axhspan(start, end, facecolor='yellow', alpha=0.5)
    
    plt.tight_layout()
    plt.show()