#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_seismic |##############################
#########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import MultiPoint
from scipy.signal import convolve2d
from .math_tools import compute_variance


def visualize_intersection(polygon1, polygon2, intersection_area=None, text=None):
    """
    Функция формирующая визуальный отчет о проведении теста test_coordinate_validation

    Args:
        polygon1 (_type_): полигон-проекция сейсмического куба
        polygon2 (_type_): полигон границы лицензионного участка
        intersection_area (_type_, optional): пололигон отражающий площадь пересечения polygon1 и polygon2. Defaults to None.
        text (_type_, optional): текст с выводами результата тестирования который будет отображен в визуальном отчете. Defaults to None.
    """
    fig, ax = plt.subplots()

    x_coords1, y_coords1 = polygon1.exterior.xy
    ax.plot(x_coords1, y_coords1, label='Проекция сейсмического куба', color='red')

    x_coords2, y_coords2 = polygon2.exterior.xy
    ax.plot(x_coords2, y_coords2, label='Граница лицензионного участка', color='orange')

    if intersection_area:
        intersection_x, intersection_y = intersection_area.exterior.xy
        ax.fill(intersection_x, intersection_y, color='red', alpha=0.5, label='Зона пересечения полигонов')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Проверка корректности координат')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.945))
    ax.grid()

    if text:
        ax.text(0.99, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment='center', style='italic')

    fig.set_size_inches(8, 8)
    plt.show()


def visualize_miss_traces(mask, percent_false):
    """
    Функция формирующая визуальный отчет о проведении теста test_miss_traces

    Args:
        mask (_type_): 2D масиив с булевими значениями, где True означает пыстые сейсмические трассы
        percent_false (_type_): процент отсутствующих сейсмических трасс от общего их колличества в кубе
    """
    colors = ['red', 'blue']
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


def build_polygon_from_points(x_coords: np.array, y_coords: np.array):
    """
    Получение полигона из облака точек

    Args:
        x_coords (np.array): X координаты облака точек
        y_coords (np.array): Y координаты облака точек

    Returns:
        list: список кортежей координат полигона в который вписано облако точек
    """
    points = MultiPoint(list(zip(x_coords, y_coords)))
    convex_hull = points.convex_hull
    return list(convex_hull.exterior.coords)


def find_border(mask):
    """
    Находит границу в бинарной маске.

    Args:
        mask (numpy.ndarray): Бинарная маска, где True - это интересующий объект, а False - фон.

    Returns:
        numpy.ndarray: Бинарное изображение с той же размерностью, что и mask, где True отмечает границы объекта.
    """
    # Создаем 3x3 ядро (kernel) с единицами
    kernel = np.ones((3, 3))
    # Применяем свертку к маске с ядром
    conv_result = convolve2d(mask, kernel, mode='same')
    # Идентифицируем границы: все пиксели в результирующем изображении, которые меньше 9
    # (и соответствующие пиксели в исходной маске равны True), являются границами.
    border = np.logical_and(conv_result < 9, mask)
    return border


def best_split_point(variance_list):
    """
    Находит оптимальную точку разделения кривой, основываясь на дисперсии производных.

    Args:
        variance_list (list or numpy.ndarray): Одномерный массив или список значений дисперсии.

    Returns:
        int: Индекс, на котором достигается наибольшая разница в дисперсии производных слева и справа.
    """
    # Вычисляем производные кривой
    derivatives = np.diff(variance_list)
    # Инициализируем переменную для хранения максимальной разницы в дисперсии и лучшей точки разделения
    best_point, max_variance_diff = 0, 0
    # Перебор всех возможных точек разделения
    for i in range(1, len(derivatives)):
        # Вычисляем дисперсию для производных слева и справа от текущей точки
        left_variance = compute_variance(derivatives[:i])
        right_variance = compute_variance(derivatives[i:])
        # Вычисляем разницу в дисперсиях
        variance_diff = np.abs(left_variance - right_variance)
        # Если текущая разница в дисперсиях больше максимальной, обновляем максимальное значение и индекс лучшей точки
        if variance_diff > max_variance_diff:
            max_variance_diff = variance_diff
            best_point = i
    return best_point


def visualize_edge_zone_evaluation(mask, variance_list, split_point):
    """
    Визуализирует оценку наличия краевой зоны на основе маски и списка дисперсий.

    Args:
        mask (numpy.ndarray): Маска, содержащая значения:
                              0 - Сейсмотрассы отсутствуют
                              1 - Основная часть сейсмического куба
                              2 - Краевая часть сейсмического куба.
        variance_list (list ): Список дисперсий для последовательности.
        split_point (int): Индекс разделения для отсечки краевой зоны.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 2]})
    # Добавляем общий заголовок для всей фигуры
    fig.suptitle("Оценка наличия краевой зоны", fontsize=16)
    # Визуализация графика дисперсии
    x_vals = np.arange(1, len(variance_list) + 1)  # Начало с 1
    axes[0].plot(x_vals, variance_list, label="Дисперсия", color="blue")
    axes[0].axvline(x=split_point, color="red", linestyle="--", label=f"Ширина краевой зоны: {split_point + 1}")
    axes[0].text(split_point, variance_list[split_point - 1] + 0.05 * (max(variance_list) - min(variance_list)),
                 f"{split_point + 1}", color='black', ha="left", va="bottom")
    axes[0].legend()
    axes[0].set_xlabel("Ширина краевой зоны в количестве дискретов")
    axes[0].set_ylabel("Дисперсия")

    # Визуализация маски
    im = axes[1].imshow(mask)
    axes[1].grid(ls=':', alpha=.5)
    axes[1].invert_yaxis()
    colors = im.cmap(im.norm(range(3)))
    labels = [
        'Сейсмотрассы отсутствуют',
        'Основная часть сейсмического куба',
        'Краевая часть сейсмического куба'
    ]
    legend_labels = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    axes[1].legend(handles=legend_labels, loc='center left', bbox_to_anchor=(1, 0.945))
    plt.tight_layout()
    plt.show()
