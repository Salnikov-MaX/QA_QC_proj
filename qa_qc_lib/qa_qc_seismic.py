#import os
import segyio
import numpy as np
from shapely.geometry import MultiPoint, Polygon, Point
import matplotlib.pyplot as plt
import datetime
from typing import Any


class QA_QC_seismic():
    def __init__(self, file_path: str, license_area_poly: list = None, surfaces_path_list: list = None, faults_file_path: str = None) -> None:
        
        self.seismic_cube, self.coordinate_x, self.coordinate_y, self.coordinate_z = self.__get_seismic_grid(file_path)
        self.cube_poly = Polygon(build_polygon_from_points(self.coordinate_x, self.coordinate_y))

        self.license_area_poly = license_area_poly    # полигон лицензионного участка
        self.surfaces_path_list = surfaces_path_list  # список путей к файлам с поверхностями структурных карт / карт изохрон
        self.faults_file_path = faults_file_path

        self.report_text = ""
        self.ident = ' '*5   # отступ при формировании отчета


    def __get_seismic_grid(self, segy_file_path: str):
        """
        Метод предназначенный для чтения сейсмического куба и метаданных из файла формата SEG-Y 

        Args:
            segy_file_path (str): Путь к файлу с сейсмическими данными

        Returns:
            _type_: (куб сейсмических трасс, вектор координат X каждой из трасс, вектор координат Y каждой из трасс, вектор глубин)
        """
        segy = segyio.open(segy_file_path, 'r', strict=False)     # Открываем SEGY-файл в режиме чтения
        coordinate_x = segy.attributes(segyio.TraceField.SourceX)
        coordinate_y = segy.attributes(segyio.TraceField.SourceY)
        coordinate_z = segy.samples
        seismic_data = segyio.tools.cube(segy)
        return seismic_data, np.array(coordinate_x), np.array(coordinate_y), coordinate_z

    
    def __open_irap_ascii_grid(self, path):
        """
        Метод для чтения карт изохрон/структурных карт в формате irap

        Args:
            path (_type_): путь к irap файлу 

        Returns:
            _type_: минимальное значение на карте, максимальное значение на карте, полигон в котором лежит карта
        """        
        with open(path, 'r') as text:
            text = text.read().replace('\n', ' ').split(' ') 
            text = [float(i) for i in text if i != '']
            grid_values = np.array(text[19:]).reshape(int(text[1]), int(text[9-1])) # хардкодом заданы значения из шапки файла согласно его структуре

            min_val, max_val = grid_values[grid_values != 9999900.0].min(), grid_values[grid_values != 9999900.0].max() 
            min_x, max_x = text[4], text[5]
            min_y, max_y = text[6], text[7]
            rectangle_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

        return min_val, max_val, rectangle_points


    def __parse_faults(self, path: str) -> dict:
        """
        Функция парсит файл с разломами

        Args:
            path (str): Путь к файлу с разломами

        Returns:
            dict: Словарь с разломами, где keys - название разлома, values - (x, y, z) координаты точек разлома.
        """    
        fault_dict = {}
        with open(path, 'r') as fault:
            for line in fault:
                words = line.strip().split()
                if len(words) >= 7:
                    u = words[6]
                    r = float(words[3])
                    t = float(words[4])
                    y = float(words[5])
                if u not in fault_dict:
                    fault_dict[u] = []
                fault_dict[u].append((r, t, y))
        return fault_dict


    def test_coordinate_validation(self, get_report=True) -> None:
        """
        Оценка корректности координат загруженного куба.
        Метод проверяет вхождение сейсического куба в границы лицензионного участка 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """
        # Проверяем наличие данных для запуска теста
        if self.coordinate_x is None or self.coordinate_y is None or self.license_area_poly is None:
            coordinate_existence = self.coordinate_x is not None or self.coordinate_y is not None
            license_area_poly_existence = self.license_area_poly is not None
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Наличие координат:{coordinate_existence}, Наличие границ лицензионного участка:{license_area_poly_existence}'
            if get_report: print('\n'+report_text) 
        # Непосредственно проведение теста
        else:
            polygon2 = Polygon(self.license_area_poly)
            if self.cube_poly.intersects(polygon2):
                intersection_area = self.cube_poly.intersection(polygon2)
                percentage_inside = (intersection_area.area / self.cube_poly.area) * 100
                report_text = f"{self.ident}Тест пройден успешно. \n{self.ident}Процент вхождения сейсмического куба в границы лицензионного участка {round(percentage_inside, 2)}%"

            else:
                intersection_area, percentage_inside = None, None
                report_text = f"{self.ident}Тест не пройден. \n{self.ident}Сейсмический куб не входит в границы лицензионного участка"
            
            if get_report: visualize_intersection(self.cube_poly, polygon2, intersection_area, report_text) 

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_coordinate_validation:\n{report_text}\n\n"


    def test_monotony(self, get_report=True):
        """
        Метод проверяет ось глубин / времени на монотонное возрастание (каждое следующее значение больше предыдущего)

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """
        if sum(np.diff(self.coordinate_z) <= 0) == 0:
            report_text = f"{self.ident}Тест пройден успешно. \n{self.ident}Отметки оси глубин/времени монотонно возрастают"
        else:
            report_text = f"{self.ident}Тест не пройден. \n{self.ident}Отметки оси глубин/времени не возрастают монотонно"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_monotony:\n{report_text}\n\n"

        if get_report: print('\n'+report_text)


    def test_miss_traces(self, get_report=True):
        """
        Метод проверяет сейсмический куб на наличие пропущенных / не записанных сейсмотрасс 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """
        seismic_cube_r = self.seismic_cube.reshape(-1, self.seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)

        percent_true = round((np.sum(mask) / mask.size) * 100, 1)
        percent_false = 100 - percent_true

        test_result = 'Тест пройден успешно.' if percent_false == 100 else 'Тест не пройден.'
        report_text = f'{self.ident}{test_result}\n{self.ident}Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_miss_traces:\n{report_text}\n\n"

        if get_report: visualize_miss_traces(mask.reshape((self.seismic_cube.shape[0], 
                                                           self.seismic_cube.shape[1])),
                                             percent_false)


    def test_surfaces_location_validation(self, get_report=True):
        """
        Метод оценивает соответствие отражающего горизонта сейсмическому кубу

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """        
        # Проверка наличия данных для проведения теста
        if not self.surfaces_path_list:
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Данные о поверхностях не были переданы'
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text) 
        
        # Непосредственно проведение теста
        else:
            cube_z_min, cube_z_max = self.coordinate_z.min(), self.coordinate_z.max()
            # Проведем тест для каждой поврхности отдельно
            for path in self.surfaces_path_list:
                try:
                    min_val, max_val, rectangle_points = self.__open_irap_ascii_grid(path)
                    # проверяем совпадение по X и Y коррдинатам
                    polygon2 = Polygon(rectangle_points)
                    x_y_coords_validation = self.cube_poly.intersects(polygon2)
                    # проверяем совпадение по Z коррдинатe
                    z_coords_validation = cube_z_min <= min_val and cube_z_max >= max_val

                    # формируем отчет о прохождении теста
                    test_result = x_y_coords_validation and z_coords_validation
                    text = 'Тест пройден успешно.' if test_result else 'Тест не пройден.'
                    text_2 = '' if test_result else 'не '
                    report_text = f'{self.ident}{text}\n{self.ident}Путь к файлу:"{path}"; отражающий горизонт {text_2}попадает в границы сейсмического куба'
                    if not test_result: report_text = report_text + f' (совпадение по X,Y:{x_y_coords_validation}, по вертикальной шкале:{z_coords_validation})'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text)

                except FileNotFoundError: 
                    report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Некорректный путь к файлу:"{path}"'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text)


    def test_faults_location_validation(self, get_report=True):
        """
        Метод оценивает соответствие пикировки разлома сейсмическому кубу

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """ 
        # Проверка наличия данных для проведения теста
        if not self.faults_file_path:
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Данные о разломах не были переданы'
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text) 
        
        # Непосредственно проведение теста
        else: 
            try:
                fault_dict = self.__parse_faults(self.faults_file_path)
                cube_z_min, cube_z_max = self.coordinate_z.min(), self.coordinate_z.max()
                # В одном файле находится несколько разломов, проведем тест для каждого из них
                for key in fault_dict.keys():
                    points = fault_dict[key]
                    res = []
                    for point in points:
                        # проверяем совпадение по X и Y коррдинатам
                        x_y_coords_validation = Point((point[0], point[1])).within(self.cube_poly)
                        # проверяем совпадение по вертикальной оси
                        z_coords_validation = cube_z_min <= point[2] and cube_z_max >= point[2]
                        res.append(x_y_coords_validation and z_coords_validation)
                    income_points_percent = round((sum(res) * 100 / len(res)), 2)
                    test_result = income_points_percent != 0

                    # формируем отчет о прохождении теста
                    test_result = x_y_coords_validation and z_coords_validation
                    text = 'Тест пройден успешно.' if test_result else 'Тест не пройден.'
                    report_text = f'{self.ident}{text}\n{self.ident}Разлом:"{key}"; {income_points_percent}% точек разлома из {len(points)} входит в границы сейсмического куба'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text)

            except FileNotFoundError: 
                report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Некорректный путь к файлу:"{self.faults_file_path}"'
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                if get_report: print('\n'+report_text)


    def get_list_of_tests(self) -> list:
        """
        Метод для получения списка тестов для данных реализованных в классе QA_QC_seismic

        Returns:
            list: список с названиями методов реализующих тесты
        """        
        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    
    def get_method_description(self, method_name: str) -> str:
        """
        Метод для получение описания теста по его названию

         Args:
             method_name(str) - название теста
        Returns:
             str - описание теста
         """
        method = getattr(self, method_name, None)
        if method is not None:
            return method.__doc__
        else:
            return "Метод не найден."


    def start_tests(self, list_of_tests: list, get_report=True):
        """
        Метод который запускает все тесты, которые переданы в виде списка list_of_tests

        Args:
            list_of_tests (list): список названий тестов которые должны быть проведены
        """        
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method(get_report=get_report)


    def generate_test_report(self, file_name='test_report', file_path='report', data_name='Неизвестный файл'):
        """
        Метод для генерации отчета в виде текстового файла

        Args:
            file_name (str, optional): название файла с отчетом. Defaults to 'test_report'.
            file_path (str, optional): директория в которую следует сохранить отчет. Defaults to 'report'.
            data_name (str, optional): название данных который подвергались тестированию. 
                                       Данное название отобразится в итоговом отчете. Defaults to 'Неизвестный файл'.
        """        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        report = f"Отчет о тестировании от {timestamp}{self.ident}Название тестируемого файла: '{data_name}'\n\n{self.report_text} "
        with open(f"{file_path}/{file_name}.txt", "w") as file:
            file.write(report)
    





#########################################################################################################
###########################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ |########################################
#########################################################################################################

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


def build_polygon_from_points(x_coords:np.array, y_coords:np.array):
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


