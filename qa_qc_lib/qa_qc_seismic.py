import segyio
import numpy as np
from shapely.geometry import Polygon, Point
import datetime
from .qa_qc_main import QA_QC_main
from .qa_qc_tools.seismic_tools import *
from .qa_qc_tools.math_tools import compute_variance


class QA_QC_seismic(QA_QC_main):
    def __init__(self, file_path: str, license_area_poly: list = None, surfaces_path_list: list = None, faults_file_path: str = None) -> None:
        
        super().__init__()
        self.seismic_cube, self.coordinate_x, self.coordinate_y, self.coordinate_z = self.__get_seismic_grid(file_path)
        self.file_name = file_path.split('/')[-1]
        self.cube_poly = Polygon(build_polygon_from_points(self.coordinate_x, self.coordinate_y))

        self.license_area_poly = license_area_poly    # полигон лицензионного участка
        self.surfaces_path_list = surfaces_path_list  # список путей к файлам с поверхностями структурных карт / карт изохрон
        self.faults_file_path = faults_file_path


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


    def test_coordinate_validation(self, get_report=True) -> dict:
        """
        Оценка корректности координат загруженного куба.
        Метод проверяет вхождение сейсического куба в границы лицензионного участка 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """
        all_results_dict = {}
        # Проверяем наличие данных для запуска теста
        if self.coordinate_x is None or self.coordinate_y is None or self.license_area_poly is None:
            coordinate_existence = self.coordinate_x is not None or self.coordinate_y is not None
            license_area_poly_existence = self.license_area_poly is not None
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Наличие координат:{coordinate_existence}, Наличие границ лицензионного участка:{license_area_poly_existence}'
            if get_report: print('\n'+report_text) 
            all_results_dict["result"] = 'Fail'
        # Непосредственно проведение теста
        else:
            polygon2 = Polygon(self.license_area_poly)
            if self.cube_poly.intersects(polygon2):
                intersection_area = self.cube_poly.intersection(polygon2)
                percentage_inside = (intersection_area.area / self.cube_poly.area) * 100
                report_text = f"{self.ident}Тест пройден успешно. \n{self.ident}Процент вхождения сейсмического куба в границы лицензионного участка {round(percentage_inside, 2)}%"
                all_results_dict["result"] = 'True'

            else:
                intersection_area, percentage_inside = None, None
                report_text = f"{self.ident}Тест не пройден. \n{self.ident}Сейсмический куб не входит в границы лицензионного участка"
                all_results_dict["result"] = 'False'
            
            if get_report: visualize_intersection(self.cube_poly, polygon2, intersection_area, report_text) 

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_coordinate_validation:\n{report_text}\n\n"
        return all_results_dict | {"file_name": self.file_name, "date": timestamp}


    def test_monotony(self, get_report=True) -> dict:
        """
        Метод проверяет ось глубин / времени на монотонное возрастание (каждое следующее значение больше предыдущего)

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """
        result_mask = np.diff(self.coordinate_z) <= 0
        result_mask = np.insert(result_mask, 0, False)
        result = sum(result_mask) == 0
        if result:
            report_text = f"{self.ident}Тест пройден успешно. \n{self.ident}Отметки оси глубин/времени монотонно возрастают"
        else:
            report_text = f"{self.ident}Тест не пройден. \n{self.ident}Отметки оси глубин/времени не возрастают монотонно"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_monotony:\n{report_text}\n\n"
        if get_report: print('\n'+report_text)

        return {"result": result, "wrong_values": ~result_mask, "file_name": self.file_name, "date": timestamp}


    def test_miss_traces(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический куб на наличие пропущенных / не записанных сейсмотрасс 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """
        seismic_cube_r = self.seismic_cube.reshape(-1, self.seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)

        percent_true = round((np.sum(mask) / mask.size) * 100, 1)
        percent_false = 100 - percent_true
        result = percent_false == 100

        test_result = 'Тест пройден успешно.' if result else 'Тест не пройден.'
        report_text = f'{self.ident}{test_result}\n{self.ident}Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_miss_traces:\n{report_text}\n\n"

        mask_2d = mask.reshape((self.seismic_cube.shape[0], self.seismic_cube.shape[1]))
        if get_report: visualize_miss_traces(mask_2d, percent_false)

        return {"result": result, "wrong_values": mask_2d, "file_name": self.file_name, "date": timestamp}


    def test_surfaces_location_validation(self, get_report=True) -> dict:
        """
        Метод оценивает соответствие отражающего горизонта сейсмическому кубу

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        all_results_dict = {}
        # Проверка наличия данных для проведения теста
        if not self.surfaces_path_list:
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Данные о поверхностях не были переданы'
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text)
            all_results_dict['data availability'] = False 
        
        # Непосредственно проведение теста
        else:
            cube_z_min, cube_z_max = self.coordinate_z.min(), self.coordinate_z.max()
            all_results_dict['data availability'] = True
            # Проведем тест для каждой поврхности отдельно
            for path in self.surfaces_path_list:
                name = path.split('/')[-1]
                results_dict = {}
                try:
                    min_val, max_val, rectangle_points = self.__open_irap_ascii_grid(path)
                    # проверяем совпадение по X и Y коррдинатам
                    polygon2 = Polygon(rectangle_points)
                    x_y_coords_validation = self.cube_poly.intersects(polygon2)
                    results_dict['x_y_coords_validation'] = str(x_y_coords_validation)
                    # проверяем совпадение по Z коррдинатe
                    z_coords_validation = cube_z_min <= min_val and cube_z_max >= max_val
                    results_dict['z_coords_validation'] = str(z_coords_validation)

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
                    results_dict['x_y_coords_validation'], results_dict['z_coords_validation'] = 'Fail', 'Fail'
                    report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Некорректный путь к файлу:"{path}"'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text)

                all_results_dict[name] = results_dict
        return all_results_dict | {"date" : timestamp}


    def test_faults_location_validation(self, get_report=True) -> dict:
        """
        Метод оценивает соответствие пикировки разлома сейсмическому кубу

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """ 
        all_results_dict = {}
        # Проверка наличия данных для проведения теста
        if not self.faults_file_path:
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Данные о разломах не были переданы'
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text) 
            all_results_dict['data availability'] = False
        
        # Непосредственно проведение теста
        else:
            try:
                fault_dict = self.__parse_faults(self.faults_file_path)
                all_results_dict['data availability'] = True 
                cube_z_min, cube_z_max = self.coordinate_z.min(), self.coordinate_z.max()
                # В одном файле находится несколько разломов, проведем тест для каждого из них
                for key in fault_dict.keys():
                    points = fault_dict[key]
                    res = []
                    results_dict = {}
                    for point in points:
                        # проверяем совпадение по X и Y коррдинатам
                        x_y_coords_validation = Point((point[0], point[1])).within(self.cube_poly)
                        # проверяем совпадение по вертикальной оси
                        z_coords_validation = cube_z_min <= point[2] and cube_z_max >= point[2]
                        res.append(x_y_coords_validation and z_coords_validation)
                    income_points_percent = round((sum(res) * 100 / len(res)), 2)
                    test_result = income_points_percent != 0

                    # results_dict['x_y_coords_validation'] = str(x_y_coords_validation)
                    # results_dict['z_coords_validation'] = str(z_coords_validation)
                    results_dict['income_points_percent'] = income_points_percent
                    all_results_dict[key] = results_dict

                    # формируем отчет о прохождении теста
                    test_result = x_y_coords_validation and z_coords_validation
                    text = 'Тест пройден успешно.' if test_result else 'Тест не пройден.'
                    report_text = f'{self.ident}{text}\n{self.ident}Разлом:"{key}"; {income_points_percent}% точек разлома из {len(points)} входит в границы сейсмического куба'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text)

            except FileNotFoundError:
                all_results_dict['data availability'] = False 
                report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Некорректный путь к файлу:"{self.faults_file_path}"'
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                if get_report: print('\n'+report_text)
        
        return all_results_dict | {"date" : timestamp}


    def test_edge_zone_evaluation(self, get_report=True) -> dict:
        """
        Метод для оценку ширины краевой зоны сейсмического куба

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        # Создаём маску которая будет отражать наличие сейсмотрасс, а значит и геометрию сейсмополя
        seismic_cube = self.seismic_cube.copy()
        seismic_cube_r = seismic_cube.reshape(-1, seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)
        mask = mask.reshape((seismic_cube.shape[0], seismic_cube.shape[1]))
        # Создаём маску которая будет хранить условниые классы (0 - сейсмотрассы отсутствуют, 
        #                                                       1 - сейсмотрассы присутствуют / основная часть куба, 
        #                                                       2 - сейсмотрассы присутствуют / краевая часть куба )
        numeric_mask = (~mask).astype(int)
        # Далее итеративно расширяем границы краевой зоны и обсчитываем для неё дисперсию
        variance_list = []
        while (numeric_mask==1).sum() > 1:    # Условие остановки
            # find_border определяет пиксели/дискреты лежащие на границе полигона, после чего они переклассифицируются в краевую чать куба
            numeric_mask[find_border(numeric_mask==1)] = 2  
            variance = compute_variance(seismic_cube[numeric_mask==2])
            variance_list.append(variance)
        # Находим индекс разделяющий краевую зону и основную часть сейсмического куба
        split_point = best_split_point(variance_list)
        # Далее получаем маску для сейсического куба с краевой зоной
        edge_zone_mask = (~mask).astype(int)
        for i in range(split_point+1):   # задаём ширину краевой зоны в дискретах сейсмического куба`
            edge_zone_mask[find_border(edge_zone_mask==1)] = 2

        if get_report: visualize_edge_zone_evaluation(edge_zone_mask, variance_list, split_point)
        # Логирование результата
        report_text = f'{self.ident}Тест пройден успешно.\n{self.ident}Ширина краевой зоны оценена в {split_point+1} дискретов сейсмического куба'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_edge_zone_evaluation:\n{report_text}\n\n"
        return {'variance_list': variance_list, 'split_point': split_point, 'edge_zone_mask': edge_zone_mask, "file_name": self.file_name, "date": timestamp}
