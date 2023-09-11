import segyio
import numpy as np
from shapely.geometry import Polygon, Point
import datetime
from .qa_qc_main import QA_QC_main
from .qa_qc_tools.seismic_tools import *
from .qa_qc_tools.math_tools import compute_variance


class QA_QC_seismic(QA_QC_main):
    def __init__(self, file_path: str, license_area_poly: list = None, surfaces_path_list: list = None, faults_file_path: str = None) -> None:
        """
        Args:
            file_path (str): путь к SEG-Y файлу содержащему данные сейсмического куба
            license_area_poly (list, optional): список кортежей содержащий координаты X и Y 
                                                полигона лицензионного участка. Defaults to None.
            surfaces_path_list (list, optional): список путей к файлам с поверхностями 
                                                 структурных карт / карт изохрон (в зависимости от 
                                                 типа куба переданного на вход). Данный список должен 
                                                 быть упорядочен по глубине залегания (последняя 
                                                 поверхность в списке залегает глубже всех остальных). Defaults to None.
            faults_file_path (str, optional): путь к файлу содержащему данные о разломах. Defaults to None.
        """        
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
            _type_: датафрейм хранящий карту, полигон в котором лежит карта
        """        
        with open(path, 'r') as f:
            # Читаем заголовок
            unknown, nrows, dy, dx = map(float, f.readline().split())
            x_min, x_max, y_min, y_max = map(float, f.readline().split())
            n_cols, rotation, rot_x, rot_y = map(float, f.readline().split())
            f.readline()  # Строка из нулей
            
            data_vector = []
            for line in f:
                data_vector.extend(map(float, line.split()))
            data_matrix = [data_vector[i:i+int(n_cols)] for i in range(0, len(data_vector), int(n_cols))]

            rectangle_points = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]

            data = []
            for row in range(int(nrows)):
                for col in range(int(n_cols)):
                    x = x_min + col * dx
                    y = y_max - row * dy  # Если Y уменьшается вниз страницы
                    value = data_matrix[row][col]

                    # Применяем преобразование координат с учетом поворота
                    theta = np.radians(rotation)  # Переводим градусы в радианы
                    x_rotated = rot_x + (x - rot_x) * np.cos(theta) - (y - rot_y) * np.sin(theta)
                    y_rotated = rot_y + (x - rot_x) * np.sin(theta) + (y - rot_y) * np.cos(theta)
                    data.append({'X': x_rotated, 'Y': y_rotated, 'Dept': value})

            df = pd.DataFrame(data)   
            df.replace(9999900.0, np.nan, inplace=True)

            return df.dropna(), rectangle_points


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

        Required data:
            self.coordinate_x (np.ndarray): координаты x сейсмического куба
            self.coordinate_y (np.ndarray): координаты y сейсмического куба
            self.license_area_poly (list): список кортежей содержащий координаты полигона лицензионного полигона

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
            text = f'Наличие координат:{coordinate_existence}, Наличие границ лицензионного участка:{license_area_poly_existence}'
            report_text = self.generate_report_text(text, 2)
            if get_report: print('\n'+report_text+self.delimeter) 
            all_results_dict["result"] = 'Fail'
        # Непосредственно проведение теста
        else:
            polygon2 = Polygon(self.license_area_poly)
            if self.cube_poly.intersects(polygon2):
                intersection_area = self.cube_poly.intersection(polygon2)
                percentage_inside = (intersection_area.area / self.cube_poly.area) * 100
                text = f'Процент вхождения сейсмического куба в границы лицензионного участка {round(percentage_inside, 2)}%'
                report_text = self.generate_report_text(text, 1)
                all_results_dict["result"] = 'True'

            else:
                intersection_area, percentage_inside = None, None
                text = f'Сейсмический куб не входит в границы лицензионного участка'
                report_text = self.generate_report_text(text, 0)
                all_results_dict["result"] = 'False'
            
            if get_report: 
                visualize_intersection(self.cube_poly, polygon2, intersection_area, report_text)
                print(self.delimeter) 

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_coordinate_validation:\n{report_text}\n\n"
        return all_results_dict | {"file_name": self.file_name, "date": timestamp}


    def test_monotony(self, get_report=True) -> dict:
        """
        Метод проверяет ось глубин / времени на монотонное возрастание (каждое следующее значение больше предыдущего)

        Required data:
            self.coordinate_z (np.ndarray): координаты z сейсмического куба (глубины/время)

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """
        result_mask = np.diff(self.coordinate_z) <= 0
        result_mask = np.insert(result_mask, 0, False)
        result = sum(result_mask) == 0
        if result:
            text = 'Отметки оси глубин/времени монотонно возрастают'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'Отметки оси глубин/времени не возрастают монотонно'
            report_text = self.generate_report_text(text, 0)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_monotony:\n{report_text}\n\n"
        if get_report: print('\n'+report_text+self.delimeter)

        return {"result": result, "wrong_values": ~result_mask, "file_name": self.file_name, "date": timestamp}


    def test_miss_traces(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический куб на наличие пропущенных / не записанных сейсмотрасс
        
        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб

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

        test_result = 1 if result else 0
        text = f'Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        report_text = self.generate_report_text(text, test_result)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_miss_traces:\n{report_text}\n\n"

        mask_2d = mask.reshape((self.seismic_cube.shape[0], self.seismic_cube.shape[1]))
        if get_report: 
            visualize_miss_traces(mask_2d, percent_false)
            print(self.delimeter)

        return {"result": result, "wrong_values": mask_2d, "file_name": self.file_name, "date": timestamp}


    def test_surfaces_location_validation(self, get_report=True) -> dict:
        """
        Метод оценивает соответствие отражающего горизонта сейсмическому кубу
        Внимание, метод test_surfaces_values_validation имеет аналогичный и более 
        точный функционал, рекомендуется использовать его.

        Required data:
            self.surfaces_path_list (list): список содержащий пути к файлам с отражающими горизонтами
            self.coordinate_x (np.ndarray): координаты x сейсмического куба
            self.coordinate_y (np.ndarray): координаты y сейсмического куба
            self.coordinate_z (np.ndarray): координаты z сейсмического куба (глубины/время)

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
            if get_report: print('\n'+report_text+self.delimeter)
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
                    map_df, rectangle_points = self.__open_irap_ascii_grid(path)
                    min_val, max_val = map_df.Dept.min(), map_df.Dept.max()
                    # проверяем совпадение по X и Y коррдинатам
                    polygon2 = Polygon(rectangle_points)
                    x_y_coords_validation = self.cube_poly.intersects(polygon2)
                    results_dict['x_y_coords_validation'] = str(x_y_coords_validation)
                    # проверяем совпадение по Z коррдинатe
                    z_coords_validation = cube_z_min <= min_val and cube_z_max >= max_val
                    results_dict['z_coords_validation'] = str(z_coords_validation)

                    # формируем отчет о прохождении теста
                    test_result = x_y_coords_validation and z_coords_validation
                    res_text = '' if test_result else 'не '
                    text = f'Путь к файлу:"{path}"; отражающий горизонт {res_text}попадает в границы сейсмического куба'
                    report_text = self.generate_report_text(text, 1 if test_result else 0)
                    if not test_result: report_text = report_text + f' (совпадение по X,Y:{x_y_coords_validation}, по вертикальной шкале:{z_coords_validation})'

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text+self.delimeter)

                except FileNotFoundError: 
                    results_dict['x_y_coords_validation'], results_dict['z_coords_validation'] = 'Fail', 'Fail'
                    report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Некорректный путь к файлу:"{path}"'
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text+self.delimeter)

                all_results_dict[name] = results_dict
        return all_results_dict | {"date" : timestamp}


    def test_faults_location_validation(self, get_report=True) -> dict:
        """
        Метод оценивает соответствие пикировки разлома сейсмическому кубу

        Required data:
            self.faults_file_path (str): путь к файлу с координатами разломов
            self.coordinate_x (np.ndarray): координаты x сейсмического куба
            self.coordinate_y (np.ndarray): координаты y сейсмического куба
            self.coordinate_z (np.ndarray): координаты z сейсмического куба (глубины/время)

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """ 
        all_results_dict = {}
        # Проверка наличия данных для проведения теста
        if not self.faults_file_path:
            text = f'Данные о разломах не были переданы'
            report_text = self.generate_report_text(text, 2)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text+self.delimeter) 
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

                    results_dict['income_points_percent'] = income_points_percent
                    all_results_dict[key] = results_dict

                    # формируем отчет о прохождении теста
                    test_result = x_y_coords_validation and z_coords_validation
                    
                    text = f'Разлом:"{key}"; {income_points_percent}% точек разлома из {len(points)} входит в границы сейсмического куба'
                    report_text = self.generate_report_text(text, 1 if test_result else 0)


                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text+self.delimeter)

            except FileNotFoundError:
                all_results_dict['data availability'] = False 
                text = f'Некорректный путь к файлу:"{self.faults_file_path}"'
                report_text = self.generate_report_text(text, 2)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_faults_location_validation:\n{report_text}\n\n"
                if get_report: print('\n'+report_text+self.delimeter)
        
        return all_results_dict | {"date" : timestamp}


    def test_edge_zone_evaluation(self, get_report=True) -> dict:
        """
        Метод для оценку ширины краевой зоны сейсмического куба

        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб

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

        if get_report: 
            visualize_edge_zone_evaluation(edge_zone_mask, variance_list, split_point)
            print(self.delimeter)
        # Логирование результата
        text = f'Ширина краевой зоны оценена в {split_point+1} дискретов сейсмического куба'
        report_text = self.generate_report_text(text, 1)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_edge_zone_evaluation:\n{report_text}\n\n"
        return {'variance_list': variance_list, 'split_point': split_point, 'edge_zone_mask': edge_zone_mask, "file_name": self.file_name, "date": timestamp}


    def test_surfaces_values_validation(self, get_report=True) -> dict: 
        """
        Метод делает срез значений сейсмического куба по отражающему горизонту и оценивает процент
        положительных, отрицательных и равных 0 значений

        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб
            self.surfaces_path_list (list): список содержащий пути к файлам с отражающими горизонтами
            self.coordinate_x (np.ndarray): координаты x сейсмического куба
            self.coordinate_y (np.ndarray): координаты y сейсмического куба
            self.coordinate_z (np.ndarray): координаты z сейсмического куба (глубины/время)

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        all_results_dict = {}
        # Проверка наличия данных для проведения теста
        if not self.surfaces_path_list:
            text = f'Данные о поверхностях не были переданы'
            report_text = self.generate_report_text(text, 2)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_surfaces_values_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text+self.delimeter)
            all_results_dict['data availability'] = False 
        
        # Непосредственно проведение теста
        else:
            all_results_dict['data availability'] = True
            # Проведем тест для каждой поврхности отдельно
            for path in self.surfaces_path_list:
                name = path.split('/')[-1]
                results_dict = {}
                try:
                    map_df, _ = self.__open_irap_ascii_grid(path)
                    # Проверка, если есть строки вне полигона и их исключение, если есть
                    inside_polygon = map_df.apply(lambda row: Point(row['X'], row['Y']).within(self.cube_poly), axis=1)
                    inside_depth_range = (map_df['Dept'] >= self.coordinate_z.min()) & (map_df['Dept'] <= self.coordinate_z.max())
                    if not inside_polygon.all():
                        print("ВНИМАНИЕ! Часть точек поверхности выходит за пределы сейсмического куба по X, Y!")
                    if not inside_depth_range.all():
                         print("ВНИМАНИЕ! Найдены точки вне заданного диапазона глубины!")
                    map_df = map_df[inside_polygon & inside_depth_range]
                    
                    if map_df.empty:
                        empty_df = True
                        test_result = False
                        result_text = f'отражающий горизонт {name} не попадает в границы сейсмического куба'
                    else:
                        empty_df = False
                        # получаем срез сейсмических значений
                        seismic_cube_r = self.seismic_cube.reshape(-1, self.seismic_cube.shape[2])
                        indxs_0 = find_closest_indices_x_y(self.coordinate_x, self.coordinate_y, map_df.X.to_numpy(), map_df.Y.to_numpy())
                        indxs_1 = find_closest_indices_z(self.coordinate_z, map_df.Dept.to_numpy())
                        map_df['Value'] = seismic_cube_r[indxs_0, indxs_1]

                        # получаем статистику этих значений
                        total_values = map_df['Value'].size
                        zero_percent = (map_df['Value'] == 0).sum() / total_values * 100
                        less_than_zero_percent = (map_df['Value'] < 0).sum() / total_values * 100
                        greater_than_zero_percent = (map_df['Value'] > 0).sum() / total_values * 100

                        # формируем отчет о прохождении теста
                        text_2 = 'положительной' if greater_than_zero_percent > less_than_zero_percent else 'отрицательной'
                        test_result = zero_percent > 90 or less_than_zero_percent > 90 or greater_than_zero_percent > 90
                        result_text = f'отражающий горизонт снят по {text_2} амплитуде (>0:{greater_than_zero_percent:.2f}%, <0:{less_than_zero_percent:.2f})'
                        if get_report: 
                            visualize_seismic_slice(map_df, name)
                            print(self.delimeter)

             
                    text = f'Путь к файлу:"{path}";' + result_text
                    report_text = self.generate_report_text(text, 1 if test_result else 0)

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_values_validation:\n{report_text}\n\n"
                    if get_report and empty_df: print('\n'+report_text+self.delimeter)
                    
                    results_dict['result'] = test_result
                    results_dict['slise_map'] = map_df
    
                except FileNotFoundError: 
                    results_dict['x_y_coords_validation'], results_dict['z_coords_validation'] = 'Fail', 'Fail'
                    text = f'Некорректный путь к файлу:"{path}"'
                    report_text = self.generate_report_text(text, 2)

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.report_text += f"{timestamp:10} / test_surfaces_values_validation:\n{report_text}\n\n"
                    if get_report: print('\n'+report_text+self.delimeter)

                all_results_dict[name] = results_dict
        return all_results_dict | {"date" : timestamp}


    def test_surfaces_dept_validation(self, get_report=True) -> dict: 
        """
        Метод оценивает положение структурных поверхностей относительно друг друга.
        Внимание, пути к файлам содержащим поверности в списке self.surfaces_path_list 
        должны быть в порядке их залегания (сверху вниз).

        Required data:
            self.surfaces_path_list (list): список содержащий пути к файлам с отражающими горизонтами
            
        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        all_results_dict = {}
        # Проверка наличия данных для проведения теста
        if not self.surfaces_path_list:
            text = f'Данные о поверхностях не были переданы'
            report_text = self.generate_report_text(text, 2)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_surfaces_dept_validation:\n{report_text}\n\n"
            if get_report: print('\n'+report_text+self.delimeter)
            all_results_dict['data availability'] = False 
        
        # Непосредственно проведение теста
        else:
            all_results_dict['data availability'] = True
            dataframes, names, err_files = [], [], []
            for path in self.surfaces_path_list:
                try:
                    map_df, _ = self.__open_irap_ascii_grid(path)
                    dataframes.append(map_df)
                    names.append(path.split('/')[-1])
                except FileNotFoundError:
                    print(f'ВНИМАНИЕ! Файл {path} отсутствует по указанному пути!'+self.delimeter)
                    err_files.append(path)

            # непосредственно проведение теста
            results_dict = {}
            for i in range(1, len(dataframes)):
                merged = dataframes[i].merge(dataframes[i - 1], on=['X', 'Y'], how='inner', suffixes=('_curr', '_prev'))
                # Интерполяция для учета того, что координаты могут не совпадать точно
                merged.interpolate(inplace=True)
                non_conformity = merged[merged['Dept_curr'] > merged['Dept_prev']]
                if not non_conformity.empty:
                    percent_mismatch = (non_conformity.shape[0] / merged.shape[0]) * 100
                    text = f'Нижележащая структурная карта "{names[i]}" оказалась выше вышележащей структурной карты "{names[i-1]}" (несоответствие на {percent_mismatch:.2f}% площади)'
                    report_text = self.generate_report_text(text, 0)
                    results_dict['result'] = False
                    results_dict['result report'] = text
                else:
                    text = f'Нижележащая структурная карта "{names[i]}" оказалась ниже вышележащей структурной карты "{names[i-1]}"'
                    report_text = self.generate_report_text(text, 1)
                    results_dict['result'] = True
                    results_dict['result report'] = text
                
                all_results_dict[names[i-1]] = results_dict
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_surfaces_dept_validation:\n{report_text}\n\n"
                if get_report: print('\n'+report_text+self.delimeter)

            if err_files: print(f'Не удалось найти следующие файлы: {err_files}') 
            all_results_dict['files not found'] = err_files

        return all_results_dict | {"date" : timestamp}