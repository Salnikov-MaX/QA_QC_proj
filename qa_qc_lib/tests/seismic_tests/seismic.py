import segyio
from shapely.geometry import Polygon, Point
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tools.seismic_tools import *
from qa_qc_lib.tools.math_tools import compute_variance


class QA_QC_seismic(QA_QC_main):
    def __init__(self, TWT_cube_path: str = None, TVD_cube_path: str = None, seismic_attr_path: str = None) -> None:
        """
        Args:
            file_path (str): путь к SEG-Y файлу содержащему данные сейсмического куба
            
        """
        super().__init__()

        if TWT_cube_path:
            self.TWT_seismic_cube, self.TWT_coordinate_x, self.TWT_coordinate_y, self.TWT_coordinate_z = self.__get_seismic_grid(TWT_cube_path)
            self.TWT_file_name = TWT_cube_path.split('/')[-1]
        else:
            self.TWT_file_name = 'No seismic cube data'

        if TVD_cube_path:
            self.TVD_seismic_cube, self.TVD_coordinate_x, self.TVD_coordinate_y, self.TVD_coordinate_z = self.__get_seismic_grid(TVD_cube_path)
            self.TVD_file_name = TVD_cube_path.split('/')[-1]
        else:
            self.TVD_file_name = 'No seismic cube data'

        self.seismic_attr_path = seismic_attr_path


    def __get_seismic_grid(self, segy_file_path: str):
        """
        Метод предназначенный для чтения сейсмического куба и метаданных из файла формата SEG-Y 

        Args:
            segy_file_path (str): Путь к файлу с сейсмическими данными

        Returns:
            _type_: (куб сейсмических трасс, вектор координат X каждой из трасс, вектор координат Y каждой из трасс, вектор глубин)
        """
        segy = segyio.open(segy_file_path, 'r', strict=False)  # Открываем SEGY-файл в режиме чтения
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
                    data.append({'X': x_rotated, 'Y': y_rotated, 'Value': value})

            df = pd.DataFrame(data)   
            df.replace(9999900.0, np.nan, inplace=True)
            
            return df.dropna(), rectangle_points


    def test_miss_traces_TWT(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический куб TWT на наличие пропущенных / не записанных сейсмотрасс
        
        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь, specification cловарь где , wrong_values - маска с результатом ,test_name - название теста ,
                  percent_false - процент отсутствующих в кубе сейсмотрасс, error_decr -краткое описание ошибки
        """
        if self.TWT_file_name == 'No seismic cube data':
            text = f'Данные сейсмического куба TWT не были переданы'
            report_text = self.generate_report_text(text, test_result)
            self.update_report(report_text)
            return { "data_availability": False}

        seismic_cube_r = self.TWT_seismic_cube.reshape(-1, self.TWT_seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)

        percent_true = round((np.sum(mask) / mask.size) * 100, 1)
        percent_false = 100 - percent_true
        result = percent_false == 100

        test_result = 1 if result else 0
        text = f'Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        report_text = self.generate_report_text(text, test_result)

        self.update_report(report_text)

        mask_2d = mask.reshape((self.TWT_seismic_cube.shape[0], self.TWT_seismic_cube.shape[1]))

        specification = {"wrong_values" : mask_2d,
                         'test_name'    : 'test_miss_traces',
                         "percent_false": percent_false,
                         "error_decr"   : text}

        return {"data_availability": True,
                "result" : result,
                "specification": specification}


    def test_miss_traces_TVD(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический куб TVD на наличие пропущенных / не записанных сейсмотрасс
        
        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь, specification cловарь где , wrong_values - маска с результатом ,test_name - название теста ,
                  percent_false - процент отсутствующих в кубе сейсмотрасс, error_decr -краткое описание ошибки
        """
        if self.TVD_file_name == 'No seismic cube data':
            text = f'Данные сейсмического куба TWT не были переданы'
            report_text = self.generate_report_text(text, test_result)
            self.update_report(report_text)
            return { "data_availability": False}

        seismic_cube_r = self.TVD_seismic_cube.reshape(-1, self.TVD_seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)

        percent_true = round((np.sum(mask) / mask.size) * 100, 1)
        percent_false = 100 - percent_true
        result = percent_false == 100

        test_result = 1 if result else 0
        text = f'Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        report_text = self.generate_report_text(text, test_result)

        self.update_report(report_text)

        mask_2d = mask.reshape((self.TVD_seismic_cube.shape[0], self.TVD_seismic_cube.shape[1]))

        specification = {"wrong_values" : mask_2d,
                         'test_name'    : 'test_miss_traces',
                         "percent_false": percent_false,
                         "error_decr"   : text}

        return {"data_availability": True,
                "result" : result,
                "specification": specification}
    

    def test_seismic_attribute_validation(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический атрибут на корректность
        
        Required data:
            self.seismic_attr_path (np.ndarray): сейсмический атрибут

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь, specification cловарь где , error_decr - краткое описание ошибки
        """
        if not self.seismic_attr_path:
            text = f'Данные сейсмического атрибута не были переданы'
            report_text = self.generate_report_text(text, test_result)
            self.update_report(report_text)
            return { "data_availability": False}

        df, _ = self.__open_irap_ascii_grid(self.seismic_attr_path)
        result = df.Value.sum() != 0
        if result:
            text = f'Сейсмический атрибут снят корректно'
        else:
            text = f'Сейсмический атрибут не корректно'
        report_text = self.generate_report_text(text, 1 if result else 0)
        self.update_report(report_text)

        specification = {"error_decr": text}

        return {"data_availability": True,
                "result" : result,
                "specification": specification}
    

 



   