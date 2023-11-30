import segyio
from shapely.geometry import Polygon, Point
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tools.seismic_tools import *
from qa_qc_lib.tools.math_tools import compute_variance


class QA_QC_seismic(QA_QC_main):
    def __init__(self, file_path: str = None) -> None:
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
        if file_path:
            self.seismic_cube, self.coordinate_x, self.coordinate_y, self.coordinate_z = self.__get_seismic_grid(file_path)
            self.file_name = file_path.split('/')[-1]
        else:
            self.file_name = 'No seismic cube data'


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

    def test_miss_traces(self, get_report=True) -> dict:
        """
        Метод проверяет сейсмический куб на наличие пропущенных / не записанных сейсмотрасс
        
        Required data:
            self.seismic_cube (np.ndarray): сейсмический куб

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь, specification cловарь где , wrong_values - маска с результатом ,test_name - название теста ,
                  percent_false - процент отсутствующих в кубе сейсмотрасс, error_decr -краткое описание ошибки
        """
        if self.file_name == 'No seismic cube data':
            text = f'Данные сейсмического куба не были переданы'
            report_text = self.generate_report_text(text, test_result)
            self.update_report(report_text)
            return { "data_availability": False}

        seismic_cube_r = self.seismic_cube.reshape(-1, self.seismic_cube.shape[2])
        mask = np.all(seismic_cube_r == 0, axis=1)

        percent_true = round((np.sum(mask) / mask.size) * 100, 1)
        percent_false = 100 - percent_true
        result = percent_false == 100

        test_result = 1 if result else 0
        text = f'Сейсмические трассы присутствуют в {percent_false}% случаев, отсутствуют в {percent_true}%'
        report_text = self.generate_report_text(text, test_result)

        self.update_report(report_text)

        mask_2d = mask.reshape((self.seismic_cube.shape[0], self.seismic_cube.shape[1]))

        specification = {"wrong_values" : mask_2d,
                         'test_name'    : 'test_miss_traces',
                         "percent_false": percent_false,
                         "error_decr"   : text}

        return {"data_availability": True,
                "result" : result,
                "specification": specification}



   