import os
import segyio
import numpy as np
from shapely.geometry import MultiPoint, Polygon
import matplotlib.pyplot as plt
from accessify import private
import datetime
from typing import Any


class QA_QC_seismic():
    def __init__(self, file_path: str = None, license_area_poly: list = None) -> None:
        self.seismic_cube, self.coordinate_x, self.coordinate_y = self.get_seismic_grid(file_path)
        self.license_area_poly = license_area_poly
        self.report_text = ""
        self.ident = ' '*5   # отступ при формировании лога


    @private
    def get_seismic_grid(self, segy_file_path:str):
        """Метод предназначенный для чтения сейсмического куба из файла формата SEG-Y 

        Args:
            segy_file_path (str): Путь к файлу с сейсмическими данными

        Returns:
            _type_: (куб сейсмических трасс, вектор координат X каждой из трасс, вектор координат Y каждой из трасс)
        """        
        segy = segyio.open(segy_file_path, 'r', strict=False)     # Открываем SEGY-файл в режиме чтения
        coordinate_x = segy.attributes(segyio.TraceField.SourceX)
        coordinate_y = segy.attributes(segyio.TraceField.SourceY)
        seismic_data = np.array([trace.copy() for trace in segy.trace])
        return seismic_data, np.array(coordinate_x), np.array(coordinate_y)


    def test_coordinate_validation(self, get_report=True) -> None:
        """Оценка корректности координат загруженного куба.
        Метод проверяет вхождение сейсического куба в границы лицензионного участка 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.
        """
        # Проверяем наличие данных для запуска теста
        if self.coordinate_x is None or self.coordinate_y is None or self.license_area_poly is None:
            coordinate_existence = self.coordinate_x is not None or self.coordinate_y is not None
            license_area_poly_existence = self.license_area_poly is not None
            report_text = f'{self.ident}Отсутствуют данные для проведения теста.\n{self.ident}Наличие координат:{coordinate_existence}, Наличие границ лицензионного участка:{license_area_poly_existence}'
        
        # Непосредственно проведение теста
        else:
            cube_poly = build_polygon_from_points(self.coordinate_x, self.coordinate_y)
            polygon1, polygon2 = Polygon(cube_poly), Polygon(self.license_area_poly)
        
            if polygon1.intersects(polygon2):
                intersection_area = polygon1.intersection(polygon2)
                percentage_inside = (intersection_area.area / polygon1.area) * 100
                report_text = f"{self.ident}Тест пройден успешно. \n{self.ident}Процент вхождения сейсмического куба в границы лицензионного участка {round(percentage_inside, 2)}%"
            else:
                intersection_area = None
                percentage_inside = None
                report_text = f"{self.ident}Тест не пройден. \n{self.ident}Сейсмический куб не входит в границы лицензионного участка"
            
            if get_report:
                visualize_intersection(polygon1, polygon2, intersection_area, report_text)

        #self.report_file.write(f"test_coordinate_validation: {report_text}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_coordinate_validation:\n{report_text}\n\n"


    def get_list_of_tests(self) -> list:
        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    
    def get_method_description(self, method_name: str) -> str:
        """Метод для получение описания теста по названию теста

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


    def start_tests(self, list_of_tests: list) -> list[Any]:
        # results = []
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
            #results.append(method())
        # return results


    def generate_test_report(self, file_name='test_report', file_path='report', data_name='Неизвестный файл'):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        report = f"Отчет о тестировании от {timestamp}{self.ident}Название тестируемого файла: '{data_name}'\n\n{self.report_text} "
        with open(f"{file_path}/{file_name}.txt", "w") as file:
            file.write(report)
    





#########################################################################################################
###########################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ |########################################
#########################################################################################################

def visualize_intersection(polygon1, polygon2, intersection_area=None, text=None):
    fig, ax = plt.subplots()

    x_coords1, y_coords1 = polygon1.exterior.xy
    ax.plot(x_coords1, y_coords1, label='Проекция сейсмического куба', color='red')

    x_coords2, y_coords2 = polygon2.exterior.xy
    ax.plot(x_coords2, y_coords2, label='Граница лицензионного участка', color='orange')

    if intersection_area:
        intersection_x, intersection_y = intersection_area.exterior.xy
        ax.fill(intersection_x, intersection_y, color='red', alpha=0.5, label='Intersection Area')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Проверка корректности координат')
    ax.legend()
    ax.grid()

    if text:
        ax.text(1.1, 0.5, text, transform=ax.transAxes, fontsize=12, verticalalignment='center', style='italic')
        
    plt.show()


def build_polygon_from_points(x_coords:np.array, y_coords:np.array):
    """Получение полигона из облака точек

    Args:
        x_coords (np.array): X координаты облака точек
        y_coords (np.array): Y координаты облака точек

    Returns:
        list: список кортежей координат полигона в который вписано облако точек 
    """    
    points = MultiPoint(list(zip(x_coords, y_coords)))
    convex_hull = points.convex_hull
    return list(convex_hull.exterior.coords)