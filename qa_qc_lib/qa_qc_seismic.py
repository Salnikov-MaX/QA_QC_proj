import segyio
import numpy as np
from shapely.geometry import MultiPoint, Polygon
import matplotlib.pyplot as plt


class QA_QC_seismic():
    def __init__(self, file_path: str = None, license_area_poly: list = None) -> None:
        self.seismic_cube, self.coordinate_x, self.coordinate_y = self.get_seismic_grid(file_path)
        self.license_area_poly = license_area_poly


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


    def coordinate_validation(self, get_report=True) -> None:
        """Оценка корректности координат загруженного куба.
        Метод проверяет вхождение сейсического куба в границы лицензионного участка 

        Args:
            get_report (bool, optional): _description_. Defaults to True.
        """        
        cube_poly = build_polygon_from_points(self.coordinate_x, self.coordinate_y)

        polygon1 = Polygon(cube_poly)
        polygon2 = Polygon(self.license_area_poly)

        if polygon1.intersects(polygon2):
            intersection_area = polygon1.intersection(polygon2)
            percentage_inside = (intersection_area.area / polygon1.area) * 100
            report_text = f"Тест пройден успешно. \n\nПроцент вхождения сейсмического куба в границы лицензионного участка {round(percentage_inside, 2)}%"
        else:
            intersection_area = None
            percentage_inside = None
            report_text = f"Тест не пройден. \n\nСейсмический куб не входит в границы лицензионного участка"
        
        if get_report:
            visualize_intersection(polygon1, polygon2, intersection_area, report_text)

       


    def second_test(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        return True


    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """        
        return ['first_test', 'second_test']


    def start_tests(self, list_of_tests:list) -> None:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """        
        pass
    

    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """        
        return 'test results'


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