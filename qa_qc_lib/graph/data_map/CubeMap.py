import os
from dataclasses import dataclass
from typing import List


@dataclass
class CubePathInfo:
    data_key: str
    data_path: str

    def __post_init__(self):
        if not os.path.isfile(self.data_path):
            raise Exception(f'Файл типа {self.data_key} {self.data_path} не найден!')


@dataclass
class CubeData:
    """
    CubeData - отвечает за идентификацию лит кубов системе.

    Attributes:
        property_files: List[CubePathInfo] - список свойств куба и их местоположения.
        grid_name: str - имя куба.
        grid_dir: str - директория где хранится куб.
            В директории должны быть следующий файлы:
                {grid_name}.GRDECL
                {grid_name}_ACTNUM.GRDECL
                {grid_name}_COORD.GRDECL
                {grid_name}_ZCORN.GRDECL

    """
    property_files: List[CubePathInfo]
    grid_name: str
    grid_dir: str

    def __post_init__(self):
        if not os.path.isdir(self.grid_dir):
            raise Exception(f'Директории {self.grid_dir} не существует!')

        files = [f"{self.grid_name}.GRDECL",
                 f"{self.grid_name}_ACTNUM.GRDECL",
                 f"{self.grid_name}_COORD.GRDECL",
                 f"{self.grid_name}_ZCORN.GRDECL"]

        print(os.listdir(self.grid_dir))
        exist_files = [f for f in os.listdir(self.grid_dir) if f in files]

        if len(exist_files) < len(files):
            raise Exception(f'В директории {self.grid_dir} нет файлов: !')
