import json
import os
from dataclasses import dataclass
from typing import Optional, List

from dacite import from_dict


@dataclass
class DataFilter:
    column_name: str
    value: str
    operation: str


@dataclass
class KernPathInfo:
    data_key: str
    data_column: str
    sheet_name: Optional[str]


@dataclass
class KernFileMap:
    data_path: str
    map: List[KernPathInfo]

    def __post_init__(self):
        if not os.path.isfile(self.data_path):
            raise Exception(f"Файла по указанному пути не существует: \'{self.data_path}\'")
        if not any([item for item in self.map if item.data_key == 'MD']):
            raise Exception(f"Вы не указали путь до MD!")


@dataclass
class KernData:
    group_by_columns: Optional[List[str]]
    filters: Optional[List[DataFilter]]
    files: List[KernFileMap]


@dataclass
class MapSettings:
    only_ready_for_launch_test: bool

    def __post_init__(self):
        if self.only_ready_for_launch_test is None:
            self.only_ready_for_launch_test = False


@dataclass
class DataMap:
    settings: MapSettings
    kern: KernData


def read_map(path: str) -> DataMap:
    """

    :param path: Путь до файла сопоставления
    :return: DataMap
    """

    with open(path, 'r', encoding='utf-8') as file:
        map_data = json.loads(file.read())

    return from_dict(data_class=DataMap, data=map_data)
