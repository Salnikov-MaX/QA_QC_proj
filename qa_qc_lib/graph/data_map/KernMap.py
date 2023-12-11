import os
from dataclasses import dataclass
from typing import Optional, List


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
