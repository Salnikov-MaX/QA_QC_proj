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
class KernDataMap:
    data_key: str
    data_column: str
    filters: Optional[List[DataFilter]]
    group: Optional[str]


@dataclass
class KernFileMap:
    data_path: str
    md: str
    well: str
    map: List[KernDataMap]

    def __post_init__(self):
        if not os.path.isfile(self.data_path):
            raise Exception(f"Файла по указанному пути не существует: \'{self.data_path}\'")


@dataclass
class MapSettings:
    show_tests_not_ready_for_launch: bool
    main_dir: Optional[str]
    default_group: Optional[str]
    default_filter: Optional[DataFilter]

    def __post_init__(self):
        if self.show_tests_not_ready_for_launch is None:
            self.show_tests_not_ready_for_launch = False

        if self.main_dir is None:
            pass


@dataclass
class DataMap:
    settings: MapSettings
    kern_files: List[KernFileMap]


def read_map(path: str) -> DataMap:
    """

    :param path: Путь до файла сопоставления
    :return: DataMap
    """

    with open(path, 'r', encoding='utf-8') as file:
        map_data = json.loads(file.read())

    return from_dict(data_class=DataMap, data=map_data)
