from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from dacite import from_dict

from qa_qc_lib.graph.data_map.CubeMap import CubeData
from qa_qc_lib.graph.data_map.KernMap import KernData


@dataclass
class MapSettings:
    """
    MapSettings - класс настройки процесса тестирования

    Attributes:
        only_ready_for_launch_test - добавлять в конфигурацию тестирования тесты для запуска которых нехватает данных.
            False - в конфигурации будут все возможные тесты для выбранных данных
            True - в конфигурации будут только те тесты для запуска которых достаточно данных.
    """
    only_ready_for_launch_test: bool

    def __post_init__(self):
        if self.only_ready_for_launch_test is None:
            self.only_ready_for_launch_test = False


@dataclass
class DataMap:
    """
    DataMap - является классом отвечающим за идентификацию данных для тестирования.
        В нем данные разделены сгруппированы по группам которые отвечает за свою область:
        керн, кубы, гис, скв.Исследования, сейсмика


    Attributes:
        settings: MapSettings - различные настройки
        kern: Optional[KernData] - секция отвечающая за керновые данные
        cube: Optional[CubeData] - секция отвечающая за данные литологических кубов
    """
    settings: MapSettings

    kern: Optional[KernData]
    cube: Optional[CubeData]

    @staticmethod
    def read_map(path: str) -> DataMap:
        """
        Чтение карты

        :param path: Путь до файла сопоставления
        :return: DataMap
        """

        with open(path, 'r', encoding='utf-8') as file:
            map_data = json.loads(file.read())

        return from_dict(data_class=DataMap, data=map_data)
