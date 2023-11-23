import json
import os
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class DataInfo:
    data_key: str
    data_path: str
    filters: dict


@dataclass
class MapSettings:
    show_tests_not_ready_for_launch: bool
    main_dir: Optional[str]

    def __post_init__(self):
        if self.show_tests_not_ready_for_launch is None:
            self.show_tests_not_ready_for_launch = False

        if self.main_dir is None:
            pass


class DataMap:
    def __init__(self, map_path: str, data_keys_path: str):
        self.valid_keys = self.read_data_keys(data_keys_path)
        self.settings: MapSettings
        (self.settings, self.data_infos) = self.read_map(map_path)
        self.check_info(self.data_infos, self.valid_keys)

    @staticmethod
    def class_from_args(class_name, arg_dict):
        field_set = {f.name for f in fields(class_name) if f.init}
        wrong_args = [k for k, _ in arg_dict.items() if k not in field_set]

        if len(wrong_args) > 0:
            raise Exception("В Настройках не существует следующих параметров: ", wrong_args)

        filtered_arg_dict = {k: v for k, v in arg_dict.items() if k in field_set}
        return class_name(**filtered_arg_dict)

    @staticmethod
    def read_map(path: str) -> [DataInfo]:
        """

        :param path: Путь до файла сопоставления
        :return: преобразованная коллекция пар (ключ файла, путь файла)
        """

        with open(path, 'r', encoding='utf-8') as file:
            data: dict = json.loads(file.read())

        settings: MapSettings = DataMap.class_from_args(MapSettings, data.get('settings'))

        files_info: [DataInfo] = [DataMap.class_from_args(DataInfo, file_info) for file_info in data['map_files']]

        if settings.main_dir is not None:
            for f in files_info:
                f.data_path = os.path.join(settings.main_dir, f.data_path)

        return settings, files_info

    @staticmethod
    def read_data_keys(path: str) -> [str]:
        """
        Считывает уникальные идентификаторы файлов

        :param path: путь до файла с идентификаторами
        :return: коллекция идентификаторов
        """

        with open(path, 'r', encoding='utf-8') as file:
            data_keys = file.read().splitlines()

        return data_keys

    @staticmethod
    def check_info(data_infos: [DataInfo], valid_data_keys: [str]):
        """
        Проверяет существование указанных ключей и наличие файлов по указанному пути

        """
        for data_info in data_infos:
            if data_info.data_key not in valid_data_keys:
                raise Exception(f"Неверный ключ: \'{data_info.data_key}\'")
            if not os.path.isfile(data_info.data_path):
                raise Exception(f"Файла по указанному пути не существует: \'{data_info.data_path}\', "
                                f"Ключ файла: {data_info.data_key}")
