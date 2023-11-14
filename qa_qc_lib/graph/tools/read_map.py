import json
import os
from dataclasses import dataclass, fields


@dataclass
class FileInfo:
    file_key: str
    file_path: str


@dataclass
class MapSettings:
    show_tests_not_ready_for_launch: bool


default_settings = {
    "show_tests_not_ready_for_launch": True
}


class DataMap:
    def __init__(self, map_path: str, data_keys_path='config\\data_keys.txt'):
        self.valid_keys = self.read_data_keys(data_keys_path)
        self.settings: MapSettings
        self.files_info: [FileInfo]
        (self.settings, self.files_info) = self.read_map(map_path)
        self.check_info(self.files_info, self.valid_keys)

    @staticmethod
    def class_from_args(class_name, arg_dict):
        field_set = {f.name for f in fields(class_name) if f.init}
        wrong_args = [k for k, _ in arg_dict.items() if k not in field_set]

        if len(wrong_args) > 0:
            raise Exception("В Настройках не существует следующих параметров: ", wrong_args)

        filtered_arg_dict = {k: v for k, v in arg_dict.items() if k in field_set}
        return class_name(**filtered_arg_dict)

    @staticmethod
    def read_map(path: str) -> [FileInfo]:
        """

        :param path: Путь до файла сопоставления
        :return: преобразованная коллекция пар (ключ файла, путь файла)
        """

        with open(path, 'r', encoding='utf-8') as file:
            data: dict = json.loads(file.read())

        files_info = [FileInfo(file_info['file_key'], file_info['file_path']) for file_info in data['map_files']]

        data_settings = data['settings'] if data.get('settings') else default_settings

        settings = DataMap.class_from_args(MapSettings, data_settings)

        return settings, files_info

    @staticmethod
    def read_data_keys(path: str) -> [str]:
        """
        Считывает уникальные идентификаторы файлов

        :param path: путь до файла с идентификаторами
        :return: коллекция идентификаторов
        """

        with open(path, 'r', encoding='utf-8') as file:
            keys = file.read().splitlines()

        return keys

    @staticmethod
    def check_info(files_info: [FileInfo], valid_keys: [str]):
        """
        Проверяет существование указанных ключей и наличие файлов по указанному пути

        """
        for file_info in files_info:
            if file_info.file_key not in valid_keys:
                raise Exception(f"Неверный ключ: \'{file_info.file_key}\'")
            if not os.path.isfile(file_info.file_path):
                raise Exception(f"Файла по указанному пути не существует: \'{file_info.file_path}\', "
                                f"Ключ файла: {file_info.file_key}")
