import json
import os
from dataclasses import dataclass


@dataclass
class FileInfo:
    FileName: str
    FilePath: str


class DataMap:
    def __int__(self, map_path: str, data_keys_path='config\\data_keys.txt'):
        self.valid_keys = self.read_data_keys(data_keys_path)
        self.files_info = self.read_map(map_path)

        self.check_info(self.files_info, self.valid_keys)

    @staticmethod
    def read_map(path: str) -> [FileInfo]:
        """

        :param path: Путь до файла сопоставления
        :return: преобразованная коллекция пар (ключ файла, путь файла)
        """

        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())

        files_info = [FileInfo(file_info['file_key'], file_info['file_path']) for file_info in data['data']]
        return files_info

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
            if file_info.FileName not in valid_keys:
                raise f"Неверный ключ: \'{file_info.FileName}\'"
            if not os.path.isfile(file_info.FilePath):
                raise (f"Файла по указанному пути не существует: \'{file_info.FilePath}\',"
                       f" Ключ файла: {FileInfo.FileName}")
