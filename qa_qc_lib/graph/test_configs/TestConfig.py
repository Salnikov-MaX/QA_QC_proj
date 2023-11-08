import json
import os
from dataclasses import dataclass


@dataclass
class TestConfig:

    def __init__(self, file_path: str):
        """
        :param file_path: Путь до файла
        """
        if not os.path.isfile(file_path):
            raise f"Конфигурационный файл: {file_path} не найден"

        config_dict: dict = self.read(file_path)

        for file_data in config_dict:
            pass

    @staticmethod
    def read(file_path: str) -> dict:
        """

        :param file_path: Путь до файл
        :return:
        """
        with open(file_path, 'r') as file:
            config_data: dict = json.load(file)

        return config_data
