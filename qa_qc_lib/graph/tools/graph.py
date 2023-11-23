import json
from dataclasses import dataclass
from enum import Enum
from typing import List

import dacite
import pandas as pd
from dacite import from_dict


class EnumQAQCClass(str, Enum):
    Kern = "Kern"
    Seismic = "Seismic"
    Gis = "Gis"
    Cubes = "Cubes"


@dataclass
class GraphNode:
    """
            test_key: кодовое имя теста. Пример: (керн/12)
            test_name_code: Имя метода который проводит тест. Пример: (test_monotony)
            test_group: Имя класса к которому относиться тест. Пример: (QA_QC_kern)
            required_data: Обязательные данные для теста.
    """
    test_key: str
    test_name_code: str
    test_group: EnumQAQCClass
    required_data: List[str]


class Graph:
    def __init__(self, graph_json_path):
        self.graph_nodes: List[GraphNode] = self.read_graph_from_json(graph_json_path)

    def get_tests(self, data_key: str) -> List[GraphNode]:
        return [test for test in self.graph_nodes if data_key in test.required_data]

    @staticmethod
    def test_is_ready(graph_node: GraphNode, available_keys: List[str]) -> bool:
        return all([req_data in available_keys for req_data in graph_node.required_data])

    @staticmethod
    def convert_graph_from_csv_to_json(csv_paths: [str],
                                       save_path: str,
                                       data_valid_keys: [str]):
        """
        Сохраняет конфигурационный файл содержащий граф тестирования данных

        Args:
            csv_paths: пути до файлов CSV
            save_path: путь сохранения итогового json файла
            data_valid_keys: коллекция ключевых именований данных

        """

        test_groups_map = {
            "керн/": EnumQAQCClass.Kern,
            "cейсморазведка/": EnumQAQCClass.Seismic,
            "гис/": EnumQAQCClass.Gis,
        }

        dfs = [pd.read_csv(csv_path, delimiter=';') for csv_path in csv_paths]
        df = pd.concat(dfs)
        df = df[df['FLAG'].astype(float) >= 1]
        df['test_code_name'] = df['Источник данных'].astype(str) + df['№'].astype(int).astype(str)

        keys = []
        all_data = list()

        for _, row in df.iterrows():
            inner_data = row['Входные данные'].split(',')
            code_tests = row['Название теста в коде'].split(',')
            test_group = test_groups_map[row['Источник данных'].lower()]

            if test_groups_map.get(row['Источник данных'].lower()) is None:
                raise Exception(f'Указана не существующая группа данных: {row["Источник данных"].lower()}')

            for code_test, data in zip(code_tests, inner_data):
                node = GraphNode(row['test_code_name'], code_test, test_group, [data])
                all_data.append(node)

                keys += node.required_data

        with open(save_path, 'w+', encoding='utf-8') as file:
            json.dump([d.__dict__ for d in all_data], file, ensure_ascii=False)

        ignore_keys = list(set(keys) - set(data_valid_keys))

        invalid_tests = [{"test_name": row['test_code_name'], "data_key": row['Название теста в коде']}
                         for _, row in df.iterrows()
                         if row['Название теста в коде'][:4] != 'test']

        print(f'Количество невалидных тестов {len(invalid_tests)}.\n', invalid_tests)
        print(f'Количество невалидных ключей {len(ignore_keys)} из CSV.\n', ignore_keys)

    @staticmethod
    def read_graph_from_json(graph_path: str) -> List[GraphNode]:
        with open(graph_path, 'r', encoding='utf-8') as file:
            kern_graph_data = json.loads(file.read())

        config = dacite.Config(type_hooks={EnumQAQCClass: EnumQAQCClass})
        return [dacite.from_dict(data_class=GraphNode, data=d, config=config) for d in kern_graph_data]
