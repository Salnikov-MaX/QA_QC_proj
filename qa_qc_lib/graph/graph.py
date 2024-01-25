import json
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import List, Optional

import dacite
import pandas as pd

from qa_qc_lib.graph.edges import edges
from qa_qc_lib.tests.cubes_tests.cubes import QA_QC_cubes
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern
from qa_qc_lib.tests.seismic_tests.seismic import QA_QC_seismic
from qa_qc_lib.tests.wells.wells_tests import QA_QC_wells


class EnumQAQCClass(str, Enum):
    Core = "Core"
    Seismic = "Seismic"
    Gis = "Gis"
    Cubes = "Cube"
    Well = "Well"


@dataclass
class GraphEdge:
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
    config_edge_deserialize = dacite.Config(type_hooks={EnumQAQCClass: EnumQAQCClass})

    def __init__(self, graph_json_path: Optional[str] = None):
        self.graph_nodes: List[GraphEdge] = self.read_graph_from_json(graph_json_path) \
            if graph_json_path \
            else [Graph.read_edge_from_dict(edge) for edge in edges]

    @staticmethod
    def read_graph_from_json(graph_path: str) -> List[GraphEdge]:
        with open(graph_path, 'r', encoding='utf-8') as file:
            kern_graph_data = json.loads(file.read())

        return [dacite.from_dict(data_class=GraphEdge, data=d, config=Graph.config_edge_deserialize)
                for d in kern_graph_data]

    @staticmethod
    def read_edge_from_dict(edge: dict) -> GraphEdge:
        return dacite.from_dict(data_class=GraphEdge, data=edge, config=Graph.config_edge_deserialize)

    def get_tests(self, data_key: str) -> List[GraphEdge]:
        return [test for test in self.graph_nodes if data_key in test.required_data]

    def get_required_data_by_test_code_name(self, test_name_code: str) -> GraphEdge:
        return [test for test in self.graph_nodes if test.test_name_code == test_name_code][0]

    @staticmethod
    def test_is_ready(graph_edge: GraphEdge, available_data_keys: List[str]) -> bool:
        result = set(available_data_keys) >= set(graph_edge.required_data)
        return result


    @staticmethod
    def convert_graph_from_csv_to_json(csv_paths: [str],
                                       save_path: str):
        """
        Сохраняет конфигурационный файл содержащий граф тестирования данных

        Args:
            csv_paths: пути до файлов CSV
            save_path: путь сохранения итогового json файла
            data_valid_keys: коллекция ключевых именований данных

        """

        test_groups_map = {
            "керн/": EnumQAQCClass.Core,
            "сейсморазведка/": EnumQAQCClass.Seismic,
            "гис/": EnumQAQCClass.Gis,
            "геология/": EnumQAQCClass.Cubes,
            "разработка/": EnumQAQCClass.Well
        }

        dfs = [pd.read_csv(csv_path, delimiter=',') for csv_path in csv_paths]
        df = pd.concat(dfs)
        df['test_code_name'] = df['Источник данных'].astype(str) + df['№'].astype(str)
        df['test_code_name'] = df['test_code_name'].apply(lambda v: v.replace('.0', ''))
        df['Название теста в коде'] = df['Название теста в коде'].astype(str)

        keys = []
        all_data = list()
        all_test = list()

        for _, row in df.iterrows():

            inner_data = row['Входные данные'].split(',')
            inner_data = [i_d.split('/') for i_d in inner_data]
            inner_data = [combination for combination in product(*inner_data)]

            code_tests = [code_test.strip() for code_test in row['Название теста в коде'].split(',')]
            test_group = test_groups_map[row['Источник данных'].lower()]

            if test_groups_map.get(row['Источник данных'].lower()) is None:
                raise Exception(f'Указана не существующая группа данных: {row["Источник данных"].lower()}')

            if len(code_tests) == 1 and len(inner_data) > 1:
                code_tests = [code_tests[0] for _ in range(len(inner_data))]

            all_test += [
                (t if 'test' in t else row['test_code_name'], test_groups_map.get(row['Источник данных'].lower())) for t
                in code_tests]

            for code_test, data in zip(code_tests, inner_data):
                data = [d.strip() for d in data]
                node = GraphEdge(row['test_code_name'], code_test, test_group, data)
                all_data.append(node)

                keys += node.required_data

        with open(save_path, 'w+', encoding='utf-8') as file:
            json.dump([d.__dict__ for d in all_data], file, ensure_ascii=False)

        invalid_tests = [row['test_code_name'] + ' : ' + row['Название теста в коде']
                         for _, row in df.iterrows()
                         if row['Название теста в коде'][:4] != 'test']

        print(f'Количество невалидных тестов {len(invalid_tests)}')
        print('\n'.join(invalid_tests))
        print()

        print(f'Не существующие тесты:')

        for test_name, group in set(all_test):
            if group == EnumQAQCClass.Core:
                if not hasattr(QA_QC_kern, test_name):
                    print(group, test_name)
                    continue

            if group == EnumQAQCClass.Cubes:
                if not hasattr(QA_QC_cubes, test_name):
                    print(group, test_name)
                    continue

            if group == EnumQAQCClass.Well:
                if not hasattr(QA_QC_wells, test_name):
                    print(group, test_name)
                    continue

            if group == EnumQAQCClass.Seismic:
                if not hasattr(QA_QC_seismic, test_name):
                    print(group, test_name)
                    continue
