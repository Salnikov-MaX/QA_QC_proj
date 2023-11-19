from __future__ import annotations

import enum
import json
from typing import Any, List

from qa_qc_lib.graph.tools.read_map import DataInfo


class EnumQAQCClass(enum.Enum):
    Kern = 1
    Seismic = 2
    Gis = 3
    Cubes = 4


class GraphTest:
    def __init__(self, code_name: str,
                 class_name: EnumQAQCClass,
                 test_name_in_code: str,
                 required_data: [[str]]):
        """
        test_code: кодовое имя теста. Пример: (керн/12)
        test_class: Имя класса к которому относиться тест. Пример: (QA_QC_kern)
        test_method: Имя метода который проводит тест. Пример: (test_monotony)
        required_data_for_test: Обязательные данные для теста.

        Данные представляют собой коллекцию из набора альтернативных ключевых имён одного типа данных.
        т.е. для запуска теста обязательно наличие хотя бы 1 типа данных из каждого массива

        Допустим:
            self.required_data_for_test = [
                ["1", "2"],
                ["3", "4"],
            ]

        Мы можем запустить данный тест если у нас есть следующие наборы данных:
            ["1", "3", ... ]
            ["1", "4", ... ]
            ["2", "3", ... ]
            ["2", "4", ... ]


        Пример данных: (
            self.required_data_for_test = [
                ["Кп_откр|txt/xlsx|Керн|", "Кп_абс|txt/xlsx|Керн|"],
                ["Плотность_абсолютно_сухого_образца|txt/xlsx|Керн|", "Плотность_максимально_увлажненного_образца|txt/xlsx|Керн|"],
            ]
        )
        """
        self.test_code_name = code_name
        self.test_class_name = class_name
        self.test_method_name = test_name_in_code
        self.required_data_for_test = required_data

    def contains_required_data(self, target_data_key: str) -> bool:
        return any([target_data_key in data_keys for data_keys in self.required_data_for_test])

    def check_files_for_launch_test(self, data_keys: List[str]) -> bool:
        data_keys = set(data_keys)
        return all([set(r) & data_keys for r in self.required_data_for_test])

    def get_test_config(self, main_files_info: DataInfo, data_info: [DataInfo]) \
            -> dict[str, bool | str | list[list[Any]]]:
        data_keys = [f.data_key for f in data_info]
        data_for_test = [list(set(r) & set(data_keys)) for r in self.required_data_for_test]
        ready_for_launch = self.check_files_for_launch_test(data_keys)
        all_data_for_launch = [[fi.__dict__ for fi in data_info if fi.data_key in d] for d in data_for_test]

        if ready_for_launch:
            priority_data_for_launch = []
            for dat_info_item in all_data_for_launch:
                if main_files_info.data_path in [d['data_path'] for d in dat_info_item]:
                    priority_data_for_launch.append(main_files_info.__dict__)
                else:
                    priority_data_for_launch.append(dat_info_item[0])

        else:
            priority_data_for_launch = []

        return {
            "test_name": self.test_code_name,
            "priority_data_for_launch": priority_data_for_launch,
            "all_data_for_launch": all_data_for_launch,
            "ready_for_launch": ready_for_launch
        }

    @staticmethod
    def get_tests_for_data_key(data_key: str, graph_tests: List[GraphTest]):
        return [t for t in graph_tests if t.contains_required_data(data_key)]

    @staticmethod
    def read_tests_info_file_as_dict(graph_path: str) -> dict[str, GraphTest]:
        return {t.test_code_name: t for t in GraphTest.read_tests_info_file(graph_path)}

    @staticmethod
    def read_tests_info_file(graph_path: str) -> List[GraphTest]:
        with open(graph_path, 'r', encoding='utf-8') as file:
            kern_graph_data = json.loads(file.read())

        return [GraphTest(code_name=d['test_key'],
                          class_name=EnumQAQCClass[d['test_group']],
                          test_name_in_code=d['test_name'],
                          required_data=[names['alternative_names'] for names in d['required_data']]) for d in
                kern_graph_data]

    def launch_test(self, data_arr: [DataInfo]):
        print(self.test_code_name, [d for d in data_arr])
