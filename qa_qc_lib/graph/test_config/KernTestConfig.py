from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.test_config.TestConfig import DataGroupTests, get_data_groups
from qa_qc_lib.graph.graph import Graph


@dataclass
class KernTestConfig:
    tests: List[DataGroupTests]

    @staticmethod
    def get_kern_section_config(data_map: DataMap, graph: Graph) -> Optional[KernTestConfig]:
        if data_map.kern is None:
            return None

        kern_data_keys = list()
        for kern_file in data_map.kern.files:
            kern_data_keys += [n.data_key for n in kern_file.map]
        kern_data_keys = list(set(kern_data_keys))

        return KernTestConfig(get_data_groups(data_map, graph, kern_data_keys))
