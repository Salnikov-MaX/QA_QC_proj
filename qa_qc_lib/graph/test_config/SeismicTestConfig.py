from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_config.TestConfig import DataGroupTests, get_data_groups


@dataclass
class SeismicTestConfig:
    test_groups: List[DataGroupTests]

    @staticmethod
    def get_seismic_section_config(data_map: DataMap, graph: Graph) -> Optional[SeismicTestConfig]:
        if data_map.seismic is None:
            return None

        data_keys = list(set([file.data_key for file in data_map.seismic.data_files]))
        return SeismicTestConfig(get_data_groups(data_map, graph, data_keys))
