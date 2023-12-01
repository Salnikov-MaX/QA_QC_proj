from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.test_config.TestConfig import DataGroupTests, get_data_groups
from qa_qc_lib.graph.graph import Graph


@dataclass
class CubeTestConfig:
    group_test: List[DataGroupTests]

    @staticmethod
    def get_cube_section_config(data_map: DataMap, graph: Graph) -> Optional[CubeTestConfig]:
        if data_map.cube is None:
            return None

        cube_data_keys = list(set([cube_file.data_key for cube_file in data_map.cube.property_files]))
        return CubeTestConfig(get_data_groups(data_map, graph, cube_data_keys))
