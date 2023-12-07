from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.test_config.WellTestConfig import WellTestConfig
from qa_qc_lib.graph.test_config.CubeTestConfig import CubeTestConfig
from qa_qc_lib.graph.test_config.KernTestConfig import KernTestConfig
from qa_qc_lib.graph.graph import Graph


@dataclass
class MainTestConfig:
    data: DataMap
    kern_config: Optional[KernTestConfig]
    cubes_config: Optional[CubeTestConfig]
    well_config: Optional[WellTestConfig]

    @staticmethod
    def create_main_test_config(data_map: DataMap, graph: Optional[Graph] = None) -> MainTestConfig:
        graph = graph or Graph()

        kern = KernTestConfig.get_kern_section_config(data_map, graph)
        cube = CubeTestConfig.get_cube_section_config(data_map, graph)
        well = WellTestConfig.get_well_section_config(data_map)
        return MainTestConfig(data_map, kern, cube, well)
