from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_config.TestConfig import DataGroupTests, get_data_groups
from qa_qc_lib.readers.gis_reader import Reader_gis_data_for_well
from qa_qc_lib.tests.gis.gis_nodes import Nodes_gis_data
from qa_qc_lib.tests.gis.gis_tests import QA_QC_gis


@dataclass
class GisTestNode:
    well_name: str
    test_groups: List[DataGroupTests]


@dataclass
class GisTestConfig:
    gis_nodes: List[GisTestNode]

    @staticmethod
    def get_gis_section_config(data_map: DataMap, graph: Graph) -> Optional[GisTestConfig]:
        if data_map.gis is None:
            return None

        gis_reader = Reader_gis_data_for_well(data_map.gis.stratum_name,
                                              data_map.gis.mnemonics_file_path,
                                              data_map.gis.well_tops_file_path)

        gis_nodes: List[GisTestNode] = []

        for gis_file in data_map.gis.gis_file_paths:
            well_gis_nodes = Nodes_gis_data(gis_file, gis_reader)

            tests_gis = QA_QC_gis(well_gis_nodes, '')
            nodes = [node + '|las|ПЕТРОФИЗИКА|' for node in tests_gis.nodes_obj.gis_nodes.keys()]
            if data_map.gis.well_tops_file_path is not None:
                nodes.append('Отбивки_пластопересечений(стратиграфия)|txt(xlsx)|ПЕТРОФИЗИКА|')

            test_groups = get_data_groups(data_map, graph, nodes)

            gis_nodes.append(GisTestNode(well_gis_nodes.well_name, test_groups))

        return GisTestConfig(gis_nodes)
