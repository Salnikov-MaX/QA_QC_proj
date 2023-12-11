from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.tests.wells.wells_nodes import Nodes_wells_data
from qa_qc_lib.tests.wells.wells_tests import QA_QC_wells as Tests_wells_data


@dataclass
class WellTestConfig:
    wells: List[str]
    tests: List[str]

    @staticmethod
    def get_well_section_config(data_map: DataMap) -> Optional[WellTestConfig]:
        if data_map.well is None:
            return None

        nodes_obj = Nodes_wells_data(data_map.well.well_dir, tuple(data_map.well.well_files))
        tests_wells = Tests_wells_data(nodes_obj, 'data\wells_data')

        tests = tests_wells.order_tests[1] + tests_wells.order_tests[2]
        wells = nodes_obj.wells

        return WellTestConfig(wells, tests)
