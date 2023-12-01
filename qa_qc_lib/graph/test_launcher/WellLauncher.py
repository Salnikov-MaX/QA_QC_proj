from typing import Optional, List

from qa_qc_lib.graph.data_map.WellMap import WellData
from qa_qc_lib.graph.test_config import CubeTestConfig
from qa_qc_lib.graph.test_config.WellTestConfig import WellTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.data_map.CubeMap import CubeData
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.tests.cubes_tests.cubes import QA_QC_cubes
from qa_qc_lib.tests.wells.wells_nodes import Nodes_wells_data
from qa_qc_lib.tests.wells.wells_tests import QA_QC_wells


class WellLauncher(BaseLauncher):
    def __init__(self, well_config: WellTestConfig, well_data: WellData, report_dir: str):
        self.well_config = well_config
        self.well_data = well_data
        self.report_dir = report_dir

    def start_qa_qc(self) -> [Optional[dict]]:
        """
        always:
            test_right_actnum

        data:
            test_permeability J-function -- ???


        """
        nodes_obj = Nodes_wells_data(self.well_data.well_dir, tuple(self.well_data.well_files))
        tests_wells = QA_QC_wells(nodes_obj, self.report_dir)

        first_tests = [t for t in tests_wells.order_tests[1] if t in self.well_config.tests]
        second_tests = [t for t in tests_wells.order_tests[2] if t in self.well_config.tests]

        reports = []
        for well in self.well_config.wells:
            if well not in self.well_config.wells:
                continue

            for k, v in nodes_obj.nodes_wells[well].items():
                for test in first_tests:
                    test_f = getattr(tests_wells, test)

                    res_test = test_f(v, k, well, get_report=False)

                    reports.append(res_test)
                    if res_test['data_availability'] and not res_test['result']:
                        tests_wells.report_function[test](res_test['specification'], saving=True)

            for test in second_tests:
                test_f = getattr(tests_wells, test)
                res_test = test_f(well, get_report=False)

                reports.append(res_test)

                if res_test['data_availability'] and not res_test['result']:
                    tests_wells.report_function[test](res_test['specification'], saving=True)

        return reports
