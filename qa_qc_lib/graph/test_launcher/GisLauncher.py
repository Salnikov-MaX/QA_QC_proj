import itertools
import os.path
from typing import Optional

from qa_qc_lib.graph.data_map.GisMap import GisData
from qa_qc_lib.graph.test_config.GisTestConfig import GisTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.readers.gis_reader import Reader_gis_data_for_well
from qa_qc_lib.tests.gis.gis_nodes import Nodes_gis_data
from qa_qc_lib.tests.gis.gis_tests import QA_QC_gis


class GisLauncher(BaseLauncher):
    def __init__(self, gis_config: GisTestConfig, gis_data: GisData, report_dir: str):
        self.gis_config = gis_config
        self.gis_data = gis_data
        self.report_dir = report_dir

    def start_qa_qc(self) -> [Optional[dict]]:
        """


        """
        if not os.path.isdir(self.report_dir):
            os.mkdir(self.report_dir)

        report_data_dir = os.path.join(self.report_dir, 'gis')

        if not os.path.isdir(report_data_dir):
            os.mkdir(report_data_dir)

        gis_reader = Reader_gis_data_for_well(self.gis_data.stratum_name,
                                              self.gis_data.mnemonics_file_path,
                                              self.gis_data.well_tops_file_path)

        reports = []
        for las_file in self.gis_data.gis_file_paths:
            gis_nodes_for_well = Nodes_gis_data(las_file, gis_reader)
            gis_nodes_for_well.check_data()

            tests_gis = QA_QC_gis(gis_nodes_for_well, report_data_dir)

            test_groups = [node.test_groups for node in self.gis_config.gis_nodes]

            for group in itertools.chain(*test_groups):
                for test in group.tests:
                    test_f = getattr(tests_gis, test.test_name_code)
                    report = test_f(group.data_key.split('|')[0], get_report=False)

                    if report['data_availability'] and report['result'] is False:
                        report['report_data'] = tests_gis.report_function[test.test_name_code](report['specification'])

                    reports.append(report)

        return reports
