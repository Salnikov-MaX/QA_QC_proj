from typing import Optional, List

from qa_qc_lib.graph.test_config import KernTestConfig, CubesTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.tools.data_map import KernPathInfo
from qa_qc_lib.graph.tools.graph import Graph
from qa_qc_lib.tests.cubes_tests.cubes import QA_QC_cubes
from qa_qc_lib.tests.kern_tests.data_preprocessing_kern import DataPreprocessing
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern


class KernLauncher(BaseLauncher):
    def __init__(self, graph: Graph, kern_config: CubesTestConfig):
        self.graph = graph
        self.kern_config = kern_config

    @staticmethod
    def init_cubes(kern_config: KernTestConfig) -> QA_QC_cubes:
        pass

    def start_qa_qc(self) -> [Optional[dict]]:
        return []
