from typing import Optional, List

from qa_qc_lib.graph.test_config import CubeTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.data_map.CubeMap import CubeData
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.tests.cubes_tests.cubes import QA_QC_cubes


class CubeLauncher(BaseLauncher):
    def __init__(self, graph: Graph, cube_config: CubeTestConfig, cube_data: CubeData):
        self.graph = graph
        self.cube_config = cube_config
        self.cube_data = cube_data

    @staticmethod
    def init_cubes(cube_data: CubeData) -> QA_QC_cubes:
        data_dict = {pr.data_key: pr.data_path for pr in cube_data.property_files}
        qa_qc_cubes = QA_QC_cubes(cube_data.grid_dir,
                                  cube_data.grid_name,
                                  open_porosity_file_path=data_dict.get('Porosity|GRDECL|ПЕТРОФИЗИКА|'),
                                  open_perm_x_file_path=data_dict.get('PermX|GRDECL|ПЕТРОФИЗИКА|'),
                                  open_perm_y_file_path=data_dict.get('PermY|GRDECL|ПЕТРОФИЗИКА|'),
                                  open_perm_z_file_path=data_dict.get('PermZ|GRDECL|ПЕТРОФИЗИКА|'),
                                  litatype_file_path=data_dict.get(''),
                                  sgcr_file_path=data_dict.get('SGCR|GRDECL|ПЕТРОФИЗИКА|'),
                                  sgl_file_path=data_dict.get('SGL|GRDECL|ПЕТРОФИЗИКА|'),
                                  sogcr_file_path=data_dict.get('SOGCR|GRDECL|ПЕТРОФИЗИКА|'),
                                  sowcr_file_path=data_dict.get('SOWCR|GRDECL|ПЕТРОФИЗИКА|'),
                                  sw_file_path=data_dict.get('SWATINIT|GRDECL|ПЕТРОФИЗИКА|')
                                               or data_dict.get('Sw|ASCIIGRID|ПЕТРОФИЗИКА|'),
                                  sgu_file_path=data_dict.get('SGU|GRDECL|ПЕТРОФИЗИКА|'),
                                  swl_file_path=data_dict.get('SWL|GRDECL|ПЕТРОФИЗИКА|'),
                                  swcr_file_path=data_dict.get('SWCR|GRDECL|ПЕТРОФИЗИКА|'),
                                  swu_file_path=data_dict.get('SWU|GRDECL|ПЕТРОФИЗИКА|'),
                                  ntg_file_path=data_dict.get('NTG|GRDECL|ПЕТРОФИЗИКА|'),
                                  so_file_path=data_dict.get('So|ASCIIGRID|ПЕТРОФИЗИКА|')
                                               or data_dict.get('So|GRDECL|ПЕТРОФИЗИКА|'),
                                  sg_file_path=data_dict.get('Sg|ASCIIGRID|ПЕТРОФИЗИКА|')
                                               or data_dict.get('Sg|GRDECL|ПЕТРОФИЗИКА|'))

        return qa_qc_cubes

    def start_qa_qc(self) -> [Optional[dict]]:
        """
        always:
            test_right_actnum

        data:
            test_permeability J-function -- ???


        """

        test_names: List[str] = []

        for g in self.cube_config.test_groups:
            test_names += [t.test_name_code for t in g.tests]

        qa_qc_cube = self.init_cubes(self.cube_data)

        reports = qa_qc_cube.start_tests(test_names, get_report=False)
        return reports
