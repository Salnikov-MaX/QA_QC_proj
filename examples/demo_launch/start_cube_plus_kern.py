from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.test_config.MainTestConfig import MainTestConfig
from qa_qc_lib.graph.test_launcher.LaunchTest import LaunchTest

if __name__ == '__main__':
    data_map = DataMap.read_map('map.json')

    main_test_config = MainTestConfig.create_main_test_config(data_map)
    main_test_config.data.kern = None
    main_test_config.data.well = None
    main_test_config.data.cube = None

    launch_test = LaunchTest(main_test_config)
    launch_test.start_tests('reports')
