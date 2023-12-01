from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.test_config.MainTestConfig import MainTestConfig
from qa_qc_lib.graph.test_launcher.LaunchTest import LaunchTest

if __name__ == '__main__':
    data_map = DataMap.read_map('data/map.json')

    main_test_config = MainTestConfig.create_main_test_config(data_map)

    # main_test_config.kern_config.tests.pop(0)
    # for g in main_test_config.cubes_config.group_test:
        # g.tests = [t for t in g.tests if t.test_name_code.__contains__('test_swl_sw')]
        # g.tests = [t for t in g.tests if t.test_name_code.__contains__('test_sum')]

    main_test_config.well_config.wells = main_test_config.well_config.wells[:2]

    launch_test = LaunchTest(main_test_config)
    launch_test.start_tests('reports')
