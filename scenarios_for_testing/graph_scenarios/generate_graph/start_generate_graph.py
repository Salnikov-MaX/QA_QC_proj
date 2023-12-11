from os import listdir
from os.path import join

from qa_qc_lib.graph.graph import Graph

if __name__ == '__main__':
    csv_dir_path = 'graph_csvs'
    csv_paths = [join(csv_dir_path, f) for f in listdir(csv_dir_path)]

    result_path = f'graph.json'
    Graph.convert_graph_from_csv_to_json(csv_paths, result_path)
