import argparse
import os
import pathlib
from typing import Optional

from qa_qc_lib.graph.tools.read_map import read_map

parser = argparse.ArgumentParser()

parser.add_argument("--map_path",
                    type=str,
                    help="Путь до файла сопоставления")

parser.add_argument("--config_path",
                    type=str,
                    help="Путь до конфигурационного файла")


# предусмотреть вариант несколько данных с одинаковыми ключами
# data = [a1, a2, b1, b2]
# tests = [(a), (b), (a, b)]

# start_test = [(a1), (a1), (b1), (b2), (a1, b1), (a2, b1), (a1, b2), (a2, b2)]

if __name__ == '__main__':
    args = parser.parse_args()
    map_path = str(args.map_path).strip('\'').strip('\"')
    x = read_map(map_path)
    print(x)

