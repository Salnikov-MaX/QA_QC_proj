import argparse

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

