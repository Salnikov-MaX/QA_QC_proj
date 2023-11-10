import json

import pandas as pd


class AlternativeData:
    def __init__(self, alternative_data: str):
        self.all_alternative_name = []
        alternative_data = alternative_data.replace('|/', '|;')
        for name in alternative_data.split(';'):
            self.all_alternative_name.append(name.strip())

    def __str__(self):
        return f'({" || ".join(self.all_alternative_name)})'


class RequiredData:
    def __init__(self, required_data: str):
        self.required: [AlternativeData] = []
        for alternative_data in required_data.split('vs'):
            self.required.append(AlternativeData(alternative_data))

    def __str__(self):
        result = [str(i) for i in self.required]
        return "\n".join(result)

    def get_data_as_dict(self, valid_keys: [str]):
        return [{"alternative_names": [r for r in r.all_alternative_name if r in valid_keys]} for r in self.required]


def generate_test_map_from_csv(csv_paths: [str], save_path: str, data_valid_keys: [str]):
    """
    Сохраняет конфигурационный файл содержащий граф тестирования данных

    Args:
        csv_paths: пути до файлов CSV
        save_path: путь сохранения итогового json файла
        data_valid_keys: коллекция ключевых именований данных

    """

    dfs = [pd.read_csv(csv_path, delimiter=',') for csv_path in csv_paths]
    df = pd.concat(dfs)
    df = df[df['FLAG'].astype(float) >= 1]
    df['test_code_name'] = df['Источник данных'].astype(str) + df['№'].astype(int).astype(str)

    keys = []
    all_data = []

    for _, row in df.iterrows():
        inner_data = (row['Входные данные']
                      .replace('vs.', 'vs')
                      .replace('|,', '|;'))

        data = RequiredData(inner_data)
        all_data.append({
            "test_key": row['test_code_name'],
            "test_name": row['Название теста в коде'],
            "required_data": data.get_data_as_dict(data_valid_keys)
        })

        for d in data.required:
            keys += d.all_alternative_name

    with open(save_path, 'w+', encoding='utf-8') as file:
        json.dump(all_data, file, ensure_ascii=False)

    ignore_keys = list(set(keys) - set(data_valid_keys))

    print('Не валидные имена тестов:',
          [{row['test_code_name']: row['Название теста в коде']}
           for _, row in df.iterrows()
           if row['Название теста в коде'][:4] != 'test'])
    print('Не валидные ключи из csv:', ignore_keys)
