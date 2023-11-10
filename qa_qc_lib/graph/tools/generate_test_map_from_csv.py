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


df1 = pd.read_csv(
    '../../../scenarios_for_testing/graph_scenarios/Тесты первого порядка 4c26c782b8344372a162c96b9db807c8_all.csv', delimiter=',')
df2 = pd.read_csv(
    '../../../scenarios_for_testing/graph_scenarios/Тесты второго порядка c7d28051a1e9439bbd82c125966c5e6f_all.csv', delimiter=',')
df = pd.concat([df1, df2])

df = df[df['FLAG'].astype(float) > 0]

keys = []
all_data = []

with open('../../../scenarios_for_testing/graph_scenarios/data_keys.txt', encoding='utf-8') as file:
    good_keys = file.read().splitlines()

for iii, row in df.iterrows():
    inner_data = (row['Входные данные']
                  .replace('vs.', 'vs')
                  .replace('|,', '|;'))

    data = RequiredData(inner_data)
    all_data.append({
        "test_key": str(row['Источник данных']) + str(int(row['№'])),
        "test_name": row['Название теста в коде'],
        "required_data": data.get_data_as_dict(good_keys)
    })

    for d in data.required:
        keys += d.all_alternative_name

with open('../config/kern_graph.json', 'w+', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False)

all_keys = list(set(keys))
ignore_keys = list(set(all_keys) - set(good_keys))

print('Не валидные имена тестов:',
      [{str(row['Источник данных']) + str(int(row['№'])): row['Название теста в коде']} for _, row in df.iterrows() if
       row['Название теста в коде'][:4] != 'test'])
print('Не валидные ключи из csv:', ignore_keys)
