import pandas as pd


# df = df[df['FLAG'] > 0 ]

# df['data_name'] = df['Источник данных'] + df['№'].astype(str)


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


df1 = pd.read_csv('Керн Тесты первого порядка.csv', delimiter=',')
df2 = pd.read_csv('Керн Тесты второго порядка.csv', delimiter=',')
df = pd.concat([df1, df2])

df = df[df['FLAG'].astype(float) > 0]

keys = []

for iii, row in enumerate(df['Входные данные']):
    row = row.replace('vs.', 'vs')
    row = row.replace('|,', '|;')
    data = RequiredData(row)
    for d in data.required:
        keys += d.all_alternative_name
    print(row)
    print(data)
    print()

with open('data_keys.txt', encoding= 'utf-8') as file:
    good_keys = file.read().splitlines()

all_keys = list(set(keys))

print(set(all_keys) - set(good_keys))

