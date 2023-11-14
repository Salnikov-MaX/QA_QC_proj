import json

if __name__ == '__main__':
    with open('data/result.json', 'r', encoding='utf-8') as file:
        test_configs = json.load(file)

    for data_config in test_configs:
        for test_config in data_config['tests']:
            if test_config['ready_for_launch']:
                print(test_config)
