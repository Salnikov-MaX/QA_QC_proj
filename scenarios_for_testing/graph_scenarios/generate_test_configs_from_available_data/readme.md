## Генерация файла конфигурации запуска на основе доступных данных

---
Требуемые данные:

- файл map.json:

### Пример:

```json
{
  "settings": {
    "show_tests_not_ready_for_launch": true
  },
  "map_files": [
    {
      "data_key": "Кно(Sowcr)|txt/xlsx|Керн|",
      "data_path": "C:\\Users\\KosachevIV\\Desktop\\QAQC\\QA_QC_proj\\data\\PermeabilityParallelFail(L).xlsx"
    }
  ]
}
```

## settings:
* ***show_tests_not_ready_for_launch***: True - в файле конфигурации запуска будут настройка для тестов которые
не доступны для запуская из-за отсутствия всех необходимых данных
## map_file:
* ***file_key***: str ключевое имя файла (Дать ссылку на документ с описанием ключевых слов файлов)
* ***file_path***: str путь до файла

