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

```json
{
  "data_keys": [
    "ACTNUM|GRDECL|ПЕТРОФИЗИКА|",
    "J-function|GRDECL|ПЕТРОФИЗИКА|",
    "NTG|GRDECL|ПЕТРОФИЗИКА|",
    "PermX|GRDECL|ПЕТРОФИЗИКА|",
    "PermY|GRDECL|ПЕТРОФИЗИКА|",
    "PermZ|GRDECL|ПЕТРОФИЗИКА|",
    "Porosity|GRDECL|ПЕТРОФИЗИКА|",
    "SGCR|GRDECL|ПЕТРОФИЗИКА|",
    "SGL|GRDECL|ПЕТРОФИЗИКА|",
    "SGU|GRDECL|ПЕТРОФИЗИКА|",
    "SOGCR|GRDECL|ПЕТРОФИЗИКА|",
    "SOWCR|GRDECL|ПЕТРОФИЗИКА|",
    "SWATINIT|GRDECL|ПЕТРОФИЗИКА|",
    "SWCR|GRDECL|ПЕТРОФИЗИКА|",
    "SWL|GRDECL|ПЕТРОФИЗИКА|",
    "SWU|GRDECL|ПЕТРОФИЗИКА|",
    "Sg|ASCIIGRID|ПЕТРОФИЗИКА|",
    "Sg|GRDECL|ПЕТРОФИЗИКА|",
    "So|ASCIIGRID|ПЕТРОФИЗИКА|",
    "So|GRDECL|ПЕТРОФИЗИКА|",
    "Sw|ASCIIGRID|ПЕТРОФИЗИКА|",
    "ВНК|ASCIIGRID|ПЕТРОФИЗИКА|",
    "ГНК|ASCIIGRID|ПЕТРОФИЗИКА|"
  ]
}
```


