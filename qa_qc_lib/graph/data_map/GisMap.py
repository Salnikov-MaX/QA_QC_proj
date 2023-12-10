import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GisData:
    """
    Property:
        well_tops_file_path - Имя exel файла c кровлей и подошвой пласта в точках скважины.
        gis_file_paths - Пути до las файлов, с гис.
        stratum_name - Имя пласта.
        mnemonics_file_path = Имя exel файла с мнемониками узлов ГИС.
            Первый колонка - Имя узла.
            Колонка с названием: 'Мнемоники' - перечисление мнемоник через запятую.
    """

    well_tops_file_path: Optional[str]
    gis_file_paths: List[str]
    stratum_name: str
    mnemonics_file_path: str

    def __post_init__(self):
        if self.gis_file_paths is None or self.gis_file_paths is []:
            raise Exception(f'Не указаны гис файлы. (gis_file_paths)')

        names_and_paths = [('gis_file_path', f) for f in self.gis_file_paths]
        names_and_paths.append(('mnemonics_file_path', self.mnemonics_file_path))

        if self.well_tops_file_path is not None:
            names_and_paths.append(('well_tops_file_path', self.well_tops_file_path))

        for property_name, file_path in names_and_paths:
            if not os.path.isfile(file_path):
                raise Exception(f"Файла {property_name} по пути: \'{file_path}\' не найдено.")
