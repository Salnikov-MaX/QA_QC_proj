import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class WellData:
    """
    well_dir - Директория где хранится
    well_files - файлы с данными по скважинам (.vol) или (.SMSPEC + .UNSMRY)
    """
    well_dir: str
    well_files: List[str]

    def __post_init__(self):
        if not os.path.isdir(self.well_dir):
            raise Exception(f"Директории по указанному пути не существует: \'{self.well_dir}\'")
        for well_file in self.well_files:
            if not os.path.isfile(os.path.join(self.well_dir, well_file)):
                raise Exception(f"Файла по указанному пути не существует: \'{well_file}\'")
