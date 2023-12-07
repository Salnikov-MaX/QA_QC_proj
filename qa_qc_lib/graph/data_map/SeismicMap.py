from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SeismicDataInfo:
    data_key: str
    file_path: str


@dataclass
class SeismicData:
    data_files: List[SeismicDataInfo]

    def find_by_data_key(self, data_key: str) -> Optional[SeismicDataInfo]:
        for d_k in self.data_files:
            if d_k.data_key == data_key:
                return d_k

        return
