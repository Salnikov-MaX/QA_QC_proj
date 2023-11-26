from abc import ABCMeta, abstractmethod
from typing import Optional, List


class BaseLauncher(metaclass=ABCMeta):
    @abstractmethod
    def start_qa_qc(self) -> List[Optional[dict]]:
        pass
