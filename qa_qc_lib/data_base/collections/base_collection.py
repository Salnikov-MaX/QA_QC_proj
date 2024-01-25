from __future__ import annotations

import abc
import dataclasses
from typing import Callable, TypeVar, Generic, Type

from dacite import from_dict

T = TypeVar('T')


class ICollection(Generic[T], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def find(self, item_id: str) -> T:
        pass

    @abc.abstractmethod
    def where(self, where_func: Callable[[T], bool]) -> list[T]:
        pass

    @abc.abstractmethod
    def set(self, key: str, item: T):
        pass

    @abc.abstractmethod
    def set_range(self, items: dict[str, T]):
        pass

    @abc.abstractmethod
    def delete(self, key: str):
        pass

    def from_dict(self, item: dict) -> T:
        item_type = self.__orig_class__.__args__[0]
        if dataclasses.is_dataclass(item_type):
            return from_dict(item_type, item)
        return item

    @staticmethod
    def to_dict(item: T) -> dict:
        if hasattr(item, __dict__):
            return item.__dict__
        return item


class IFactoryCollection(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_collection(self, table_name: str, item_type_: Type[T] = dict) -> ICollection[T]:
        pass
