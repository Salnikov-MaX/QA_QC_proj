import os
from typing import Callable, TypeVar, Generic, Type

import elara

from qa_qc_lib.data_base.collections.base_collection import ICollection, IFactoryCollection

T = TypeVar('T')


class ElaraCollection(Generic[T], ICollection[T]):

    def __init__(self, db_dir_path: str, collection_name: str):
        bd_path = os.path.join(db_dir_path, collection_name)
        self.db = elara.exe_secure(bd_path)

    def find(self, item_id: str) -> T:
        return self.from_dict(self.db.get(item_id))

    def where(self, where_func: Callable[[T], bool]) -> list[T]:
        all_keys = self.db.getkeys()
        items: list[T] = []
        for item_key in all_keys:
            item = self.find(item_key)
            if where_func(item):
                items.append(item)

        return items

    def set(self, key: str, item: T) -> None:
        if isinstance(item, dict):
            self.db.set(key, {'id': key, **item})
        else:
            self.db.set(key, item.__dict__)

        self.db.commit()

    def set_range(self, items: dict[str, T]):
        for item_key in items:
            self.db.set(item_key, items[item_key])
        self.db.commit()

    def delete(self, key: str):
        self.db.rem(key)


class ElaraFactoryCollection(IFactoryCollection):
    def __init__(self, bd_dir_path):
        self.BdDirPath: str = bd_dir_path
        super().__init__()

    def create_collection(self, table_name: str, item_type_: Type[T] = dict) -> ElaraCollection[type(T)]:
        return ElaraCollection[item_type_](self.BdDirPath, table_name)
