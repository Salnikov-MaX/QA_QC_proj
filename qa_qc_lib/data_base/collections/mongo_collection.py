from typing import Callable, Type, Generic
from pymongo import MongoClient
from qa_qc_lib.data_base.collections.base_collection import ICollection, IFactoryCollection, T


class MongoCollection(Generic[T], ICollection[T]):
    def __init__(self, client: MongoClient, table_name: str):
        self.client = client
        self.table_name = table_name
        self.table = client[table_name]

    def find(self, item_id: str) -> dict:
        pass

    def where(self, where_func: Callable[[T], bool]) -> list[T]:
        return [self.from_dict(i) for i in self.table.find() if where_func(i)]

    def delete(self, key: str):
        pass

    def set_range(self, items: dict[str, T]):
        pass

    def set(self, key: str, item: T):
        pass


class MongoFactoryCollection(IFactoryCollection):
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)['qa_qc']

    def create_collection(self, table_name: str, item_type_: Type[T] = dict) -> MongoCollection[T]:
        return MongoCollection[item_type_](self.client, table_name)
