from dataclasses import dataclass
from typing import Any

import bson


@dataclass
class Domain:
    _id: bson.objectid.ObjectId
    name: str
    specification_modules: dict[str, dict]


@dataclass
class Node:
    _id: bson.objectid.ObjectId


@dataclass
class PrimaryData:
    _id: bson.objectid.ObjectId
    type_data: str
    values: dict
    well_name: str


@dataclass
class Reader:
    _id: bson.objectid.ObjectId
    name_function: str
    parametres: list[Any]
    domain: str
    primary_data: list[str]


@dataclass
class TestResult:
    _id: bson.objectid.ObjectId


@dataclass
class Test:
    _id: bson.objectid.ObjectId
    name: str
    types_nodes: list[dict]
    order_of_test: int
    name_visualizer: str
    primary_data: list[dict]
    parametres: dict


@dataclass
class NodeType:
    _id: bson.objectid.ObjectId
    name: str
    domain: str
    category: str
    attributes: list[str]


@dataclass
class TypePrimaryData:
    _id: bson.objectid.ObjectId
    name: str
    domain: list[str]
    required: bool
    reader: str
    description: str
    well_data: bool


@dataclass
class Well:
    _id: bson.objectid.ObjectId


@dataclass
class DataCategory:
    _id: bson.objectid.ObjectId
    name: str
    domain: list[str]
    reader: str
    description: str
    extensions_files: list[str]
