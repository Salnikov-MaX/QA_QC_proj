from qa_qc_lib.data_base.collections.base_collection import IFactoryCollection

from qa_qc_lib.data_base.models import *


class DataContext:
    def __init__(self, factory_collection: IFactoryCollection):
        create = factory_collection.create_collection
        self.Domains = create('domains', Domain)
        self.Nodes = create('nodes', Node)
        self.PrimaryDatas = create('primary_data', PrimaryData)
        self.Readers = create('readers', Reader)
        self.TestResults = create('resuls_of_tests', TestResult)
        self.Tests = create('tests', Test)
        self.TypeNodes = create('types_nodes', NodeType)
        self.PrimaryDataTypes = create('types_of_primary_data', TypePrimaryData)
        self.Wells = create('wells', Well)
        self.DataCategories = create('categories_data', DataCategory)
