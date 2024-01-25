from qa_qc_lib.data_base.collections.mongo_collection import MongoFactoryCollection
from qa_qc_lib.data_base.data_context import DataContext

if __name__ == '__main__':
    config = MongoFactoryCollection('mongodb://w09866.main.hw.tpu.ru:27017/qa_qc')
    data_context = DataContext(config)

    d = data_context.DomainTypes.where(lambda _: True)
    for i in d:
        print(i.specification_modules)
