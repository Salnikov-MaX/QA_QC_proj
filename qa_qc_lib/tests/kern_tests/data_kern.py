import pandas as pd
import numpy as np
from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts


class DataKern:
    def __init__(self, file_path=r"..\..\..\data\core_data\post_test_table.xlsx"):
        self.consts = KernConsts()
        self.file_path = file_path
        self.dict_array = []

    def __color_cells(self, df, error_indices, column, color='red'):
        """
        Args:
            df(dataframe): dataframe - для выделения выпадающих значений
            error_indices(np.ndarray[int]): индексы с ошибками
            column(string): название колонки
            color(string): цвет, в который покрасить значения

        Returns:

        """
        style = pd.DataFrame('', index=df.index, columns=df.columns)
        for err in error_indices:
            style.loc[err, column] = f'background-color: {color}'
        return style

    def get_attributes(self, column_names, filters=None):
        """
        Метод для получения массивов переданных параметров
        Args:
            column_names(array[string]): массив с названиями параметров, которы необходимо получить
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            filtered_df: dataframe с требуемыми параметрами и примененными фильтрами
        """
        filters = filters or {}
        post_test_df = pd.read_excel(self.file_path, usecols=column_names)
        for filter_item in filters:
            column_name = filter_item.get('name')
            value = filter_item.get('value')
            operation = filter_item.get('operation')
            if column_name not in post_test_df.columns:
                continue

            if operation == '=':
                post_test_df = post_test_df[post_test_df[column_name] == value]
            elif operation == '>':
                post_test_df = post_test_df[post_test_df[column_name] > value]
            elif operation == '<':
                post_test_df = post_test_df[post_test_df[column_name] < value]
            elif operation == '<=':
                post_test_df = post_test_df[post_test_df[column_name] <= value]
            elif operation == '>=':
                post_test_df = post_test_df[post_test_df[column_name] >= value]
            elif operation == '!=':
                post_test_df = post_test_df[post_test_df[column_name] != value]
        # Выбираем нужные колонки
        filtered_df = post_test_df[column_names]
        return filtered_df

    def mark_errors(self):
        """
        Метод для окрашивания аномальных значений.

        Args:
            self.dict_array - структура массива [{"test_name":[param_name,
                                                              error_desc,
                                                              error_mask,
                                                              md]}] - нужен для пометки значений
        Returns:
            excel - таблица с покрашенными значениями и отмеченными ошибками
        """
        post_test_df = pd.read_excel(self.file_path)

        for dict_array in self.dict_array:
            for test_name, values in dict_array.items():
                column_name, error_description, error_mask, md = values
                error_indices = md[error_mask == 1]
                if not test_name in post_test_df.columns:
                    post_test_df[test_name] = np.nan
                post_test_df[test_name] = post_test_df[test_name].astype(object)
                post_test_df.loc[post_test_df.index.isin(error_indices), test_name] = error_description

        styled_df = post_test_df.style
        for dict_array in self.dict_array:
            for test_name, values in dict_array.items():
                column_name, _, error_mask, md = values
                error_indices = md[error_mask == 1]
                styled_df = styled_df.apply(self.__color_cells, axis=None, error_indices=error_indices,
                                            column=column_name)

        styled_df.to_excel(self.file_path, sheet_name='Sheet1', index=False)
