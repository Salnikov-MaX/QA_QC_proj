import os
import numpy as np
import pandas as pd

from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts


class CoreNode:
    def __init__(self):
        self.name = ""
        self.well_name = ""
        self.MD = []
        self.data = []
        self.direction = None  # не обязательный массив, только при наличии столбца Направление
        self.wrong_values = []
        self.wrong_description = ""


class CoreData:
    def __init__(self, files, path_to_save="../../../data/post_test_table.xlsx", get_report=True):
        self.data = []
        self.files = files
        self.path_to_save = path_to_save
        self.df_result = None
        self.consts = KernConsts()
        self.headers = [
            self.consts.kp_open,
            self.consts.kp_abs,
            self.consts.kp_din,
            self.consts.kp_eff,
            self.consts.cut_off_kvo,
            self.consts.cut_off_clay,
            self.consts.cut_off_poro,
            self.consts.cut_off_perm,
            self.consts.dt_matrix,
            self.consts.dt_mud,
            self.consts.dt_shale,
            self.consts.j_func,
            self.consts.ro_matrix,
            self.consts.ro_mud,
            self.consts.ro_shale,
            self.consts.sg,
            self.consts.sgl,
            self.consts.so,
            self.consts.sogcr,
            self.consts.sw,
            self.consts.clay_water,
            self.consts.kern_removal,
            self.consts.kern_removal_perc,
            self.consts.kern_depth,
            self.consts.fractional_flow,
            self.consts.sampling_intervals,
            self.consts.capillarometry,
            self.consts.carbonation,
            self.consts.kvo,
            self.consts.kno,
            self.consts.humble_const,
            self.consts.archie_const,
            self.consts.poisson_coef,
            self.consts.kpr_abs_Y,
            self.consts.kpr_abs_Z,
            self.consts.kpr_rel,
            self.consts.kpr_phase,
            self.consts.sampling_bottom,
            self.consts.sampling_below_roof,
            self.consts.mineral_density,
            self.consts.volume_density,
            self.consts.saturation_param,
            self.consts.poro_param,
            self.consts.ads_density,
            self.consts.mms_density,
            self.consts.sampling_top,
            self.consts.vs,
            self.consts.vp,
            self.consts.wettability,
            self.consts.rw,
            self.consts.perm_eff,
            self.consts.well,
            self.consts.md
        ]
        self.get_report = get_report
        self.__process_data()
        self.__create_core_structure()

    def delete_file(self):
        """
        Метод для удаления промежуточного файла, если он не требуется для предоставление результата.

        """
        if not self.get_report:
            os.remove(self.path_to_save)

    def __process_data(self):
        """
            Проходится по файлам и из каждого файла берет нужный столбец. Собирает единую таблицу.
        Args:
            self.path_to_save(string):путь для сохранения файла
            self.columns_mapping (dic[string:string]): -словарь формата путь до файла->расшифровка параметра

        Returns:
            excel: таблица с собранными параметрами
        """
        self.df_result = pd.DataFrame(columns=self.headers)
        # получаем название столбца и путь, откуда его брать
        for col_name, file_col in self.files.items():
            file_items = file_col.split("->")
            file_path = file_items[0]
            col_name_in_file = file_items[-1]
            if not os.path.exists(file_path):
                print(f"Предупреждение: Файл {file_path} не найден. Пропуск.")
                continue
            if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                sheet_name = file_items[1] if len(file_items) == 3 else 0
                data = pd.read_excel(file_path, sheet_name=sheet_name)
            elif file_path.endswith(".txt"):
                data = pd.read_csv(file_path, delimiter="\t")
            else:
                print(f"Предупреждение: Неизвестный формат файла {file_path}. Пропуск.")
                continue
            self.df_result[col_name] = data[col_name_in_file]
        # сортируем df так, чтобы все пустые колонки были справа
        column_order = np.concatenate(
            [self.df_result.columns[~self.df_result.isna().all(axis=0)],
             self.df_result.columns[self.df_result.isna().all(axis=0)]])
        self.df_result = self.df_result[column_order]
        columns_with_data = []

        # Проходим по DataFrame и добавляем названия колонок с данными в список
        for col in self.df_result.columns:
            if self.df_result[col].notna().any():
                columns_with_data.append(col)

        self.save_to_excel()

    def __apply_filter(self, data, value, operation) -> list:
        """
        Вспомогательный метод для фильтрации данных.
        Args:
            data(array[int[]):массив с данными для фильтрации
            value(str):значение с которым происходят операции фильтрации
            operation: операции фильтрации [<,>,=,!=]

        Returns:

        """
        data = np.array(data)

        if operation == "=":
            return data[data == value].tolist()

        elif operation == "!=":
            return data[data != value].tolist()

        elif operation == ">":
            return data[data > value].tolist()

        elif operation == "<":
            return data[data < value].tolist()

    def save_to_excel(self):
        self.df_result.to_excel(self.path_to_save, sheet_name='Sheet1', index=False)

    def __create_core_structure(self):
        """
        Метод для разбивания файла на отдельные узлы CoreNode
        """
        data_frame = pd.read_excel(self.path_to_save)
        columns = data_frame.columns
        for column in columns:
            core_node = CoreNode()
            core_node.name = column
            core_node.MD = data_frame["MD"].tolist()
            core_node.data = data_frame[column].tolist()

            if "Скважина" in columns:
                core_node.well_name = data_frame["Скважина"].tolist()

            if "Направление" in columns:
                core_node.direction = data_frame["Направление"].tolist()

            self.data.append(core_node)

    def addNode(self, coreNode):
        """
        Метод для добавления нового узла для хранения.
        Args:
            coreNode(CoreNode): узел, который надо добавить.

        Returns:

        """
        self.data.append(coreNode)

    def getAllData(self) -> list[CoreNode]:
        """
        Метод для получения всех узлов.
        """
        return self.data

    def getNode(self, node_name, filters=None) -> CoreNode:
        """
        Метод для получения одного узла по имени
        Args:
            node_name(str):
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]
        Returns:
            CoreNode: искомый узел, отфильтрованный по заданным параметрам.
        """
        if filters is None:
            filters = []
        filtered_node = CoreNode()
        for node in self.data:
            if node.name == node_name:
                filtered_node = node

        for filter in filters:
            filter_name = filter["name"]
            filter_value = filter["value"]
            filter_operation = filter["operation"]
            if filter_name == "Скважина":
                filtered_node.well_name = self.__apply_filter(filtered_node.well_name, filter_value, filter_operation)
            if filter_name == "MD":
                filtered_node.MD = self.__apply_filter(filtered_node.MD, filter_value, filter_operation)
            elif filter_name == "data":
                filtered_node.data = self.__apply_filter(filtered_node.data, filter_value, filter_operation)
            elif filter_name == "direction":
                filtered_node.direction = self.__apply_filter(filtered_node.direction, filter_value, filter_operation)

        return filtered_node
