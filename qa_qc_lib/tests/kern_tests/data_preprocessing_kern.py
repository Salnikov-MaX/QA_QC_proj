import os
import pandas as pd
import numpy as np
from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts


class DataPreprocessing:
    def __init__(self):
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

    def process_data(self, columns_mapping, path_to_save="..\\..\\data\\post_test_table.xlsx"):
        """
            Проходится по файлам и из каждого файла берет нужный столбец. Собирает единую таблицу.
        Args:
            path_to_save(string):путь для сохранения файла
            columns_mapping (dic[string:string]): -словарь формата путь до файла->расшифровка параметра

        Returns:
            excel: таблица с собранными параметрами
        """
        self.df_result = pd.DataFrame(columns=self.headers)
        # получаем название столбца и путь, откуда его брать
        for col_name, file_col in columns_mapping.items():
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

        self.save_to_excel(path_to_save)

    def save_to_excel(self, path_save):
        self.df_result.to_excel(path_save,
                                sheet_name='Sheet1', index=False)
