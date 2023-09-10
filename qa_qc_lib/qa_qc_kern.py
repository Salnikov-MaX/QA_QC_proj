import datetime
from datetime import datetime
from typing import Any
import numpy as np
from qa_qc_lib.qa_qc_main import QA_QC_main
from scipy.stats import t
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from qa_qc_lib.qa_qc_tools.math_tools import linear_dependence_function
from qa_qc_lib.qa_qc_tools.math_tools import exponential_function
from qa_qc_lib.qa_qc_tools.kern_tools import linear_function_visualization, expon_function_visualization, \
    logarithmic_function_visualization


class QA_QC_kern(QA_QC_main):
    def __init__(self, pas=np.array([]), note=None, kno=np.array([]), kp_plast=None, density=None,
                 water_permeability=None, kp_pov=None, perpendicular=None, perpendicular_density=None, kp=np.array([]),
                 top=None, core_removal_in_meters=None, parallel_carbonate=None, perpendicular_carbonate=None,
                 perpendicular_porosity=None, intervals=None, bottom=None, percent_core_removal=None,
                 outreach_in_meters=None, sw_residual=np.array([]), core_sampling=None, kpr=None, parallel_density=None,
                 parallel_porosity=None, parallel=None, rp=None, pmu=None, rn=None, obplnas=None, poroTBU=None,
                 poroHe=None, porosity_open=np.array([]), porosity_kerosine=None, porosity_effective=None, sw=None,
                 parallel_permeability=None, klickenberg_permeability=None, effective_permeability=None, md=None,
                 kgo=None, knmng=None,
                 lithology=None, file_report_name="test_report", show=True, file_path="report\\",
                 file_name="не указан", r2=0.7) -> None:
        """_summary_

        Args:
            data (str): _description_
        """

        super().__init__()
        self.knmng = knmng
        self.kgo = kgo
        self.md = md
        self.water_permeability = water_permeability
        self.effective_permeability = effective_permeability
        self.klickenberg_permeability = klickenberg_permeability
        self.parallel_permeability = parallel_permeability
        self.water_saturation = sw
        self.poroHe = poroHe
        self.residual_water_saturation = sw_residual
        self.porosity_effective = porosity_effective
        self.porosity_kerosine = porosity_kerosine
        self.porosity_open = porosity_open
        self.obplnas = obplnas
        self.kno = kno
        self.pas = pas
        self.kp_din = []
        self.kp_plast = kp_plast
        self.kp_pov = kp_pov
        self.density = density
        self.kp_ef = []
        self.rn = rn
        self.pmu = pmu
        self.rp = rp
        self.parallel = parallel
        self.parallel_porosity = parallel_porosity
        self.parallel_density = parallel_density
        self.kpr = kpr
        self.core_sampling = core_sampling
        self.sw_residual = sw_residual
        self.outreach_in_meters = outreach_in_meters
        self.percent_core_removal = percent_core_removal
        self.bottom = bottom
        self.intervals = intervals
        self.poro_tbu = poroTBU
        self.perpendicular_porosity = perpendicular_porosity
        self.perpendicular_carbonate = perpendicular_carbonate
        self.parallel_carbonate = parallel_carbonate
        self.core_removal_in_meters = core_removal_in_meters
        self.top = top
        self.kp = kp
        self.perpendicular_density = perpendicular_density
        self.perpendicular = perpendicular
        self.table = note
        self.file_name = file_name
        self.r2 = r2
        self.dt_now = datetime.now()
        self.file = open(f"{file_path}/{file_report_name}.txt", "w")
        self.dict_of_wrong_values = {}
        self.get_report = show
        self.lithology = lithology
        self.__parameter_calculation()

    def __del__(self):
        self.file.close()

    def __check_data(self, array, param_name, test_name):
        """
        Тест предназначен для проверки условия - все элементы массива должны быть числовыми.

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """
        if not isinstance(array, np.ndarray):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / {test_name}:\n Не запускался. Причина {param_name}" \
                                f" не является массивом. Входной файл {self.file_name}\n\n"
            self.dict_of_wrong_values[test_name] = [{param_name: [0]}, "не является массивом"]
            return False

        try:
            lent = array.size
            elem = array[0]
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / {test_name}:\n Не запускался. Причина {param_name}" \
                                f" пустой. Входной файл {self.file_name}\n\n"
            self.dict_of_wrong_values[test_name] = [{param_name: [0]}, "пустой"]
            return False

        for i in range(array.size):
            if array[i] == "nan":
                self.dict_of_wrong_values[test_name] = [{
                    param_name: [i]}, "содержит nan"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / {test_name}:\n Не запускался. Причина {param_name}" \
                                    f" содержит nan. Входной файл {self.file_name}\n\n"

                return False

        return True

    def __water_saturation(self, array) -> tuple[bool, list[int]]:
        wrong_values = []
        result = True
        for i in range(len(array)):
            if array[i] < 0 or array[i] > 1:
                result = False
                wrong_values.append(i)
        return result, wrong_values

    def __test_porosity(self, array) -> tuple[bool, list[int]]:
        wrong_values = []
        result = True
        if array[0] < 1:
            for i in range(len(array)):
                if array[i] < 0 or array[i] > 0.476:
                    result = False
                    wrong_values.append(i)
        else:
            for i in range(len(array)):
                if array[i] < 0 or array[i] > 47.6:
                    result = False
                    wrong_values.append(i)
        return result, wrong_values

    def __permeability(self, array) -> tuple[bool, list[int]]:
        wrong_values = []
        result = True
        for i in range(len(array)):
            if array[i] < 0:
                result = False
                wrong_values.append(i)
        return result, wrong_values

    def __parameter_calculation(self):
        try:
            if self.porosity_open is not None and self.sw_residual is not None:
                for i in range(len(self.porosity_open)):
                    self.kp_ef.append(self.porosity_open[i] * (1 - self.sw_residual[i]))
                self.kp_ef = np.array(self.kp_ef)
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10}:\nНе удалось вычислить значение Эффективная пористость\n\n"
            self.kp_ef = np.array([])
        try:
            if self.porosity_open is not None and self.sw_residual is not None and self.kno is not None:
                for i in range(len(self.porosity_open)):
                    self.kp_din.append(self.porosity_open[i] * (1 - self.sw_residual[i] - self.kno[i]))
                self.kp_din = np.array(self.kp_din)
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10}:\nНе удалось вычислить значение Динамическая пористость\n\n"
            self.kp_din = np.array([])
        try:
            if self.pmu is None:
                self.pmu = []
                if self.pas is not None and self.kp is not None:
                    for i in range(len(self.pas)):
                        self.pmu.append(self.pas[i] + (self.kp[i] * 1))
                    self.pmu = np.array(self.pmu)
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10}:\nНе удалось вычислить значение Объемная плотность\n\n"
            self.pmu = np.array([])

    """
    Тесты первого порядка 
    """

    def test_open_porosity(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

            Required data:
                Открытая пористость по жидкости;
            Args:
                self.porosity_open (array[int/float]): массив с открытой пористостью в атмосферных условия для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.porosity_open, "Открытая пористость по жидкости", "test_open_porosity"):
            result, wrong_values = self.__test_porosity(self.porosity_open)
            if not result:
                report_text = f"{result}.\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 47,6."
            else:
                report_text = f"{result}."
            self.dict_of_wrong_values["test open porosity"] = [{
                "Открытая пористость по жидкости": wrong_values,
            }, "не лежит в интервале от 0 до 47,6"]

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_open_porosity:\n{report_text}\n\n"

            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_HE(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

            Args:
                self.poroHe (array[int/float]): массив с открытой пористостью по гелию для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.poroHe, "Открытая пористость по газу", "test porosity HE"):
            result, wrong_values = self.__test_porosity(self.poroHe)
            if not result:
                report_text = f"{result}.\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 47,6."
            else:
                report_text = f"{result}."
            self.dict_of_wrong_values["test open porosity HE"] = [{"Открытая пористость по газу": wrong_values},
                                                                  "не лежит в интервале от 0 до 47,6"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_porosity_HE:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_TBU(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

            Required data:
                Открытая пористость в пластовых условиях
            Args:
                self.poro_tbu (array[int/float]): массив с открытой пористостью в пластовых условиях для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.poro_tbu, "Открытая пористость в пластовых условиях", "test porosity TBU"):
            result, wrong_values = self.__test_porosity(self.poro_tbu)
            if not result:
                report_text = f"{result}.\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 47,6."
            else:
                report_text = f" {result}."
            self.dict_of_wrong_values["test open porosity TBU"] = [{
                "Открытая пористость в пластовых условиях": wrong_values}, "не лежит в интервале от 0 до 47,6"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_porosity_TBU:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_kerosine(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

            Required data:
                Открытая пористость по керосину
            Args:
                self.porosity_kerosine (array[int/float]): массив с открытой пористостью по керосину для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.porosity_kerosine, "Открытая пористость по керосину", "test porosity kerosine"):
            result, wrong_values = self.__test_porosity(self.porosity_kerosine)
            if not result:
                report_text = f"{result}.\nДанные с индексом {wrong_values} лежат не в" \
                              f"интервале от 0 до 47,6."
            else:
                report_text = f" {result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_porosity_kerosine:\n{report_text}\n\n"
            self.dict_of_wrong_values["test porosity kerosine"] = [{"Открытая пористость по керосину": wrong_values},
                                                                   "не лежит в интервале от 0 до 47,6"]
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_effective(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

            Required data:
                Эффективная пористость;
            Args:
                self.porosity_effective (array[int/float]): массив с эффективной пористостью для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.porosity_effective, "porosity effective", "test porosity effective"):
            result, wrong_values = self.__test_porosity(self.porosity_effective)
            if not result:
                report_text = f"Test 'test porosity effective': {result}." \
                              f"\nДанные с индексом {wrong_values} лежат не в" \
                              f"интервале от 0 до 47,6."
            else:
                report_text = f"Test 'test porosity effective': {result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.dict_of_wrong_values["test_porosity_effective"] = [{"Кво": wrong_values},
                                                                    "не соответсвует интервалу от 0 до 1"]
            self.report_text += f"{timestamp:10} / test_porosity_effective:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_residual_water_saturation(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]
            Required data:
                Кво;
            Args:
                self.residual_water_saturation (array[int/float]): массив с остаточной водонасыщенностью для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.residual_water_saturation, "Кво",
                             "test residual water saturation"):
            result, wrong_values = self.__water_saturation(self.residual_water_saturation)
            if not result:
                report_text = f"Test 'test residual water saturation': {result}.\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"Test 'test residual water saturation': {result}."
            self.dict_of_wrong_values["test residual water saturation"] = [{"Кво": wrong_values},
                                                                           "не соответсвует интервалу от 0 до 1"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_residual_water_saturation:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_water_saturation(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]
            Required data:
                Sw;
            Args:
                self.water_saturation (array[int/float]): массив с водонасыщенностью для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.water_saturation, "Sw", "test water saturation"):
            result, wrong_values = self.__water_saturation(self.water_saturation)
            if not result:
                report_text = f"Test 'test water saturation': {result}.\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"Test 'test water saturation': {result}."
            self.dict_of_wrong_values["test water saturation"] = [{"Sw": wrong_values},
                                                                  "не соответсвует интервалу от 0 до 1"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_water_saturation:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_parallel_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0
            Required data:
                Газопроницаемость, mkm2 (parallel);

            Args:
                self.parallel_permeability (array[int/float]): массив с газопроницаемостью
                                                               параллельно напластованию для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.parallel_permeability, "Газопроницаемость, mkm2 (parallel)",
                             "test parallel permeability"):
            result, wrong_values = self.__permeability(self.parallel_permeability)
            if not result:
                report_text = f"Test 'test_parallel_permeability': {result}." \
                              f"\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"Test 'test_parallel_permeability': {result}." \
                              f" Дата выполнения {self.dt_now}\n"
            self.dict_of_wrong_values["test parallel permeability"] = [
                {"Газопроницаемость, mkm2 (parallel)": wrong_values}, "значение меньше 0"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_parallel_permeability:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_klickenberg_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

            Required data:
                Газопроницаемость Кликенбергу;

            Args:
                self.klickenberg_permeability (array[int/float]): массив с газопроницаемостью
                                                                  с поправкой по Кликенбергу для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.klickenberg_permeability, "Газопроницаемость Кликенбергу",
                             "test klickenberg permeability"):
            result, wrong_values = self.__permeability(self.klickenberg_permeability)
            if not result:
                report_text = f"{result}." \
                              f"\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"{result}.\nВходной файл {self.file_name}."
            self.dict_of_wrong_values["test klickenberg permeability"] = [
                {"Газопроницаемость Кликенбергу": wrong_values}, "значение меньше 0"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_klickenberg_permeability:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_effective_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0
            Required data:
                Эффективная проницаемость;
            Args:
                self.effective_permeability (array[int/float]): массив с эффективной проницаемостью для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.effective_permeability, "Эффективная проницаемость", "test_effective_permeability"):
            result, wrong_values = self.__permeability(self.effective_permeability)
            if not result:
                report_text = f"{result}." \
                              f"\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"{result}."
            self.dict_of_wrong_values["test effective permeability"] = [{"Эффективная проницаемость": wrong_values},
                                                                        "значение меньше 0"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_effective_permeability:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name,
                    "date": self.dt_now}

    def test_water_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

           Required data:
                Газопроницаемость по воде;
            Args:
                self.water_permeability (array[int/float]): массив с газопроницаемостью по воде для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.water_permeability, "Газопроницаемость по воде", "test_water_permeability"):
            result, wrong_values = self.__permeability(self.water_permeability)
            if not result:
                report_text = f"{result}." \
                              f"\nДанные с индексом {wrong_values} лежат не в " \
                              f"интервале от 0 до 1."
            else:
                report_text = f"result"
            self.dict_of_wrong_values["test water permeability"] = [{"Газопроницаемость по воде": wrong_values},
                                                                    "значение меньше 0"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_water_permeability:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name,
                    "date": self.dt_now}

    def test_monotony(self, get_report=True) -> dict[str, bool | datetime | list[int] | str]:
        """
        Тест предназначен для проверки монотонности возрастания значения глубины

            Required data:
                Глубина отбора, м
            Args:
                self.core_sampling (array[int/float]): массив с местом отбора для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.core_sampling, "Глубина отбора, м", "test monotony"):
            result = True
            wrong_values = []
            for i in range(len(self.core_sampling) - 1):
                if self.core_sampling[i + 1] - self.core_sampling[i] <= 0:
                    result = False
                    wrong_values.append(i)
            self.dict_of_wrong_values["test monotony"] = [{"Глубина отбора, м": wrong_values},
                                                          "нарушена монотонность"]
            if not result:
                report_text = f"{result}.\nДанные с индексом {wrong_values} не монотоны."

            else:
                report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_monotony:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name,
                    "date": self.dt_now}

    """
        Тесты второго порядка
    """

    def test_quo_kp_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

            Required data:
                Кво; Коэффициентом пористости

            Args:
                self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
                self.kp (array[int/float]): массив с данными Открытая пористость по жидкости для проверки

            Returns:
                image: визуализация кроссплота
                dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.sw_residual, "Кво",
                             "test quo kp dependence") and \
                self.__check_data(self.kp, "Открытая пористость по жидкости", "test quo kp dependence"):

            r2 = self.test_general_dependency_checking(self.sw_residual, self.kp, "test quo kp dependence",
                                                       "Кво",
                                                       "Коэффициентом пористости")["r2"]
            result = True
            a, b = linear_dependence_function(self.kp, self.sw_residual)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.sw_residual, a, b, r2,
                                                                         get_report, "Коэффициентом пористости", "Кво",
                                                                         "test_quo_kp_dependence")

            self.dict_of_wrong_values["test_quo_kp_dependence"] = [{"Кво": wrong_values1,
                                                                    "Открытая пористость по жидкости": wrong_values2
                                                                    }, "выпадает из линии тренда"]
            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_quo_kp_dependence:\n{report_text}\n\n"

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_density_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции y=a*x+b, при этом a<0

        Required data:
            Открытая пористость по жидкости;Плотность абсолютно сухого образца
        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.density (array[int/float]): массив с данными плотности для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp, "Открытая пористость по жидкости", "test kp density dependence") and \
                self.__check_data(self.density, "Плотность абсолютно сухого образца", "test kp density dependence"):

            r2 = self.test_general_dependency_checking(self.kp, self.density, "test kp density dependence",
                                                       "Коэффициента пористости",
                                                       "Плотности")["r2"]

            result = True
            a, b = linear_dependence_function(self.kp, self.density)
            if a >= 0 or r2 < 0.7:
                result = False

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_density_dependence:\n{report_text}\n\n"
            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.density, a, b, r2, get_report,
                                                                         "Коэффициента пористости", "Плотности",
                                                                         "kp_density_dependence")
            self.dict_of_wrong_values["test kp density dependence"] = [
                {"Открытая пористость по жидкости": wrong_values1,
                 "Плотность абсолютно сухого образца": wrong_values2,
                 }, "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kgo_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*x+b, при этом a<0

        Required data:
            Коэффициента пористости; Cвязанная газонасыщенность

        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.kgo (array[int/float]): массив с данными связанная газонасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp, "Открытая пористость по жидкости", "test_kp_kgo_dependence") and \
                self.__check_data(self.kgo, "Cвязанная газонасыщенность", "test_kp_kgo_dependence"):

            r2 = self.test_general_dependency_checking(self.kp, self.kgo, "test_kp_kgo_dependence",
                                                       "Коэффициента пористости",
                                                       "Cвязанная газонасыщенность")["r2"]

            result = True
            a, b = linear_dependence_function(self.kp, self.kgo)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.kgo, a, b, r2, get_report,
                                                                         "Коэффициента пористости",
                                                                         "Cвязанная газонасыщенность",
                                                                         "test_kp_kgo_dependence")
            report_text = f"Test 'kp_kgo_dependence': {result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_kgo_dependence:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_kp_kgo_dependence"] = [
                {"Открытая пористость по жидкости": wrong_values1,
                 "Cвязанная газонасыщенность": wrong_values2,
                 }, "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_knmng_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных) и текущей.  Характерной
        зависимостью является линейная по функции y=a*x+b, при этом a<0

        Required data:
            Коэффициента пористости; Критическая нефтенасыщенность

        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.knmng (array[int/float]): массив с данными критическая нефтенасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp, "Открытая пористость по жидкости", "test_kp_knmng_dependence") and \
                self.__check_data(self.knmng, "Критическая нефтенасыщенность", "test_kp_knmng_dependence"):

            r2 = self.test_general_dependency_checking(self.kp, self.knmng, "test_kp_knmng_dependence",
                                                       "Коэффициента пористости",
                                                       "Критическая нефтенасыщенность")["r2"]

            result = True
            a, b = linear_dependence_function(self.kp, self.knmng)
            if a >= 0 or r2 < 0.7:
                result = False

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_knmng_dependence:\n{report_text}\n\n"
            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.knmng, a, b, r2, get_report,
                                                                         "Коэффициента пористости",
                                                                         "Cвязанная газонасыщенность",
                                                                         "test_kp_knmng_dependence")
            self.dict_of_wrong_values["test_kp_knmng_dependence"] = [
                {"Открытая пористость по жидкости": wrong_values1,
                 "Критическая нефтенасыщенность": wrong_values2,
                 }, "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kno_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*x+b, при этом a<0

        Required data:
            Коэффициента пористости; Коэффициент остаточной нефтенасыщенности

        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.kno (array[int/float]): массив с данными связанная газонасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp, "Открытая пористость по жидкости", "test_kp_kno_dependence") and \
                self.__check_data(self.kno, "Коэффициент остаточной нефтенасыщенности", "test_kp_kno_dependence"):

            r2 = self.test_general_dependency_checking(self.kp, self.kno, "test_kp_kno_dependence",
                                                       "Коэффициента пористости",
                                                       "Коэффициент остаточной нефтенасыщенности")["r2"]

            result = True
            a, b = linear_dependence_function(self.kp, self.kno)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.knmng, a, b, r2, get_report,
                                                                         "Коэффициента пористости",
                                                                         "Коэффициент остаточной нефтенасыщенности",
                                                                         "test_kp_kno_dependence")

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_kno_dependence:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_kp_kno_dependence"] = [
                {"Открытая пористость по жидкости": wrong_values1,
                 "Коэффициент остаточной нефтенасыщенности": wrong_values2,
                 }, "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_sw_residual_kp_din_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции y=a*x+b, при этом a<0

        Required data:
            Кво; Коэффициент остаточной нефтенасыщенности; Открытая пористость по жидкости

        Args:
            self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kp_din (array[int/float]): массив с данными коэффициент динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sw_residual, "Кво", "test sw_residual kp din dependence") and \
                self.__check_data(self.kp_din, "kp din", "test sw_residual kp din dependence"):

            r2 = self.test_general_dependency_checking(self.sw_residual, self.kp_din,
                                                       "test sw_residual kp din dependence",
                                                       "Коэффициента остаточной водонасыщенности",
                                                       "Коэффициента динамической пористости")["r2"]
            result = True
            a, b = linear_dependence_function(self.kp_din, self.sw_residual)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.sw_residual, self.kp_din, a, b, r2,
                                                                         get_report,
                                                                         "Коэффициент остаточной водонасыщенности",
                                                                         "Коэффициент динамической пористости",
                                                                         "test_sw_residual_kp_din_dependence")

            report_text = {result}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_sw_residual_kp_din_dependence:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_sw_residual_kp_din_dependence"] = [{"Кво": wrong_values1},
                                                                               "выпадает из линии тренда"]
            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_obblnas_kp_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для проверки физичности
        взаимосвязи двух кроссплотов - Обплнас-Кп и
        Минпл-Кп. Пусть первый аппроксимируется
        линией тренда y=a1*x+b1, а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Минералогическая плотность; Открытая пористость по жидкости; Объемная плотность
        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.obblnas (array[int/float]): массив с данными объемная плотность для проверки
            self.pmu (array[int/float]): массив с данными минералогическая плотность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.pmu, "Минералогическая плотность", "test obblnas kp dependence") and \
                self.__check_data(self.kp, "Открытая пористость по жидкости", "test obblnas kp dependence") and \
                self.__check_data(self.obplnas, "Объемная плотность", "test obblnas kp dependence"):

            r2_pmu = self.test_general_dependency_checking(self.pmu, self.kp, "test obblnas kp dependence",
                                                           "Минералогической плотность",
                                                           "Коэффициента пористости")["r2"]
            r2_obp = self.test_general_dependency_checking(self.obplnas, self.kp, "test obblnas kp dependence",
                                                           "Объемной плотность",
                                                           "Коэффициента пористости")["r2"]

            coeffs1 = np.polyfit(self.kp, self.pmu, 1)
            a1, b1 = coeffs1[0], coeffs1[1]
            trend_line1 = np.polyval(coeffs1, self.pmu)

            coeffs2 = np.polyfit(self.kp, self.obplnas, 1)
            a2, b2 = coeffs2[0], coeffs2[1]
            trend_line2 = np.polyval(coeffs2, self.obplnas)

            result = True
            if a1 >= a2 or r2_obp < 0.7 or r2_pmu < 0.7:
                result = False

            wrong_values1 = []
            wrong_values2 = []
            for i in range(len(self.obplnas)):
                if self.obplnas[i] < a1 * self.kp[i] + b1:
                    wrong_values1.append(self.obplnas[i])
                    wrong_values2.append(self.kp[i])
            self.dict_of_wrong_values["test_obblnas_kp_dependence"] = [{"Объемная плотность": wrong_values1,
                                                                        "Открытая пористость по жидкости": wrong_values2},
                                                                       "выпадает из линии тренда"]

            report_text = f" {result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_obblnas_kp_dependence:\n{report_text}\n\n"

            y_pred = a2 * self.kp + b2

            # Окрашиваем точки, которые не соответствуют линии тренда, в красный
            for obplnas_val, kp_val, pred_val in zip(self.obplnas, self.kp, y_pred):
                if obplnas_val + (pred_val * 0.1) < pred_val:
                    plt.scatter(kp_val, obplnas_val, color='g')

            plt.title("test obblnas kp dependence")
            plt.scatter(self.kp, self.obplnas, color='red', label='Обплнас-Кп')
            plt.scatter(self.kp, self.pmu, color='blue', label='Минпл-Кп')
            plt.plot(self.kp, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
            plt.plot(self.kp, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
            plt.xlabel('kp')
            plt.ylabel('obplnas')
            plt.legend()
            plt.grid(True)
            equation = f'y = {a1:.2f}x + {b1:.2f}, r2_pmu={r2_pmu:.2f}'
            plt.text(np.mean(self.kp), np.min(self.pmu) + 2, equation, ha='center', va='bottom')
            equation = f'y = {a2:.2f}x + {b2:.2f}, r2_obp={r2_obp:.2f}'
            plt.text(np.mean(self.kp), np.min(self.obplnas), equation, ha='center', va='bottom')
            plt.savefig("report\\test_obblnas_kp_dependence.png")
            if get_report:
                plt.show()

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_pmu_kp_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для проверки физичности
        взаимосвязи двух кроссплотов - Обплнас-Кп и
        Минпл-Кп. Пусть первый аппроксимируется
        линией тренда y=a1*x+b1, а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Минералогическая плотность; Открытая пористость по жидкости; Объемная плотность
        Args:
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки
            self.obblnas (array[int/float]): массив с данными объемная плотность для проверки
            self.pmu (array[int/float]): массив с данными минералогическая плотность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.pmu, "Минералогическая плотность", "test pmu kp dependence") and \
                self.__check_data(self.kp, "Открытая пористость по жидкости", "test pmu kp dependence") and \
                self.__check_data(self.obplnas, "Объемная плотность", "test pmu kp dependence"):

            r2_pmu = self.test_general_dependency_checking(self.pmu, self.kp, "test pmu kp dependence",
                                                           "Минералогической плотность",
                                                           "Коэффициента пористости")["r2"]
            r2_obp = self.test_general_dependency_checking(self.obplnas, self.kp, "test pmu kp dependence",
                                                           "Объемной плотность",
                                                           "Коэффициента пористости")["r2"]

            coeffs1 = np.polyfit(self.kp, self.obplnas, 1)
            a1, b1 = coeffs1[0], coeffs1[1]
            trend_line1 = np.polyval(coeffs1, self.obplnas)

            coeffs2 = np.polyfit(self.kp, self.pmu, 1)
            a2, b2 = coeffs2[0], coeffs2[1]
            trend_line2 = np.polyval(coeffs2, self.pmu)

            result = True
            if a1 <= a2 or r2_obp < 0.7 or r2_pmu < 0.7:
                result = False

            wrong_values1 = []
            wrong_values2 = []
            for i in range(len(self.pmu)):
                if self.pmu[i] < a2 * self.kp[i] + b2:
                    wrong_values1.append(self.pmu[i])
                    wrong_values2.append(self.kp[i])
            self.dict_of_wrong_values["test_pmu_kp_dependence"] = [{"Минералогическая плотность": wrong_values1,
                                                                    "Открытая пористость по жидкости": wrong_values2},
                                                                   "выпадает из линии тренда"]
            report_text = f"Test 'dependence pmu kp': {result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_pmu_kp_dependence:\n{report_text}\n\n"
            plt.title("test pmu kp dependence")
            plt.scatter(self.obplnas, self.kp, color='red', label='Обплнас-Кп')
            plt.scatter(self.pmu, self.kp, color='blue', label='Минпл-Кп')
            y_pred = a2 * self.kp + b2

            # Окрашиваем точки, которые не соответствуют линии тренда, в красный
            for pmu_val, kp_val, pred_val in zip(self.pmu, self.kp, y_pred):
                if pmu_val + (pred_val * 0.1) < pred_val:
                    plt.scatter(kp_val, pmu_val, color='g')

            plt.plot(self.kp, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
            plt.plot(self.kp, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
            plt.xlabel('kp')
            plt.ylabel('pmu')
            plt.legend()
            plt.grid(True)
            equation = f'y = {a1:.2f}x + {b1:.2f}, r2_obpl = {r2_obp}'
            plt.text(np.mean(self.pmu), np.min(self.kp) + 2, equation, ha='center', va='bottom')
            equation = f'y = {a2:.2f}x + {b2:.2f}, r2_pmu ={r2_pmu}'
            plt.text(np.mean(self.obplnas), np.min(self.kp), equation, ha='center', va='bottom')
            plt.savefig("report\\test_pmu_kp_dependence.png")

            if get_report:
                plt.show()

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_ef_kpdin_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость
        линейная по функции y=a*x+b, при этом a>0, b>0
        Required data:
            Открытая пористость; Кво; So
        Args:
            self.kp_ef (array[int/float]): массив с данными коэффициент эффективной пористости для проверки
            self.kp_din (array[int/float]): массив с данными коэффициент динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.kp_ef, "kp ef", "test kpf kp din dependence") \
                and self.__check_data(self.kp_din, "kp din", "test kpf kp din dependence"):

            r2 = self.test_general_dependency_checking(self.kp_ef, self.kp_din, "test kpf kp din dependence",
                                                       "Коэффициента эффективной пористости",
                                                       "Динамическая пористость")["r2"]
            result = True
            a, b = linear_dependence_function(self.kp_din, self.kp_ef)
            if a <= 0 or b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp_din, self.kp_ef, a, b, r2, get_report,
                                                                         "Коэффициент эффективной пористости",
                                                                         "Динамическая пористость",
                                                                         "test_kp_ef_kp_din_dependence")
            report_text = result
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_ef_kpdin_dependence:\n{report_text}\n\n"

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_ef_kp_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость
        линейная по функции y=a*x+b, при этом a>0, b<0

        Required data:
            Открытая пористость; Кво; Открытая пористость по жидкости;
        Args:
            self.kp_ef (array[int/float]): массив с данными коэффициента эффективной пористости для проверки
            self.kp (array[int/float]): массив с данными коэффициента пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp_ef, "kp ef", "test kp ef kp dependence") \
                and self.__check_data(self.kp, "Открытая пористость по жидкости", "test kp ef kp dependence"):

            r2 = self.test_general_dependency_checking(self.kp_ef, self.kp, "test kp ef kp dependence",
                                                       "Коэффициента эффективной пористости",
                                                       "Коэффициента пористости")["r2"]
            result = True
            a, b = linear_dependence_function(self.kp, self.kp_ef)
            if a <= 0 or b >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.kp_ef, a, b, r2, get_report,
                                                                         "Коэффициента пористости",
                                                                         "Коэффициент эффективной пористости",
                                                                         "test_kp_ef_kp_dependence")
            self.dict_of_wrong_values["test_kp_ef_kp_dependence"] = [{
                "Открытая пористость по жидкости": wrong_values2
            }, "выпадает из линии тренда"]
            report_text = {result}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_ef_kp_dependence:\n{report_text}\n\n"

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kp_din_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость
        линейная по функции y=a*x+b, при этом a>0, b<0

        Required data:
            Открытая пористость; Кво; So; Открытая пористость по жидкости

        Args:
            self.kp (array[int/float]): массив с данными Открытая пористость по жидкости для проверки
            self.kp_din (array[int/float]): массив с данными коэффициент динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp, "Открытая пористость по жидкости", "test kp kp din dependence") \
                and self.__check_data(self.kp_din, "kp din", "test kp kp din dependence"):
            r2 = self.test_general_dependency_checking(self.kp, self.kp_din, "test kp kp din dependence",
                                                       "Коэффициента пористости",
                                                       "Коэффициента динамической пористости")["r2"]
            a, b = linear_dependence_function(self.kp, self.kp_din)
            result = True
            if a <= 0 or b >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(self.kp, self.kp_din, a, b, r2, get_report,
                                                                         "Коэффициента пористости",
                                                                         "Коэффициент динамической пористости",
                                                                         "test_kp_kp_din_dependence")
            self.dict_of_wrong_values["test_kp_kp_din_dependence"] = [{"Открытая пористость по жидкости": wrong_values1,
                                                                       }, "выпадает из линии тренда"]
            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_kp_kp_din_dependence:\n{report_text}\n\n"

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kpr_kp(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость по
        функции y=a*exp(b*x) при этом b>0

        Required data:
            Кпр_газ(гелий); Открытая пористость по жидкости

        Args:
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки
            self.kp (array[int/float]): массив с данными Открытая пористость по жидкости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence kpr kp") and \
                self.__check_data(self.kp, "Открытая пористость по жидкости", "test dependence kpr kp"):
            r2 = self.test_general_dependency_checking(self.kpr, self.kp, "test dependence kpr kp",
                                                       "Коэффициента проницаемости",
                                                       "Коэффициента пористости")["r2"]

            result = True
            a, b = exponential_function(self.kp, self.kpr)
            if b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = expon_function_visualization(self.kp, self.kpr, a, b, r2, get_report,
                                                                        "Коэффициент пористости",
                                                                        "Коэффициент проницаемости",
                                                                        "test_dependence_kpr_kp")

            self.dict_of_wrong_values["test_dependence_kpr_kp"] = [{"Кпр_газ(гелий)": wrong_values1,
                                                                    "Открытая пористость по жидкости": wrong_values2},
                                                                   "выпадает из линии тренда"]
            report_text = {result}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_dependence_kpr_kp:\n{report_text}\n\n"
            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kpr_kp_din(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой для
        данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*exp(b*x) при этом b>0

        Required data:
            Открытая пористость; Кво; So; Кпр_газ(гелий)

        Args:
            self.kpr (array[int/float]): массив с данными коэффициента проницаемости для проверки
            self.kp_din (array[int/float]): массив с данными коэффициента динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence kpr kp din") and \
                self.__check_data(self.kp_din, "Kp_din", "test dependence kpr kp din"):

            r2 = self.test_general_dependency_checking(self.kpr, self.kp_din, "test dependence kpr kp din",
                                                       "Коэффициента проницаемости",
                                                       "Коэффициента динамической пористости")["r2"]
            result = True
            a, b = exponential_function(self.kp_din, self.kpr)
            if b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = expon_function_visualization(self.kpr, self.kp_din, a, b, r2, get_report,
                                                                        "Коэффициент проницаемости",
                                                                        "Коэффициент динамической пористости",
                                                                        "test_dependence_kpr_kp_din")

            self.dict_of_wrong_values["test_dependence_kpr_kp_din"] = [{"Кпр_газ(гелий)": wrong_values1,
                                                                        }, "выпадает из линии тренда"]
            report_text =  {result}
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} :\n{report_text}\n\n"
            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_sw_residual_kpr(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*ln(x)+b при этом a>0
        Required data:
            Кво; Кпр_газ(гелий)

        Args:
            self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.sw_residual, "Кво", "test dependence sw_residual kpr") and \
                self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence sw_residual kpr"):

            r2 = self.test_general_dependency_checking(self.sw_residual, self.kpr, "test dependence sw_residual kpr",
                                                       "Коэффициента остаточной водонасыщенности",
                                                       "Коэффициента проницаемости")["r2"]
            coefficients = np.polyfit(self.kpr, np.exp(self.sw_residual), 1)
            a, b = coefficients[0], coefficients[1]
            result = True
            if a <= 0 or r2 < 0.7:
                result = False
            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_dependence_sw_residual_kpr:\n{report_text}\n\n"
            wrong_values1, wrong_values2 = logarithmic_function_visualization(self.kpr, self.sw_residual, a, b, r2,
                                                                              get_report, "Коэффициент проницаемости",
                                                                              "Коэффициент остаточной водонасыщенности",
                                                                              "test_dependence_sw_residual_kpr")

            self.dict_of_wrong_values["test_dependence_sw_residual_kpr"] = [{"Кво": wrong_values1,
                                                                             "Кпр_газ(гелий)": wrong_values2},
                                                                            "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_rn_sw_residual_dependence(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=b/(kв^n) при этом 1,1<n<5
        Required data:
            Параметр насыщенности; Sw
        Args:
            self.rn (array[int/float]): массив с данными параметр насыщенности для проверки
            self.sw_residual (array[int/float]): массив с данными коэффициент водонасыщенности для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.rn, "Параметр насыщенности", "test rn sw_residual dependencies") \
                and self.__check_data(self.sw_residual, "Sw",
                                      "test rn sw_residual dependencies"):
            r2 = self.test_general_dependency_checking(self.rn, self.sw_residual,
                                                       "est rn sw_residual dependencies",
                                                       "Рн",
                                                       "Коэффициента водонасыщенности")["r2"]
            coefficients = np.polyfit(np.log(self.sw_residual), np.log(self.rn), 1)
            b, n = np.exp(coefficients[1]), coefficients[0]
            result = True
            if 1.1 >= n or n >= 5 or r2 < 0:
                result = False

            wrong_values1 = []
            wrong_values2 = []
            for i in range(len(self.rn)):
                if self.rn[i] > b / (self.sw_residual[i] ** n):
                    wrong_values1.append(self.rn[i])
                    wrong_values2.append(self.sw_residual[i])
            self.dict_of_wrong_values["test_rn_sw_residual_dependence"] = [{"Параметр насыщенности": wrong_values1,
                                                                            "Sw": wrong_values2
                                                                            }, "выпадает из линии тренда"]
            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_rn_sw_residual_dependence:\n{report_text}\n\n"
            plt.title("test rn sw_residual dependencies")
            y_pred = b / (self.sw_residual ** n)

            # Окрашиваем точки, которые не соответствуют линии тренда, в красный
            for sw_residual_val, rn_val, pred_val in zip(self.sw_residual, self.rn, y_pred):
                if rn_val + (pred_val * 0.1) < pred_val:
                    plt.scatter(sw_residual_val, rn_val, color='r')

            plt.scatter(self.sw_residual, self.rn, color='blue', label='Исходные данные')
            plt.plot(self.rn, b / (self.sw_residual ** n), color='red', label='Линия тренда')
            plt.xlabel('sw_residual')
            plt.ylabel('rn')
            plt.legend()
            equation = f'y = {b:.2f}/(x^{b:.2f}),  r2={r2:.2f}'
            plt.text(np.mean(self.sw_residual), np.min(self.rn), equation, ha='center', va='bottom')
            plt.savefig("report\\test_dependence_kpr_kp_din.png")
            if get_report:
                plt.show()

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_rp_kp_dependencies(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a/(kп^m)
        при этом m>0. a>0 и a<2,5, 1,1<m<3,8

        Required data:
            Параметр пористости; Открытая пористость по жидкости

        Args:
            self.rp (array[int/float]): массив с данными параметр пористости для проверки
            self.kp (array[int/float]): массив с данными Открытая пористость по жидкости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.rp, "Параметр пористости", "test rp kp dependencies") \
                and self.__check_data(self.kp, "Открытая пористость по жидкости", "test rp kp dependencies"):
            r2 = self.test_general_dependency_checking(self.rp, self.kp, "test rp kp dependencies",
                                                       "Параметр пористости",
                                                       "Коэффициента пористости")["r2"]
            coefficients = np.polyfit(np.log(self.kp), -np.log(self.rp), 1)
            a, m = np.exp(-coefficients[1]), coefficients[0]
            result = True
            if 1.1 >= m or m >= 3.8 or 0 >= a or a >= 2.5 or r2 < 0.7:
                result = False

            wrong_values1 = []
            wrong_values2 = []
            for i in range(len(self.rp)):
                if self.rp[i] > a / (self.kp[i] ** m):
                    wrong_values1.append(self.rp[i])
                    wrong_values2.append(self.kp[i])
            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_rp_kp_dependencies:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_rp_kp_dependencies"] = [{"Параметр пористости": wrong_values1,
                                                                     "Открытая пористость по жидкости": wrong_values2
                                                                     }, "выпадает из линии тренда"]
            plt.scatter(self.kp, self.rp, color='blue', label='Исходные данные')
            plt.plot(self.kp, a / (self.kp ** m), color='red', label='Линия тренда')
            y_pred = a / (self.kp ** m)

            # Окрашиваем точки, которые не соответствуют линии тренда, в красный
            for rp_val, kv_val, pred_val in zip(self.rp, self.kp, y_pred):
                if rp_val + (pred_val * 0.1) < pred_val:
                    plt.scatter(kv_val, rp_val, color='r')

            plt.title("test rp kp dependencies")
            plt.xlabel('kp')
            plt.ylabel('rp')
            plt.legend()
            equation = f'y = {a:.2f}/(x^{m:.2f}), r2={r2:.2f}'
            plt.text(np.mean(self.rp), np.min(self.kp), equation, ha='center', va='bottom')
            plt.savefig("report\\test_rp_kp_dependencies.png")
            if get_report:
                plt.show()

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_general_dependency_checking(self, x, y, test_name="не указано", x_name="не указано", y_name="не указано") \
            -> dict[str, bool | str | Any] | dict[
                str, bool | str | Any]:
        """
        Тест предназначен для оценки дисперсии входных данных.
        Он проводится по следующему алгоритму: изначально,
        используя статистические методы, детектируются и удаляются
        выбросные точки, затем полученное облако точек  аппроксимируется
        и считается коэффициент детерминации R2. Если его значение больше
        0.7, то тест считается пройденным. Если значение меньше 0.7, то
        точки сортируются по удаленности от линии тренда, и запускается
        цикл, за одну итерацию которого удаляется самая отдаленная от
        линии аппроксимации точка, и считается R2, если значение больше
        0.7, и удалено менее 10% точек, то тест пройден, иначе - нет.

            Args:
                x (array[int/float]): массив с данными для проверки
                y (array[int/float]): массив с данными для проверки

            Returns:
                dict[str, bool | str | Any] | dict[str, bool | str | Any]: словарь с результатом теста, значением
                                                                          Коэффициента r2, датой выполнения теста
                file: запись результата теста для сохранения состояния
        """
        alpha = 0.05  # Уровень значимости
        n = len(x)
        dof = n - 2  # Число степеней свободы для распределения Стьюдента

        residuals = y - np.polyval(np.polyfit(x, y, 1), x)
        std_error = np.sqrt(np.sum(residuals ** 2) / dof)

        t_critical = t.ppf(1 - alpha / 2, dof)
        upper_limit = np.polyval(np.polyfit(x, y, 1), x) + t_critical * std_error
        lower_limit = np.polyval(np.polyfit(x, y, 1), x) - t_critical * std_error

        x_filtered = []
        y_filtered = []

        for i in range(n):
            if lower_limit[i] <= y[i] <= upper_limit[i]:
                x_filtered.append(x[i])
                y_filtered.append(y[i])

        # Аппроксимация линии тренда на отфильтрованных данных
        coeffs = np.polyfit(x_filtered, y_filtered, 1)
        trend_line = np.polyval(coeffs, x_filtered)

        # Вычисление R2 score
        r2 = r2_score(y_filtered, trend_line)
        result = True
        # Проверка условия R2 score и удаление точек при несоответствии
        while r2 < self.r2 and len(x_filtered) > 0.9 * n:
            # Вычисление расстояний от линии тренда до каждой точки
            distances = np.abs(y_filtered - np.polyval(coeffs, x_filtered))

            # Сортировка расстояний по убыванию
            sorted_indices = np.argsort(distances)[::-1]

            # Удаление самой дальней точки
            x_filtered.pop(sorted_indices[0])
            y_filtered.pop(sorted_indices[0])

            # Повторное вычисление линии тренда и R2 score
            coeffs = np.polyfit(x_filtered, y_filtered, 1)
            trend_line = np.polyval(coeffs, x_filtered)
            r2 = r2_score(y_filtered, trend_line)

            # Проверка пройденного теста
            if r2 >= 0.7:
                result = True
                report_text = f"{result}. Выполнен для теста {test_name}.Тест проводился для {x_name} {y_name}. Коэффициент r2 - {r2}\n"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_general_dependency_checking:\n{report_text}\n\n"
                return {"result": result, "r2": r2, "file_name": self.file_name, "date": self.dt_now}
            else:
                result = False
        report_text = f"{result}. Выполнен для теста {test_name}.Тест проводился для {x_name} {y_name}. Коэффициент r2 - {r2}\n"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_general_dependency_checking:\n{report_text}\n\n"
        return {"result": result, "r2": r2, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_first(self, get_report=True):
        """
        Тест проводится для оценки отсутствия монотонности интервалов долбления. Т.е.,
        подошва вышележащего интервала долбления должна быть выше или равна кровле нижележащего
            Required data:
                Кровля интервала отбора; Подошва интервала отбора
            Args:
                self.top (array[int/float]): массив с данными кровли для проверки
                self.bottom (array[int/float]): массив с данными подошвы для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.top, "Кровля интервала отбора", "test coring depths first") \
                and self.__check_data(self.bottom, "Подошва интервала отбора", "test coring depths first"):
            result = True
            wrong_values = []
            for i in range(len(self.top)):
                if self.bottom[i] < self.top[i]:
                    result = False
                    wrong_values.append(i)
            self.dict_of_wrong_values["test_coring_depths_first"] = [{"Кровля интервала отбора": wrong_values,
                                                                      "Подошва интервала отбора": wrong_values, },
                                                                     "подошва вышележащего интервала долбления ниже кровли нижележащего"]
            if result:
                report_text = f"{result}."

            else:
                report_text = f"{result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_coring_depths_first:\n{report_text}\n\n"

            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_second(self, get_report=True):
        """
        Тест проводится для оценкци соответствия интервала долбления: подошва-кровля ≥ выносу в метрах
             Required data:
                Кровля интервала отбора; Подошва интервала отбора; Вынос керна, м
            Args:
                self.top (array[int/float]): массив с данными кровли для проверки
                self.bottom (array[int/float]): массив с данными подошвы для проверки
                self.core_removal_in_meters (array[int/float]): массив с данными выноса в метрах для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.top, "Кровля интервала отбора", "test coring depths second") \
                and self.__check_data(self.bottom, "Подошва интервала отбора", "test coring depths second") \
                and self.__check_data(self.core_removal_in_meters, "Вынос керна, м",
                                      "test coring depths second"):
            wrong_values = []
            result = True
            for i in range(len(self.top)):
                if self.bottom[i] - self.top[i] < self.core_removal_in_meters[i]:
                    wrong_values.append(i)
                    result = False
            self.dict_of_wrong_values["test_coring_depths_second"] = [{"Кровля интервала отбора": wrong_values,
                                                                       "Подошва интервала отбора": wrong_values,
                                                                       "Вынос керна, м": wrong_values
                                                                       },
                                                                      "разница между подошвой и кровлей меньше выноса в м"]
            if result:
                report_text = f" {result}."
            else:
                report_text = f"{result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_coring_depths_second:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_third(self, get_report=True):
        """
        Тест оценивает соответствие значений выноса керна в метрах и в процентах
             Required data:
                Вынос керна, м; Вынос керна, %
            Args:
                self.intervals (array[[int/float]]): массив с массивамими,
                                                    содержашими начало интервала и конец интервала
                self.percent_core_removal (array[int/float]): массив со значениями выноса в процентах
                self.outreach_in_meters(array[int/float]): массив со значениями выноса в метрах

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.outreach_in_meters, "Вынос керна, м", "test coring depths third") \
                and self.__check_data(self.percent_core_removal, "Вынос керна, %", "test coring depths third"):
            result = True
            wrong_values = []
            for i in range(len(self.intervals)):
                interval_length = max(self.intervals[i]) - min(self.intervals[i])
                displacement_percent_calculated = (self.outreach_in_meters[i] / interval_length) * 100
                if abs(self.percent_core_removal[i] - displacement_percent_calculated) > 0.001:
                    result = False
                    wrong_values.append(i)
            self.dict_of_wrong_values["test_coring_depths_third"] = [{"Вынос керна, %": wrong_values,
                                                                      "Вынос керна, м": wrong_values,
                                                                      },
                                                                     "вынос керна в процентах и метрах не совпадает"]

            if result:
                report_text = f" {result}."
            else:
                report_text = f"{result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_coring_depths_third:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_four(self, get_report=True):
        """
        Тест проводится с целью соответствия глубин отбора образцов с глубинами выноса керна
            Required data:
                Вынос керна, м; Глубина отбора, м; Подошва интервала отбора
            Args:
                self.core_removal_in_meters (array[int/float]): массив с выносом керна в метрах
                self.core_sampling (array[int/float]): массив с глубинами отбора образцов
                self.bottom(array[int/float]): массив с подошвой отбора образцов

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.core_removal_in_meters, "Вынос керна, м", "test coring depths four") \
                and self.__check_data(self.core_sampling, "Глубина отбора, м", "test coring depths four") \
                and self.__check_data(self.bottom, "Подошва интервала отбора", "test coring depths four"):
            result = True
            wrong_values = []
            for i in range(len(self.core_removal_in_meters)):
                if self.core_removal_in_meters[i] + self.core_sampling[i] > self.bottom[i]:
                    result = False
                    wrong_values.append(i)

            self.dict_of_wrong_values["test_coring_depths_four"] = [{"Глубина отбора, м": wrong_values,
                                                                     "Вынос керна, м": wrong_values,
                                                                     "Подошва интервала отбора": wrong_values,
                                                                     }, "глубина отбора ниже фактического выноса керна"]
            if result:
                report_text =result
            else:
                report_text =  f"{result}.Индексы элементов с ошибкой {wrong_values}."

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_coring_depths_four:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_data_tampering(self, get_report=True):
        """
        Тест выполняется с целью фиксации подлога измерений.
        Подлог заключается в том, что существуют значения параметров,
        схожие вплоть до 3-его знака после запятой
            Required data:
                    Кпр_газ; Открытая пористость по жидкости; Кво; Параметр пористости; Плотность абсолютно сухого образца; Sw; Параметр насыщенности
            Args:
                self.kpr (array[int/float]): массив с данными коэффициент проницаемости для сравнения
                self.kp (array[int/float]): массив с данными Открытая пористость по жидкости для сравнения
                self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для сравнения
                self.rp (array[int/float]): массив с данными параметр пористости для сравнения
                self.density (array[int/float]): массив с данными всех плотностей для сравнения
                self.rn (array[int/float]): массив с данными параметр насыщенности для сравнения
                self.water_permeability (array[int/float]): массив с данными коэффициент водонасыщенности для сравнения

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """
        if self.__check_data(self.kpr, "Кпр_газ", "test data tampering") and \
                self.__check_data(self.kp, "Открытая пористость по жидкости", "test data tampering") and \
                self.__check_data(self.sw_residual, "Кво", "test data tampering") and \
                self.__check_data(self.rp, "Параметр пористости", "test data tampering") and \
                self.__check_data(self.density, "Плотность абсолютно сухого образца", "test data tampering") and \
                self.__check_data(self.water_permeability, "Sw", "test data tampering") and \
                self.__check_data(self.rn, "Параметр насыщенности", "test data tampering"):
            result = True
            wrong_values_kpr = []
            wrong_values_kp = []
            wrong_values_sw_residual = []
            wrong_values_rp = []
            wrong_values_density = []
            wrong_values_rn = []
            wrong_values_water_permeability = []

            unique, indices, counts = np.unique(self.kpr, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result=False
                dup_indices = np.where(self.kpr == dup)[0]
                wrong_values_kpr.extend(dup_indices)

            unique, indices, counts = np.unique(self.kp, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.kp == dup)[0]
                wrong_values_kp.extend(dup_indices)

            unique, indices, counts = np.unique(self.sw_residual, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.sw_residual == dup)[0]
                wrong_values_sw_residual.extend(dup_indices)

            unique, indices, counts = np.unique(self.rp, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.rp == dup)[0]
                wrong_values_rp.extend(dup_indices)

            unique, indices, counts = np.unique(self.density, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.density == dup)[0]
                wrong_values_density.extend(dup_indices)

            unique, indices, counts = np.unique(self.water_permeability, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.water_permeability == dup)[0]
                wrong_values_water_permeability.extend(dup_indices)

            unique, indices, counts = np.unique(self.rn, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.rn == dup)[0]
                wrong_values_rn.extend(dup_indices)

            wrong_values = []
            wrong_values.append(wrong_values_kpr)
            wrong_values.append(wrong_values_kp)
            wrong_values.append(wrong_values_sw_residual)
            wrong_values.append(wrong_values_density)
            wrong_values.append(wrong_values_rp)
            wrong_values.append(wrong_values_rn)
            wrong_values.append(wrong_values_water_permeability)
            self.dict_of_wrong_values["test_data_tampering"] = [{"Кпр_газ(гелий)": wrong_values_kpr,
                                                                 "Открытая пористость по жидкости": wrong_values_kp,
                                                                 "Кво": wrong_values_sw_residual,
                                                                 "Параметр пористости": wrong_values_rp,
                                                                 "Параметр насыщенности": wrong_values_rn,
                                                                 "Плотность абсолютно сухого образца": wrong_values_density,
                                                                 "Sw": wrong_values_water_permeability},
                                                                "схожие значения"]
            if result:
                report_text = result
            else:
                report_text = f" {result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_data_tampering:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_in_surface_and_reservoir_conditions(self, get_report=True):
        """
        Тест выполянется для сравнения коэффициентов пористости, оцененных пластовых
        и атмосферных условиях. Кп в атмосферных условиях всегда больше чем Кп в пластовых условиях.

            Required data:
                Открытая пористость в пластовых условиях; Открытая пористость по жидкости

            Args:
                self.kp_pov (array[int/float]): массив с данными Открытая пористость по жидкости в
                                                поверхностных условиях для проверки
                self.kp_plast (array[int/float]): массив с данными Открытая пористость по жидкости
                                                в пластовых условиях для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """
        if self.__check_data(self.kp_pov, "Открытая пористость по жидкости",
                             "test kp in surface and reservoir conditions") and \
                self.__check_data(self.kp_plast, "Открытая пористость в пластовых условиях",
                                  "test kp in surface and reservoir conditions"):
            result = True
            wrong_values = []
            for i in range(len(self.kp_pov)):
                if abs(self.kp_plast[i] - self.kp_pov[i]) < 0.05:
                    result = False
                    wrong_values.append(i)
            self.dict_of_wrong_values["test_kp_in_surface_and_reservoir_conditions"] = [
                {"Открытая пористость по жидкости": wrong_values,
                 "Открытая пористость в пластовых условиях": wrong_values,
                 }, "кп в пластовых больше или равно кп в атмосферных"]
            if result:
                report_text = f"Test 'coring depths second': {result}."
            else:
                report_text = f"Test 'coring depths second': {result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_open_porosity:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_table_notes(self, get_report=True):
        """
        Тест проводится с целью ранее установленных несоответствий/аномалий
        при лаборатоном анализе керна, указанных в “примечаниях”

           Required data:
                Примечание
            Args:
                self.table (array[int/float]): массив с данными из таблицы с
                примечанием, при наличии ошибки в массиве будет находиться 1,
                при отсутсвии 0

            Returns:
                array[int]: индексы на которых находятся дефекты
                file: запись результата теста для сохранения состояния
        """
        try:
            mask = np.array([isinstance(x, str) for x in self.table])

            # Получаем индексы, где есть строки
            indexes = np.where(mask)[0]
            report_text = f"Test 'table notes': {indexes}\n"
            self.dict_of_wrong_values["test_table_notes"] = [{"Примечание": indexes}, "присутствует неисправность"]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_open_porosity:\n{report_text}\n\n"
            return indexes
        except:
            self.dict_of_wrong_values["test_table_notes"] = [{"Примечание": [0]}, "пустой"]

    def test_quo_and_qno(self, get_report=True):
        """
        Тест оценивает величину суммарную насыщения водой и нефтью,
        которая не должна привышать 100% или 1 в долях
            Required data:
                Sw; So
            Args:
                self.water_permeability (array[int/float]): массив с данными коэффиент водонасыщенности для проверки
                self.kno (array[int/float]): массив с данными коэффициент нефтенащенности для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sw_residual, "Sw", "test quo and qno") and \
                self.__check_data(self.kno, "So", "test quo and qno"):
            converted_sw_residual = []
            converted_kno = []
            result = True
            wrong_values = []
            # перевод из процентов в доли при необходимости
            for val in self.sw_residual:
                if val < 1:
                    converted_sw_residual.append(val * 100)
                else:
                    converted_sw_residual.append(val)
            for val in self.kno:
                if val < 1:
                    converted_kno.append(val * 100)
                else:
                    converted_kno.append(val)

            for i in range(len(converted_sw_residual)):
                if (converted_sw_residual[i] + converted_kno[i]) >= 100:
                    result = False
                    wrong_values.append(i)
            self.dict_of_wrong_values["test_quo_and_qno"] = [{"Sw": wrong_values,
                                                              "So": wrong_values},
                                                             "суммарное насыщение превышает 1 или 100%"]
            if result:
                report_text = f"Test 'coring depths second': {result}."
            else:
                report_text = f"Test 'coring depths second': {result}.Индексы элементов с ошибкой {wrong_values}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_open_porosity:\n{report_text}\n\n"
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_correctness_of_p_sk_kp(self, get_report=True):
        """
            Тест выполянется с целью оценки соответствия плотности/карбонатности
            и пористости образцов в зависимости от направления измерений
            (перпендикулярный или параллельный образец). Разница не должна превышать
            0.5 % абсолютных Например, 7.5 и 7.1 будут схожими замерами по пористости,
            а 7.5 и 6.9 уже непохожими
            Required data:
                 Плотность абсолютно сухого образца; Ск; Открытая пористость по жидкости
            Args:
                self.parallel (array[int/float]): массив с параллельными образцами
                self.parallel_density (array[int/float]): массив с плотностью для параллельных образцов
                self.parallel_carbonate (array[int/float]): массив с карбонатностью для параллельных образцов
                self.parallel_porosity (array[int/float]): массив с пористостью для параллельных образцов
                self.perpendicular (array[int/float]): массив с перпендикулярными образцами
                self.perpendicular_density (array[int/float]): массив с плотностью для перпендикулярных образцов
                self.perpendicular_carbonate (array[int/float]): массив с карбонатностью для перпендикулярных образцов
                self.perpendicular_porosity (array[int/float]): массив с пористостью для перпендикулярных образцов

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        # Создание словарей для хранения значений свойств по индексам
        parallel_properties = {}
        perpendicular_properties = {}
        wrong_values = []
        # Заполнение словаря для первого массива
        for i in range(len(self.parallel)):
            parallel_properties[self.parallel[i][0]] = (
                self.parallel_density[i], self.parallel_carbonate[i], self.parallel_porosity[i])

        # Заполнение словаря для второго массива
        for i in range(len(self.perpendicular)):
            perpendicular_properties[self.perpendicular[i][0]] = (
                self.perpendicular_density[i], self.perpendicular_carbonate[i], self.perpendicular_porosity[i])

        # Сравнение значений свойств по одноименным индексам
        result = True
        wrong_poro = []
        wrong_den = []
        wrong_carb = []
        for key in parallel_properties:
            if key in perpendicular_properties:
                if abs(parallel_properties[key][0][0] - perpendicular_properties[key][0][0]) > \
                        perpendicular_properties[key][0][0] * 0.005:
                    wrong_den.append(parallel_properties[key][0][1])
                    wrong_den.append(perpendicular_properties[key][0][1])
                    result = False

                if abs(parallel_properties[key][1][0] - perpendicular_properties[key][1][0]) > \
                        perpendicular_properties[key][1][0] * 0.005:
                    wrong_carb.append(parallel_properties[key][1][1])
                    wrong_carb.append(perpendicular_properties[key][1][1])
                    result = False

                if abs(parallel_properties[key][2][0] - perpendicular_properties[key][2][0]) > \
                        perpendicular_properties[key][2][0] * 0.005:
                    wrong_poro.append(parallel_properties[key][2][1])
                    wrong_poro.append(perpendicular_properties[key][2][1])
                    result = False
        wrong_values.append(wrong_den)
        wrong_values.append(wrong_carb)
        wrong_values.append(wrong_poro)
        self.dict_of_wrong_values["test correctness of p sk kp"] = [{
            "Открытая пористость по жидкости": wrong_den,
            "Ск": wrong_carb,
            "Плотность абсолютно сухого образца": wrong_poro
        }, "расхождение параметров выше 5%"]

        if result:
            report_text = f"Test 'test_correctness_of_p_sk_kp': {result}."
        else:
            report_text = f"Test 'test_correctness_of_p_sk_kp': {result}.Индексы элементов с ошибкой {wrong_values}."
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_open_porosity:\n{report_text}\n\n"
        return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kno_kpr(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*ln(x)+b при этом a<0, b>0
        Required data:
             Кно; Кпр_газ(гелий)
        Args:
            self.kno (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.kno, "Кно", "test dependence kno kpr") and \
                self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence kno kpr"):

            r2 = self.test_general_dependency_checking(self.kno, self.kpr, "test dependence kno kpr",
                                                       "Коэффициент остаточной нефтенасыщенности",
                                                       "Коэффициента проницаемости")["r2"]

            coefficients = np.polyfit(self.kno, np.exp(self.kpr), 1)
            a, b = coefficients[0], coefficients[1]
            result = True
            if a >= 0 or b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = logarithmic_function_visualization(self.kno, self.kpr, a, b, r2, get_report,
                                                                              "Коэффициент остаточной нефтенасыщенности",
                                                                              "Коэффициента проницаемости",
                                                                              "test_dependence_kno_kpr")

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_dependence_kno_kpr:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_dependence_kno_kpr"] = [{"Кно": wrong_values1,
                                                                     "Кпр_газ(гелий)": wrong_values2},
                                                                    "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kpr_kgo(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*ln(x)+b при этом a<0, b>0
        Required data:
             Sgl; Кпр_газ(гелий)
        Args:
            self.kgo (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.kgo, "Кго", "test dependence kpr kgo") and \
                self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence kpr kgo"):

            r2 = self.test_general_dependency_checking(self.kgo, self.kpr, "test dependence kno kpr",
                                                       "Коэффициент остаточной нефтенасыщенности",
                                                       "Коэффициента проницаемости")["r2"]

            coefficients = np.polyfit(self.kpr, np.exp(self.kgo), 1)
            a, b = coefficients[0], coefficients[1]
            result = True
            if a >= 0 or b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = logarithmic_function_visualization(self.kgo, self.kpr, a, b, r2, get_report,
                                                                              "Cвязанная газонасыщенность",
                                                                              "Коэффициента проницаемости",
                                                                              "test_dependence_kno_kpr")

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_dependence_kno_kpr:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_dependence_kno_kpr"] = [{"Sgl": wrong_values1,
                                                                     "Кпр_газ(гелий)": wrong_values2},
                                                                    "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kpr_knmng(self, get_report=True) -> dict[str, bool | datetime | str]:
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*ln(x)+b при этом a<0, b>0
        Required data:
             Sogcr; Кпр_газ(гелий)
        Args:
            self.knmng (array[int/float]): массив с данными критическая нефтенасыщенность для проверки
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """

        if self.__check_data(self.knmng, "Кнмнг", "test dependence kpr knmng") and \
                self.__check_data(self.kpr, "Кпр_газ(гелий)", "test dependence kpr knmng"):

            r2 = self.test_general_dependency_checking(self.knmng, self.kpr, "test dependence kno kpr",
                                                       "Коэффициент остаточной нефтенасыщенности",
                                                       "Коэффициента проницаемости")["r2"]

            coefficients = np.polyfit(self.kpr, np.exp(self.knmng), 1)
            a, b = coefficients[0], coefficients[1]
            result = True
            if a >= 0 or b <= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = logarithmic_function_visualization(self.knmng, self.kpr, a, b, r2,
                                                                              get_report,
                                                                              "Критическая нефтенасыщенность",
                                                                              "Коэффициента проницаемости",
                                                                              "test_dependence_kpr_knmng")

            report_text = f"{result}."
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_dependence_kno_kpr:\n{report_text}\n\n"
            self.dict_of_wrong_values["test_dependence_kno_kpr"] = [{"Sgl": wrong_values1,
                                                                     "Кпр_газ(гелий)": wrong_values2},
                                                                    "выпадает из линии тренда"]

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def start_tests(self, list_of_tests: list) -> dict[str | Any, dict[Any, Any] | Any]:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """

        results = {}
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            results[method_name] = method(get_report=self.get_report)
        results["wrong_parameters"] = self.dict_of_wrong_values
        return results
