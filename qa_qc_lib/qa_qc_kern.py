import datetime
from datetime import datetime
import numpy as np
from qa_qc_lib.qa_qc_main import QA_QC_main
from scipy.stats import t
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from qa_qc_lib.qa_qc_tools.math_tools import linear_dependence_function
from qa_qc_lib.qa_qc_tools.math_tools import exponential_function
from qa_qc_lib.qa_qc_tools.kern_tools import linear_function_visualization, expon_function_visualization, \
    logarithmic_function_visualization, remap_wrong_values, remove_nan_pairs


class QA_QC_kern(QA_QC_main):
    def __init__(self, pas=np.array([]), note=None, kno=np.array([]), density=None,
                 water_permeability=None, perpendicular=None, perpendicular_density=None,
                 top=None, core_removal_in_meters=None, parallel_carbonate=None, perpendicular_carbonate=None,
                 perpendicular_porosity=None, intervals=None, bottom=None, percent_core_removal=None,
                 outreach_in_meters=None, sw_residual=np.array([]), core_sampling=None, kpr=None, parallel_density=None,
                 parallel_porosity=None, parallel=None, rp=None, pmu=None, rn=None, obplnas=None, poroTBU=None,
                 poroHe=None, porosity_open=None, porosity_kerosine=None, sw=None,
                 parallel_permeability=None, klickenberg_permeability=None, effective_permeability=None, md=None,
                 kgo=None, knmng=None, longitudinal_wave_velocity=None, description_kern=None, kpr_abs_Y=None,
                 fractional_flow_data=None, kpr_abs_Z=None, resistance_of_plastic_water=None, ro_matrix=None,
                 sg=None, constants_of_the_Archie_equation=None, constants_equations_Humble=None,
                 chemical_composition_of_natural_water_and_reservoir_temperature=None,
                 clay_hydrogen_content=None, kpc_phase=None, wettability_wettability_angle=None,
                 cut_off_clay_content=None,
                 cut_off_porosity=None, poissons_coefficient=None, vs=None, DT_matrix=None,
                 kpc_r=None, cut_off_permeability=None, cut_off_water_saturation=None, lithotype=None,
                 capillarometry=None, sgl=None,
                 facies=None, sk=None, vp=None,
                 lithology=None, show=True,
                 sogcr=None,
                 file_name="не указан", r2=0.7) -> None:
        """_summary_

        Args:
            data (str): _description_
        """

        super().__init__()
        self.sogcr = sogcr
        self.sgl = sgl
        self.facies = facies
        self.capillarometry = capillarometry
        self.lithotype = lithotype
        self.cut_off_water_saturation = cut_off_water_saturation
        self.kpc_r = kpc_r
        self.cut_off_permeability = cut_off_permeability
        self.cut_off_porosity = cut_off_porosity
        self.poissons_coefficient = poissons_coefficient
        self.vs = vs
        self.vp = vp
        self.DT_matrix = DT_matrix
        self.sk = sk
        self.cut_off_clay_content = cut_off_clay_content
        self.chemical_composition_of_natural_water_and_reservoir_temperature = chemical_composition_of_natural_water_and_reservoir_temperature
        self.constants_equations_Humble = constants_equations_Humble
        self.wettability_wettability_angle = wettability_wettability_angle
        self.kpc_phase = kpc_phase
        self.clay_hydrogen_content = clay_hydrogen_content
        self.sg = sg
        self.fractional_flow_data = fractional_flow_data
        self.constants_of_the_Archie_equation = constants_of_the_Archie_equation
        self.resistance_of_plastic_water = resistance_of_plastic_water
        self.ro_matrix = ro_matrix
        self.kpr_abs_Y = kpr_abs_Y
        self.description_kern = description_kern
        self.longitudinal_wave_velocity = longitudinal_wave_velocity
        self.kpr_abs_Z = kpr_abs_Z
        self.knmng = knmng
        self.kgo = kgo
        self.md = md
        self.water_permeability = np.array(water_permeability)
        self.effective_permeability = effective_permeability
        self.klickenberg_permeability = klickenberg_permeability
        self.parallel_permeability = parallel_permeability
        self.water_saturation = sw
        self.poroHe = poroHe
        self.residual_water_saturation = sw_residual
        self.porosity_kerosine = porosity_kerosine
        self.obplnas = obplnas
        self.kno = kno
        self.pas = pas
        self.kp_din = []
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
        self.poro_tbu = np.array(poroTBU)
        self.perpendicular_porosity = perpendicular_porosity
        self.perpendicular_carbonate = perpendicular_carbonate
        self.parallel_carbonate = parallel_carbonate
        self.core_removal_in_meters = core_removal_in_meters
        self.top = top
        self.porosity_open = porosity_open
        self.perpendicular_density = perpendicular_density
        self.perpendicular = perpendicular
        self.table = note
        self.file_name = file_name
        self.r2 = r2
        self.dt_now = datetime.now()
        self.dict_of_wrong_values = {}
        self.get_report = show
        self.lithology = lithology
        self.porosity_array = {}
        self.kpr_array = {}
        self.kpr_name_dic = {
            "self.water_permeability": "Газопроницаемость по воде",
            "self.klickenberg_permeability": "Газопроницаемость по Кликенбергу",
            "self.parallel_permeability": "Кпр абс",
            "self.kpr": "Кпр_газ(гелий)",
        }
        self.poro_name_dic = {
            "self.poroHe": "Открытая пористость по газу",
            "self.porosity_open": "Кп откр",
            "self.porosity_kerosine": "Открытая пористость по керосину",
        }
        self.poro_preproccess()
        self.kpr_preproccess()
        self.__parameter_calculation()

    def __check_data(self, array, param_name, test_name):
        """
        Тест предназначен для проверки условия - все элементы массива должны быть числовыми.

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """
        if not isinstance(array, np.ndarray):
            text = f"Не запускался. Причина {param_name}" \
                   f" не является массивом. Входной файл {self.file_name}\n\n"
            self.dict_of_wrong_values[test_name] = [{param_name: [0]}, "не является массивом"]
            report_text = self.generate_report_text(text, 2)
            self.update_report(report_text)
            if self.get_report:
                print('\n' + report_text + self.delimeter)
            return False

        try:
            lent = array.size
            elem = array[0]
        except:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"{timestamp:10} / {test_name}:\n Не запускался. Причина {param_name}" \
                   f" пустой. Входной файл {self.file_name}\n\n"
            self.dict_of_wrong_values[test_name] = [{param_name: [0]}, "пустой"]
            report_text = self.generate_report_text(text, 2)
            self.update_report(report_text)
            if self.get_report:
                print('\n' + report_text + self.delimeter)
            return False
        string_indices = np.where(np.array(list(map(lambda x: isinstance(x, str), array))))
        if string_indices[0].size != 0:
            self.dict_of_wrong_values[test_name] = [{
                param_name: string_indices[0]}, "содержит не числовое значение"]
            text = f"Не запускался. Причина {param_name}" \
                   f"содержит не числовое значение. Входной файл {self.file_name}\n\n"
            report_text = self.generate_report_text(text, 3)
            self.update_report(report_text)
            if self.get_report:
                print('\n' + report_text + self.delimeter)

            return False
        for i in range(array.size):
            if array[i] == np.nan or str(array[i]) == 'nan':
                self.dict_of_wrong_values[test_name] = [{
                    param_name: [i]}, "содержит nan"]
                text = f"Не запускался. Причина {param_name}" \
                       f" содержит nan. Входной файл {self.file_name}\n\n"
                report_text = self.generate_report_text(text, 2)
                self.update_report(report_text)
                if self.get_report:
                    print('\n' + report_text + self.delimeter)

                return False
        return True

    def poro_preproccess(self):
        if not np.array_equal(self.porosity_open, np.array([None])):
            self.porosity_array["self.porosity_open"] = self.porosity_open

        if not np.array_equal(self.poroHe, np.array([None])):
            self.porosity_array["self.poroHe"] = self.poroHe

        if not np.array_equal(self.porosity_kerosine, np.array([None])):
            self.porosity_array["self.porosity_kerosine"] = self.porosity_kerosine

    def kpr_preproccess(self):
        if not np.array_equal(self.klickenberg_permeability, np.array([None])):
            self.kpr_array["self.klickenberg_permeability"] = self.klickenberg_permeability

        if not np.array_equal(self.kpr, np.array([None])):
            self.kpr_array["self.kpr"] = self.kpr

        if not np.array_equal(self.parallel_permeability, np.array([None])):
            self.kpr_array["self.parallel_permeability"] = self.parallel_permeability

        if not np.array_equal(self.water_permeability, np.array([None])):
            self.kpr_array["self.water_permeability"] = self.water_permeability

    def __water_saturation(self, array):
        wrong_values = []
        result = True
        for i in range(len(array)):
            if array[i] < 0 or array[i] > 1:
                result = False
                wrong_values.append(i)
        return result, wrong_values

    def __vp_vs(self, array):
        wrong_values = []
        result = True
        for i in range(len(array)):
            if array[i] < 0.3 or array[i] > 10:
                result = False
                wrong_values.append(i)
        return result, wrong_values

    def __test_porosity(self, array):
        parts_bound_1, parts_bound_2 = 0, 1
        percentages_bound = 1
        parts = np.where((array > parts_bound_1) & (array < parts_bound_2))[0]
        percentages = np.where((array > percentages_bound))[0]
        if len(parts) >= len(percentages):
            lower_bound_1, upper_bound_1 = 0, 0.476
            wrong_indices_1 = np.where((array < lower_bound_1) | (array > upper_bound_1))[0]
            return len(wrong_indices_1) == 0, wrong_indices_1.tolist()
        else:
            lower_bound_2, upper_bound_2 = 0, 47.6
            wrong_indices_2 = np.where((array < lower_bound_2) | (array > upper_bound_2))[0]
            return len(wrong_indices_2) == 0, wrong_indices_2.tolist()

    def __permeability(self, array):
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
            report_text = self.generate_report_text("Не удалось вычислить кп эф", 2)
            self.update_report(report_text)
            self.kp_ef = np.array([])
        try:
            if self.porosity_open is not None and self.sw_residual is not None and self.kno is not None:
                for i in range(len(self.porosity_open)):
                    self.kp_din.append(self.porosity_open[i] * (1 - self.sw_residual[i] - self.kno[i]))
                self.kp_din = np.array(self.kp_din)
        except:
            report_text = self.generate_report_text("Не удалось вычислить кп дин", 2)
            self.update_report(report_text)
            self.kp_din = np.array([])
        try:
            if self.pmu is None:
                self.pmu = []
                if self.pas is not None and self.porosity_open is not None:
                    for i in range(len(self.pas)):
                        self.pmu.append(self.pas[i] + (self.porosity_open[i] * 1))
                    self.pmu = np.array(self.pmu)
        except:
            report_text = self.generate_report_text("Не удалось вычислить pmu", 2)
            self.update_report(report_text)
            self.pmu = np.array([])

    """
    Тесты первого порядка 
    """

    def test_open_porosity(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп откр

        Args:
            self.porosity_open (array[int/float]): массив с открытой пористостью в атмосферных условия для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.porosity_open, "Кп откр", "test_open_porosity"):
            result, wrong_values = self.__test_porosity(self.porosity_open)
            if not result:
                text = f"Данные с индексом {wrong_values} лежат не в " \
                       f"интервале от 0 до 47,6."
                report_text = self.generate_report_text(text, 0)
            else:
                report_text = self.generate_report_text("Все данные лежат в интервале от 0 до 47.6", 1)

            self.dict_of_wrong_values["test open porosity"] = [{
                "Кп откр": wrong_values,
            }, "не лежит в интервале от 0 до 47,6"]

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_vp(self, get_report=True):
        """
        Тест предназначен для проверки нахождения значения в физичном диапазоне

        Required data:
            Скорость продольной волны(Vp)

        Args:
            self.vp (array[int/float]): массив с cкорость продольной волны(Vp) для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result, wrong_values = self.__vp_vs(self.vp)
        if not result:
            text = f"Данные с индексом {wrong_values} лежат не в " \
                   f"интервале от 0.3 до 10 км\c"
            report_text = self.generate_report_text(text, 0)
        else:
            report_text = self.generate_report_text("Все данные лежат в интервале от 0.3 до 10 км\c", 1)

        self.dict_of_wrong_values["test vp"] = [{
            "Скорость продольной волны(Vp)": wrong_values,
        }, "не лежит в интервале от 0.3 до 10 км\c"]

        self.update_report(report_text)
        if get_report:
            print('\n' + report_text + self.delimeter)
        return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_vs(self, get_report=True):
        """
        Тест предназначен для проверки нахождения значения в физичном диапазоне

        Required data:
            Скорость поперечной волны (Vs)
        Args:
            self.vs (array[int/float]): массив с cкорость продольной волны(Vp) для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result, wrong_values = self.__vp_vs(self.vs)
        if not result:
            text = f"Данные с индексом {wrong_values} лежат не в " \
                   f"интервале от 0.3 до 10 км\c"
            report_text = self.generate_report_text(text, 0)
        else:
            report_text = self.generate_report_text("Все данные лежат в интервале от 0.3 до 10 км\c", 1)

        self.dict_of_wrong_values["test vp"] = [{
            "Скорость поперечной волны (Vs)": wrong_values,
        }, "не лежит в интервале от 0.3 до 10 км\c"]

        self.update_report(report_text)
        if get_report:
            print('\n' + report_text + self.delimeter)
        return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_HE(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Открытая пористость по газу

        Args:
            self.poroHe (array[int/float]): массив с открытой пористостью по гелию для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.poroHe, "Открытая пористость по газу", "test porosity HE"):
            result, wrong_values = self.__test_porosity(self.poroHe)
            if not result:
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в интервале от 0 "
                                                        "до 47,6.", 0)
            else:
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в интервале от 0 "
                                                        "до 47,6.", 1)
            self.dict_of_wrong_values["test open porosity HE"] = [{"Открытая пористость по газу": wrong_values},
                                                                  "не лежит в интервале от 0 до 47,6"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_TBU(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп откр TBU

        Args:
            self.poro_tbu (array[int/float]): массив с открытой пористостью в пластовых условиях для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.poro_tbu, "Кп откр TBU", "test porosity TBU"):
            result, wrong_values = self.__test_porosity(self.poro_tbu)
            if not result:
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в интервале от 0 "
                                                        "до 47,6.", 0)
            else:
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в "
                                                        f"интервале от 0 до 47,6.", 1)
            self.dict_of_wrong_values["test open porosity TBU"] = [{
                "Кп откр TBU": wrong_values}, "не лежит в интервале от 0 до 47,6"]

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
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
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в интервале от 0 "
                                                        "до 47,6.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 47.6", 1)

            self.dict_of_wrong_values["test porosity kerosine"] = [{"Открытая пористость по керосину": wrong_values},
                                                                   "не лежит в интервале от 0 до 47,6"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_porosity_effective(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп эфф
        Args:
            self.kp_ef (array[int/float]): массив с эффективной пористостью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kp_ef, "porosity effective", "test porosity effective"):
            result, wrong_values = self.__test_porosity(self.kp_ef)
            if not result:
                report_text = self.generate_report_text(f"Данные с индексом {wrong_values} лежат не в интервале от 0 "
                                                        "до 47,6.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 47.6", 1)

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_residual_water_saturation(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Кво
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
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test residual water saturation"] = [{"Кво": wrong_values},
                                                                           "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_sg(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sg
        Args:
            self.sg (array[int/float]): массив с остаточной водонасыщенностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sg, "Sg",
                             "test sg"):
            result, wrong_values = self.__water_saturation(self.sg)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test sg"] = [{"Sg": wrong_values}, "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_sgl(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sgl
        Args:
            self.sgl (array[int/float]): массив с остаточной водонасыщенностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sgl, "Sgl",
                             "test sgl"):
            result, wrong_values = self.__water_saturation(self.sgl)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test sgl"] = [{"+": wrong_values}, "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_so(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            So
        Args:
            self.so (array[int/float]): массив с остаточной водонасыщенностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.kno, "+",
                             "test so"):
            result, wrong_values = self.__water_saturation(self.kno)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test so"] = [{"So": wrong_values}, "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_sogcr(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            +
        Args:
            self.+ (array[int/float]): массив с остаточной водонасыщенностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sogcr, "+",
                             "test sogcr"):
            result, wrong_values = self.__water_saturation(self.sogcr)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test sogcr"] = [{"Sogcr": wrong_values}, "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_carbonation(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Карбонатность

        Args:
            self.sk (array[int/float]): массив с карбонатностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.sk, "Карбонатность",
                             "test carbonation"):
            result, wrong_values = self.__water_saturation(self.sk)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test carbonation"] = [{"Карбонатность": wrong_values},
                                                             "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_core_removal(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

        Required data:
            Глубина отбора

        Args:
            self.core_removal_in_meters (array[int/float]): массив с газопроницаемостью
                                                               параллельно напластованию для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.core_removal_in_meters, "Глубина отбора",
                             "test core removal"):
            result, wrong_values = self.__permeability(self.core_removal_in_meters)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} меньше или равны 0", 0)
            else:
                report_text = self.generate_report_text(
                    f"Все данные больше 0", 1)
            self.dict_of_wrong_values["test core removal"] = [
                {"Глубина отбора": wrong_values}, "значение меньше 0"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_water_saturation(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sw

        Args:
            self.water_saturation (array[int/float]): массив с водонасыщенностью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.water_saturation, "Sw", "test water saturation"):
            result, wrong_values = self.__water_saturation(self.water_saturation)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} лежат не в интервале от 0 до 1.", 0)
            else:
                report_text = self.generate_report_text(f"Все данные лежат в интервале от 0 до 1.", 1)
            self.dict_of_wrong_values["test water saturation"] = [{"Sw": wrong_values},
                                                                  "не соответсвует интервалу от 0 до 1"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_parallel_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

        Required data:
            Кпр абс

        Args:
            self.parallel_permeability (array[int/float]): массив с газопроницаемостью
                                                               параллельно напластованию для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.parallel_permeability, "Кпр абс",
                             "test parallel permeability"):
            result, wrong_values = self.__permeability(self.parallel_permeability)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} меньше или равны 0", 0)
            else:
                report_text = self.generate_report_text(
                    f"Все данные больше 0", 1)
            self.dict_of_wrong_values["test parallel permeability"] = [
                {"Кпр абс": wrong_values}, "значение меньше 0"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_klickenberg_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

        Required data:
            Газопроницаемость по Кликенбергу

        Args:
            self.klickenberg_permeability (array[int/float]): массив с газопроницаемостью
                                                              с поправкой по Кликенбергу для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.klickenberg_permeability, "Газопроницаемость по Кликенбергу",
                             "test klickenberg permeability"):

            result, wrong_values = self.__permeability(self.klickenberg_permeability)

            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} меньше или равны 0", 0)
            else:
                report_text = self.generate_report_text(
                    f"Все данные больше 0", 1)

            self.dict_of_wrong_values["test klickenberg permeability"] = [
                {"Газопроницаемость по Кликенбергу": wrong_values}, "значение меньше 0"]

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_effective_permeability(self, get_report=True):
        """
        Тест предназначен для проверки физичности данных.
        Значение должно быть больше 0

        Required data:
            Эффективная проницаемость

        Args:
            self.effective_permeability (array[int/float]): массив с эффективной проницаемостью для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.effective_permeability, "Эффективная проницаемость", "test_effective_permeability"):
            result, wrong_values = self.__permeability(self.effective_permeability)
            if not result:
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} меньше или равны 0", 0)
            else:
                report_text = self.generate_report_text(
                    f"Все данные больше 0", 1)
            self.dict_of_wrong_values["test effective permeability"] = [{"Эффективная проницаемость": wrong_values},
                                                                        "значение меньше 0"]

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
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
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} меньше или равны 0", 0)
            else:
                report_text = self.generate_report_text(
                    f"Все данные больше 0", 1)

            self.dict_of_wrong_values["test water permeability"] = [{"Газопроницаемость по воде": wrong_values},
                                                                    "значение меньше 0"]
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name,
                    "date": self.dt_now}

    def test_monotony(self, get_report=True):
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
                report_text = self.generate_report_text(
                    f"Данные с индексом {wrong_values} нарушают монотонность", 0)

            else:
                report_text = self.generate_report_text(
                    f"Все данные больше монотонно возрастают", 1)

            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name,
                    "date": self.dt_now}

    """
        Тесты второго порядка
    """

    def test_quo_kp_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кво; Кп откр

        Args:
            self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.porosity_open (array[int/float]): массив с данными Кп откр для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_sw_residual, index_dic = remove_nan_pairs(porosity_open, self.sw_residual)
            if self.__check_data(parsed_sw_residual, "Кво",
                                 f"test quo {poro_name} dependence") and \
                    self.__check_data(parsed_porosity_open, f"{poro_name}", f"test quo {poro_name} dependence"):

                r2 = \
                    self.test_general_dependency_checking(parsed_sw_residual, parsed_porosity_open,
                                                          f"test quo {poro_name} dependence",
                                                          "Кво",
                                                          f"{poro_name}")["r2"]
                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_sw_residual)
                if a >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_sw_residual,
                                                                             a, b,
                                                                             r2,
                                                                             get_report, f"{poro_name}",
                                                                             "Кво",
                                                                             f"test_quo_{poro_name}_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                self.dict_of_wrong_values[f"test_quo_{poro_name}_dependence"] = [{"Кво": wrong_values1,
                                                                                  f"{poro_name}": wrong_values2
                                                                                  }, "выпадает из линии тренда"]
                result_array.append(result)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)

        return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_density_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции y=a*x+b, при этом a<0

        Required data:
            Кп откр; Плотность абсолютно сухого образца
        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.density (array[int/float]): массив с данными плотности для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_density, index_dic = remove_nan_pairs(porosity_open, self.density)
            if self.__check_data(parsed_porosity_open, f"{poro_name}", f"test {poro_name} density dependence") and \
                    self.__check_data(parsed_density, "Плотность абсолютно сухого образца",
                                      f"test {poro_name} density dependence"):

                r2 = self.test_general_dependency_checking(parsed_porosity_open, parsed_density,
                                                           f"test {poro_name} density dependence",
                                                           f"{poro_name}",
                                                           "Плотности")["r2"]

                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_density)
                if a >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_density, a, b,
                                                                             r2,
                                                                             get_report,
                                                                             f"{poro_name}", "Плотности",
                                                                             f"{poro_name}_density_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                self.dict_of_wrong_values[f"test {poro_name} density dependence"] = [
                    {f"{poro_name}": wrong_values1,
                     "Плотность абсолютно сухого образца": wrong_values2,
                     }, "выпадает из линии тренда"]
                result_array.append(result)
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)

                if get_report:
                    print(report_text)

        return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kgo_dependence(self, get_report=True):
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*x+b, при этом a<0

        Required data:
            Кп откр; Cвязанная газонасыщенность

        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.kgo (array[int/float]): массив с данными связанная газонасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_kgo, index_dic = remove_nan_pairs(porosity_open, self.kgo)
            if self.__check_data(parsed_porosity_open, f"{poro_name}", f"test_{poro_name}_kgo_dependence") and \
                    self.__check_data(parsed_kgo, "Cвязанная газонасыщенность", f"test_{poro_name}_kgo_dependence"):

                r2 = \
                    self.test_general_dependency_checking(parsed_porosity_open, parsed_kgo,
                                                          f"test_{poro_name}_kgo_dependence",
                                                          f"{poro_name}",
                                                          "Cвязанная газонасыщенность")["r2"]

                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_kgo)
                if a >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_kgo, a, b, r2,
                                                                             get_report,
                                                                             f"{poro_name}",
                                                                             "Cвязанная газонасыщенность",
                                                                             f"test_{poro_name}_kgo_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                self.dict_of_wrong_values[f"test_{poro_name}_kgo_dependence"] = [
                    {f"{poro_name}": wrong_values1,
                     "Cвязанная газонасыщенность": wrong_values2,
                     }, "выпадает из линии тренда"]
                result_array.append(result)

            return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_knmng_dependence(self, get_report=True):
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных) и текущей.  Характерной
        зависимостью является линейная по функции y=a*x+b, при этом a<0

        Required data:
            Кп откр; Критическая нефтенасыщенность

        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.knmng (array[int/float]): массив с данными критическая нефтенасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_knmng, index_dic = remove_nan_pairs(porosity_open, self.knmng)
            if self.__check_data(parsed_porosity_open, f"{poro_name}", f"test_{poro_name}_knmng_dependence") and \
                    self.__check_data(parsed_knmng, "Критическая нефтенасыщенность",
                                      f"test_{poro_name}_knmng_dependence"):

                r2 = self.test_general_dependency_checking(parsed_porosity_open, parsed_knmng,
                                                           f"test_{poro_name}_knmng_dependence",
                                                           f"{poro_name}",
                                                           "Критическая нефтенасыщенность")["r2"]

                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_knmng)
                if a >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_knmng, a, b,
                                                                             r2,
                                                                             get_report,
                                                                             f"{poro_name}",
                                                                             "Cвязанная газонасыщенность",
                                                                             f"test_{poro_name}_knmng_dependence")

                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)

                self.dict_of_wrong_values[f"test_{poro_name}_knmng_dependence"] = [
                    {f"{poro_name}": wrong_values1,
                     "Критическая нефтенасыщенность": wrong_values2,
                     }, "выпадает из линии тренда"]

            return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kno_dependence(self, get_report=True):
        """
        Тест применяется для сравнения двух аппроксимаций: характерной
        (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*x+b, при этом a<0

        Required data:
            Коэффициента пористости; Коэффициент остаточной нефтенасыщенности

        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.kno (array[int/float]): массив с данными связанная газонасыщенность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_kno, index_dic = remove_nan_pairs(porosity_open, self.kno)
            if self.__check_data(parsed_porosity_open, f"{poro_name}", f"test_{poro_name}_kno_dependence") and \
                    self.__check_data(parsed_kno, "Коэффициент остаточной нефтенасыщенности",
                                      f"test_{poro_name}_kno_dependence"):

                r2 = \
                    self.test_general_dependency_checking(parsed_porosity_open, parsed_kno,
                                                          f"test_{poro_name}_kno_dependence",
                                                          f"{poro_name}",
                                                          "Коэффициент остаточной нефтенасыщенности")["r2"]

                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_kno)
                if a >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_kno, a, b, r2,
                                                                             get_report,
                                                                             f"{poro_name}",
                                                                             "Коэффициент остаточной нефтенасыщенности",
                                                                             f"test_{poro_name}_kno_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                self.dict_of_wrong_values[f"test_{poro_name}_kno_dependence"] = [
                    {f"{poro_name}": wrong_values1,
                     "Коэффициент остаточной нефтенасыщенности": wrong_values2,
                     }, "выпадает из линии тренда"]
                result_array.append(result)

            return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_sw_residual_kp_din_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции y=a*x+b, при этом a<0

        Required data:
            Кво; Коэффициент остаточной нефтенасыщенности; Кп откр

        Args:
            self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kp_din (array[int/float]): массив с данными коэффициент динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        parsed_sw_residual, parsed_kp_din, index_dic = remove_nan_pairs(self.sw_residual, self.kp_din)
        if self.__check_data(parsed_sw_residual, "Кво", "test sw_residual kp din dependence") and \
                self.__check_data(parsed_kp_din, "kp din", "test sw_residual kp din dependence"):

            r2 = self.test_general_dependency_checking(parsed_sw_residual, parsed_kp_din,
                                                       "test sw_residual kp din dependence",
                                                       "Коэффициента остаточной водонасыщенности",
                                                       "Коэффициента динамической пористости")["r2"]
            result = True
            a, b = linear_dependence_function(parsed_kp_din, parsed_sw_residual)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values1, wrong_values2 = linear_function_visualization(parsed_sw_residual, parsed_kp_din, a, b, r2,
                                                                         get_report,
                                                                         "Коэффициент остаточной водонасыщенности",
                                                                         "Коэффициент динамической пористости",
                                                                         "test_sw_residual_kp_din_dependence")
            wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
            wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

            if result:
                report_text = self.generate_report_text(
                    f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
            else:
                report_text = self.generate_report_text(
                    f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            self.dict_of_wrong_values["test_sw_residual_kp_din_dependence"] = [{"Кво": wrong_values1},
                                                                               "выпадает из линии тренда"]
            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_obblnas_kp_dependence(self, get_report=True):
        """
        Тест предназначен для проверки физичности
        взаимосвязи двух кроссплотов - Обплнас-Кп и
        Минпл-Кп. Пусть первый аппроксимируется
        линией тренда y=a1*x+b1, а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Минералогическая плотность; Кп откр; Объемная плотность
        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.obblnas (array[int/float]): массив с данными объемная плотность для проверки
            self.pmu (array[int/float]): массив с данными минералогическая плотность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            if self.__check_data(self.pmu, "Минералогическая плотность", f"test obblnas {poro_name} dependence") and \
                    self.__check_data(porosity_open, f"{poro_name}", f"test obblnas {poro_name} dependence") and \
                    self.__check_data(self.obplnas, "Объемная плотность", f"test obblnas {poro_name} dependence"):

                r2_pmu = self.test_general_dependency_checking(self.pmu, porosity_open,
                                                               f"test obblnas {poro_name} dependence",
                                                               "Минералогической плотность",
                                                               f"{poro_name}")["r2"]
                r2_obp = \
                    self.test_general_dependency_checking(self.obplnas, porosity_open,
                                                          f"test obblnas {poro_name} dependence",
                                                          "Объемной плотность",
                                                          f"{poro_name}")["r2"]

                coeffs1 = np.polyfit(porosity_open, self.pmu, 1)
                a1, b1 = coeffs1[0], coeffs1[1]
                trend_line1 = np.polyval(coeffs1, self.pmu)

                coeffs2 = np.polyfit(porosity_open, self.obplnas, 1)
                a2, b2 = coeffs2[0], coeffs2[1]
                trend_line2 = np.polyval(coeffs2, self.obplnas)

                result = True
                if a1 >= a2 or r2_obp < 0.7 or r2_pmu < 0.7:
                    result = False

                wrong_values1 = []
                wrong_values2 = []
                for i in range(len(self.obplnas)):
                    if self.obplnas[i] < a1 * porosity_open[i] + b1:
                        wrong_values1.append(self.obplnas[i])
                        wrong_values2.append(porosity_open[i])
                self.dict_of_wrong_values[f"test_obblnas_{poro_name}_dependence"] = [
                    {"Объемная плотность": wrong_values1,
                     f"{poro_name}": wrong_values2},
                    "выпадает из линии тренда"]

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                y_pred = a2 * porosity_open + b2

                # Окрашиваем точки, которые не соответствуют линии тренда, в красный
                for obplnas_val, kp_val, pred_val in zip(self.obplnas, porosity_open, y_pred):
                    if obplnas_val + (pred_val * 0.1) < pred_val:
                        plt.scatter(kp_val, obplnas_val, color='g')

                plt.title(f"test obblnas {poro_name} dependence")
                plt.scatter(porosity_open, self.obplnas, color='red', label='Обплнас-Кп')
                plt.scatter(porosity_open, self.pmu, color='blue', label='Минпл-Кп')
                plt.plot(porosity_open, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
                plt.plot(porosity_open, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
                plt.xlabel(f'{poro_name}')
                plt.ylabel('obplnas')
                plt.legend()
                plt.grid(True)
                equation = f'y = {a1:.2f}x + {b1:.2f}, r2_pmu={r2_pmu:.2f}'
                plt.text(np.mean(porosity_open), np.min(self.pmu) + 2, equation, ha='center', va='bottom')
                equation = f'y = {a2:.2f}x + {b2:.2f}, r2_obp={r2_obp:.2f}'
                plt.text(np.mean(porosity_open), np.min(self.obplnas), equation, ha='center', va='bottom')
                plt.savefig(f"report\\test_obblnas_{poro_name}_dependence.png")
                if get_report:
                    plt.show()
                    print(report_text)

            return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_pmu_kp_dependence(self, get_report=True):
        """
        Тест предназначен для проверки физичности
        взаимосвязи двух кроссплотов - Обплнас-Кп и
        Минпл-Кп. Пусть первый аппроксимируется
        линией тренда y=a1*x+b1, а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Минералогическая плотность; Кп откр; Объемная плотность
        Args:
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки
            self.obblnas (array[int/float]): массив с данными объемная плотность для проверки
            self.pmu (array[int/float]): массив с данными минералогическая плотность для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            if self.__check_data(self.pmu, "Минералогическая плотность", f"test pmu {poro_name} dependence") and \
                    self.__check_data(porosity_open, f"{poro_name}", f"test pmu {poro_name} dependence") and \
                    self.__check_data(self.obplnas, "Объемная плотность", f"test pmu {poro_name} dependence"):

                r2_pmu = \
                    self.test_general_dependency_checking(self.pmu, porosity_open,
                                                          f"test pmu {poro_name} dependence",
                                                          "Минералогической плотность",
                                                          f"{poro_name}")["r2"]
                r2_obp = self.test_general_dependency_checking(self.obplnas, porosity_open,
                                                               f"test pmu {poro_name} dependence",
                                                               "Объемной плотность",
                                                               f"{poro_name}")["r2"]

                coeffs1 = np.polyfit(porosity_open, self.obplnas, 1)
                a1, b1 = coeffs1[0], coeffs1[1]
                trend_line1 = np.polyval(coeffs1, self.obplnas)

                coeffs2 = np.polyfit(porosity_open, self.pmu, 1)
                a2, b2 = coeffs2[0], coeffs2[1]
                trend_line2 = np.polyval(coeffs2, self.pmu)

                result = True
                if a1 <= a2 or r2_obp < 0.7 or r2_pmu < 0.7:
                    result = False

                wrong_values1 = []
                wrong_values2 = []
                for i in range(len(self.pmu)):
                    if self.pmu[i] < a2 * porosity_open[i] + b2:
                        wrong_values1.append(self.pmu[i])
                        wrong_values2.append(porosity_open[i])

                self.dict_of_wrong_values[f"test_pmu_{poro_name}_dependence"] = [
                    {"Минералогическая плотность": wrong_values1,
                     f"{poro_name}": wrong_values2},
                    "выпадает из линии тренда"]
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                plt.title(f"test pmu {poro_name} dependence")
                plt.scatter(self.obplnas, porosity_open, color='red', label='Обплнас-Кп')
                plt.scatter(self.pmu, porosity_open, color='blue', label='Минпл-Кп')
                y_pred = a2 * porosity_open + b2

                # Окрашиваем точки, которые не соответствуют линии тренда, в красный
                for pmu_val, kp_val, pred_val in zip(self.pmu, porosity_open, y_pred):
                    if pmu_val + (pred_val * 0.1) < pred_val:
                        plt.scatter(kp_val, pmu_val, color='g')

                plt.plot(porosity_open, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
                plt.plot(porosity_open, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
                plt.xlabel(f'{poro_name}')
                plt.ylabel('pmu')
                plt.legend()
                plt.grid(True)
                equation = f'y = {a1:.2f}x + {b1:.2f}, r2_obpl = {r2_obp}'
                plt.text(np.mean(self.pmu), np.min(porosity_open) + 2, equation, ha='center', va='bottom')
                equation = f'y = {a2:.2f}x + {b2:.2f}, r2_pmu ={r2_pmu}'
                plt.text(np.mean(self.obplnas), np.min(porosity_open), equation, ha='center', va='bottom')
                plt.savefig(f"report\\test_pmu_{poro_name}_dependence.png")

                if get_report:
                    plt.show()
                    print(report_text)
                result_array.append(result)

        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_kp_ef_kpdin_dependence(self, get_report=True):
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
            if result:
                report_text = self.generate_report_text(
                    f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
            else:
                report_text = self.generate_report_text(
                    f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_ef_kp_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость
        линейная по функции y=a*x+b, при этом a>0, b<0

        Required data:
            Открытая пористость; Кво; Кп откр;
        Args:
            self.kp_ef (array[int/float]): массив с данными коэффициента эффективной пористости для проверки
            self.porosity_open (array[int/float]): массив с данными коэффициента пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_kp_ef, parsed_porosity_open, index_dic = remove_nan_pairs(self.kp_ef, porosity_open)
            if self.__check_data(parsed_kp_ef, "kp ef", f"test kp ef {poro_name} dependence") \
                    and self.__check_data(parsed_porosity_open, f"{poro_name}", f"test kp ef {poro_name} dependence"):

                r2 = self.test_general_dependency_checking(parsed_kp_ef, parsed_porosity_open,
                                                           f"test kp ef {poro_name} dependence",
                                                           "Коэффициента эффективной пористости",
                                                           f"{poro_name}")["r2"]
                result = True
                a, b = linear_dependence_function(parsed_porosity_open, parsed_kp_ef)
                if a <= 0 or b >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_kp_ef, a, b,
                                                                             r2,
                                                                             get_report,
                                                                             f"{poro_name}",
                                                                             "Коэффициент эффективной пористости",
                                                                             f"test_kp_ef_{poro_name}_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                self.dict_of_wrong_values["test_kp_ef_kp_dependence"] = [{
                    "Кп откр": wrong_values2
                }, "выпадает из линии тренда"]
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                result_array.append(result)

            return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def test_kp_kp_din_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость
        линейная по функции y=a*x+b, при этом a>0, b<0

        Required data:
            Открытая пористость; Кво; So; Кп откр

        Args:
            self.porosity_open (array[int/float]): массив с данными Кп откр для проверки
            self.kp_din (array[int/float]): массив с данными коэффициент динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_porosity_open, parsed_kp_din, index_dic = remove_nan_pairs(porosity_open, self.kp_din)
            if self.__check_data(parsed_porosity_open, f"{poro_name}", f"test {poro_name} kp din dependence") \
                    and self.__check_data(parsed_kp_din, "kp din", f"test {poro_name} kp din dependence"):
                r2 = self.test_general_dependency_checking(parsed_porosity_open, parsed_kp_din,
                                                           f"test {poro_name} kp din dependence",
                                                           f"{poro_name}",
                                                           "Коэффициента динамической пористости")["r2"]
                a, b = linear_dependence_function(parsed_porosity_open, parsed_kp_din)
                result = True
                if a <= 0 or b >= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = linear_function_visualization(parsed_porosity_open, parsed_kp_din, a, b,
                                                                             r2,
                                                                             get_report,
                                                                             f"{poro_name}",
                                                                             "Коэффициент динамической пористости",
                                                                             f"test_{poro_name}_kp_din_dependence")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                self.dict_of_wrong_values[f"test_{poro_name}c_kp_din_dependence"] = [{f"{poro_name}": wrong_values1,
                                                                                      }, "выпадает из линии тренда"]
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                result_array.append(result)

            return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_dependence_kpr_kp(self, get_report=False):
        """
        Тест предназначен для оценки соответствия
        типовой для данного кроссплота и полученной
        аппроксимации. В данном случае зависимость по
        функции y=a*exp(b*x) при этом b>0

        Required data:
            Кпр абс; Кп откр

        Args:
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки
            self.porosity_open (array[int/float]): массив с данными Кп откр для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for kpr_elem in self.kpr_array:
            kpr = self.kpr_array[kpr_elem]
            kpr_name = self.kpr_name_dic[kpr_elem]
            for poro in self.porosity_array:
                porosity_open = self.porosity_array[poro]
                poro_name = self.poro_name_dic[poro]
                parsed_kpr, parsed_porosity_open, index_dic = remove_nan_pairs(kpr, porosity_open)
                if self.__check_data(parsed_kpr, f"{kpr_name}", f"test dependence {kpr_name} {poro_name}") and \
                        self.__check_data(parsed_porosity_open, f"{poro_name}",
                                          f"test dependence {kpr_name} {poro_name}"):
                    r2 = \
                        self.test_general_dependency_checking(parsed_kpr, parsed_porosity_open,
                                                              f"test dependence {kpr_name} {poro_name}",
                                                              f"{kpr_name}",
                                                              f"{poro_name}")["r2"]

                    result = True
                    a, b = exponential_function(parsed_porosity_open, parsed_kpr)
                    if b <= 0 or r2 < 0.7:
                        result = False

                    wrong_values1, wrong_values2 = expon_function_visualization(parsed_porosity_open.astype(float),
                                                                                parsed_kpr.astype(float), a, b, r2,
                                                                                get_report,
                                                                                f"{poro_name}",
                                                                                f"{kpr_name}",
                                                                                f"test_dependence_{kpr_name}_{poro_name}")
                    wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                    wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                    self.dict_of_wrong_values[f"test_dependence_{kpr_name}_{poro_name}"] = [
                        {f"{kpr_name}": wrong_values1,
                         f"{poro_name}": wrong_values2},
                        "выпадает из линии тренда"]
                    if result:
                        report_text = self.generate_report_text(
                            f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                    else:
                        report_text = self.generate_report_text(
                            f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                    self.update_report(report_text)
                    if get_report:
                        print(report_text)
                    result_array.append(result)
        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_dependence_kpr_kp_din(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой для
        данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*exp(b*x) при этом b>0

        Required data:
            Открытая пористость; Кво; So; Кпр абс

        Args:
            self.kpr (array[int/float]): массив с данными коэффициента проницаемости для проверки
            self.kp_din (array[int/float]): массив с данными коэффициента динамической пористости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for kpr_elem in self.kpr_array:
            kpr = self.kpr_array[kpr_elem]
            kpr_name = self.kpr_name_dic[kpr_elem]
            parsed_kpr, parsed_kp_din, index_dic = remove_nan_pairs(kpr, self.kp_din)
            if self.__check_data(parsed_kpr, f"{kpr_name}", f"test dependence {kpr_name} kp din") and \
                    self.__check_data(parsed_kp_din, "Kp_din", f"test dependence {kpr_name} kp din"):

                r2 = self.test_general_dependency_checking(self.kpr, self.kp_din, f"test dependence {kpr_name} kp din",
                                                           f"{kpr_name}",
                                                           "Коэффициента динамической пористости")["r2"]
                result = True
                a, b = exponential_function(parsed_kp_din, parsed_kpr)
                if b <= 0 or r2 < 0.7:
                    result = False

                result_array.append(result)

                wrong_values1, wrong_values2 = expon_function_visualization(parsed_kpr, parsed_kp_din, a, b, r2,
                                                                            get_report,
                                                                            f"{kpr_name}",
                                                                            "Коэффициент динамической пористости",
                                                                            f"test_dependence_{kpr_name}_kp_din")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                self.dict_of_wrong_values[f"test_dependence_{kpr_name}_kp_din"] = [{f"{kpr_name}": wrong_values1,
                                                                                    }, "выпадает из линии тренда"]
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)

        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_dependence_sw_residual_kpr(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a*ln(x)+b при этом a>0
        Required data:
            Кво; Кпр абс

        Args:
            self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            self.kpr (array[int/float]): массив с данными коэффициент проницаемости для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for kpr_elem in self.kpr_array:
            kpr = self.kpr_array[kpr_elem]
            kpr_name = self.kpr_name_dic[kpr_elem]
            parsed_sw_residual, parsed_kpr, index_dic = remove_nan_pairs(self.sw_residual, kpr)
            if self.__check_data(parsed_sw_residual, "Кво", f"test dependence sw_residual {kpr_name}") and \
                    self.__check_data(parsed_kpr, f"{kpr_name}", f"test dependence sw_residual {kpr_name}"):

                r2 = self.test_general_dependency_checking(parsed_sw_residual, parsed_kpr,
                                                           f"test dependence sw_residual {kpr_name}",
                                                           "Коэффициента остаточной водонасыщенности",
                                                           f"{kpr_name}")["r2"]
                coefficients = np.polyfit(parsed_kpr, np.exp(parsed_sw_residual), 1)
                a, b = coefficients[0], coefficients[1]
                result = True
                wrong_values1, wrong_values2 = logarithmic_function_visualization(parsed_kpr, parsed_sw_residual, a, b,
                                                                                  r2,
                                                                                  get_report, f"{kpr_name}",
                                                                                  "Коэффициент остаточной водонасыщенности",
                                                                                  f"test_dependence_sw_residual_{kpr_name}")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if a <= 0 or r2 < 0.7:
                    result = False

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)

                self.dict_of_wrong_values[f"test_dependence_sw_residual_{kpr_name}"] = [{"Кво": wrong_values1,
                                                                                         f"{kpr_name}": wrong_values2},
                                                                                        "выпадает из линии тренда"]

        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_rn_sw_residual_dependence(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=b/(kв^n) при этом 1,1<n<5
        Required data:
            Параметр насыщенности(RI); Sw
        Args:
            self.rn (array[int/float]): массив с данными параметр насыщенности для проверки
            self.sw_residual (array[int/float]): массив с данными коэффициент водонасыщенности для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        parsed_rn, parsed_sw_residual, index_dic = remove_nan_pairs(self.rn, self.sw_residual)
        if self.__check_data(parsed_rn, "Параметр насыщенности(RI)", "test rn sw_residual dependencies") \
                and self.__check_data(parsed_sw_residual, "Sw",
                                      "test rn sw_residual dependencies"):
            r2 = self.test_general_dependency_checking(parsed_rn, parsed_sw_residual,
                                                       "test rn sw_residual dependencies",
                                                       "Параметр_насыщенности(RI)",
                                                       "Коэффициента водонасыщенности")["r2"]
            coefficients = np.polyfit(np.log(parsed_sw_residual), np.log(parsed_rn), 1)
            b, n = np.exp(coefficients[1]), coefficients[0]
            result = True
            if 1.1 >= n or n >= 5 or r2 < 0:
                result = False

            wrong_values1 = []
            wrong_values2 = []
            for i in range(len(parsed_rn)):
                if parsed_rn[i] + (b / (parsed_sw_residual[i] ** n)) * 0.1 > b / (parsed_sw_residual[i] ** n):
                    wrong_values1.append(parsed_rn[i])
                    wrong_values2.append(parsed_sw_residual[i])

            wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
            wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
            self.dict_of_wrong_values["test_rn_sw_residual_dependence"] = [{"Параметр насыщенности(RI)": wrong_values1,
                                                                            "Sw": wrong_values2
                                                                            }, "выпадает из линии тренда"]
            if result:
                report_text = self.generate_report_text(
                    f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
            else:
                report_text = self.generate_report_text(
                    f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

            self.update_report(report_text)
            plt.title("test rn sw_residual dependencies")
            y_pred = b / (parsed_sw_residual ** n)

            # Окрашиваем точки, которые не соответствуют линии тренда, в красный
            for sw_residual_val, rn_val, pred_val in zip(parsed_sw_residual, parsed_rn, y_pred):
                if rn_val + (pred_val * 0.1) < pred_val:
                    plt.scatter(sw_residual_val, rn_val, color='r')

            plt.scatter(parsed_sw_residual, parsed_rn, color='blue', label='Исходные данные')
            plt.plot(parsed_rn, b / (parsed_sw_residual ** n), color='red', label='Линия тренда')
            plt.xlabel('sw_residual')
            plt.ylabel('rn')
            plt.legend()
            equation = f'y = {b:.2f}/(x^{b:.2f}),  r2={r2:.2f}'
            plt.text(np.mean(parsed_sw_residual), np.min(parsed_rn), equation, ha='center', va='bottom')
            plt.savefig("report\\test_rn_sw_residual_dependence.png")
            if get_report:
                plt.show()
                print(report_text)

            return {"result": result, "file_name": self.file_name, "date": self.dt_now}

    def test_rp_kp_dependencies(self, get_report=True):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость по функции y=a/(kп^m)
        при этом m>0. a>0 и a<2,5, 1,1<m<3,8

        Required data:
            Параметр пористости(F); Кп откр

        Args:
            self.rp (array[int/float]): массив с данными Параметр пористости(F) для проверки
            self.porosity_open (array[int/float]): массив с данными Кп откр для проверки

        Returns:
            image: визуализация кроссплота
            dict[str, bool | datetime | str]: словарь с результатом выполнения теста, датой выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        for poro in self.porosity_array:
            porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            parsed_rp, parsed_porosity_open, index_dic = remove_nan_pairs(self.rp, porosity_open)
            if self.__check_data(parsed_rp, "Параметр пористости(F)", f"test rp {poro_name} dependencies") \
                    and self.__check_data(parsed_porosity_open, f"{poro_name}", f"test rp {poro_name} dependencies"):
                r2 = \
                    self.test_general_dependency_checking(parsed_rp, parsed_porosity_open,
                                                          f"test rp {poro_name} dependencies",
                                                          "Параметр пористости(F)",
                                                          f"{poro_name}")["r2"]
                coefficients = np.polyfit(np.log(parsed_porosity_open), -np.log(parsed_rp), 1)
                a, m = np.exp(-coefficients[1]), coefficients[0]
                result = True
                if 1.1 >= m or m >= 3.8 or 0 >= a or a >= 2.5 or r2 < 0.7:
                    result = False

                wrong_values1 = []
                wrong_values2 = []
                for i in range(len(parsed_rp)):
                    if parsed_rp[i] + (a / (parsed_porosity_open[i] ** m)) * 0.1 > a / (parsed_porosity_open[i] ** m):
                        wrong_values2.append(i)

                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)
                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                self.dict_of_wrong_values["test_rp_kp_dependencies"] = [{"Параметр пористости(F)": wrong_values1,
                                                                         f"{poro_name}": wrong_values2
                                                                         }, "выпадает из линии тренда"]
                plt.scatter(parsed_porosity_open, parsed_rp, color='blue', label='Исходные данные')
                plt.plot(parsed_porosity_open, a / (parsed_porosity_open ** m), color='red', label='Линия тренда')
                y_pred = a / (parsed_porosity_open ** m)

                # Окрашиваем точки, которые не соответствуют линии тренда, в красный
                for rp_val, kv_val, pred_val in zip(parsed_rp, parsed_porosity_open, y_pred):
                    if rp_val + (pred_val * 0.1) < pred_val:
                        plt.scatter(kv_val, rp_val, color='r')

                plt.title(f"test rp {poro_name} dependencies")
                plt.xlabel(f'{poro_name}')
                plt.ylabel('rp')
                plt.legend()
                equation = f'y = {a:.2f}/(x^{m:.2f}), r2={r2:.2f}'
                plt.text(np.mean(parsed_rp), np.min(parsed_porosity_open), equation, ha='center', va='bottom')
                if get_report:
                    plt.show()
                    print(report_text)
                # plt.savefig(f"report\\test_rp_{poro_name}_dependencies.png")
                plt.close()
                result_array.append(result)

        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_general_dependency_checking(self, x, y, test_name="не указано", x_name="не указано", y_name="не указано",
                                         get_report=False):
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
        x = list(x)
        y = list(y)
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
                report_text = self.generate_report_text(
                    f"{result}. Выполнен для теста {test_name}.Тест проводился для {x_name} {y_name}. Коэффициент r2 - {r2}\n",
                    1)
                self.update_report(report_text)
                return {"result": result, "r2": r2, "file_name": self.file_name, "date": self.dt_now}
            else:
                result = False

        report_text = self.generate_report_text(
            f"{result}. Выполнен для теста {test_name}.Тест проводился для {x_name} {y_name}. Коэффициент r2 - {r2}\n",
            1)
        self.update_report(report_text)

        if get_report:
            print(report_text)

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
                report_text = self.generate_report_text(
                    f"Все данные монотонны", 1)
            else:
                report_text = self.generate_report_text(
                    f"Индексы выпадающие из монотонности {wrong_values}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)

            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_second(self, get_report=True):
        """
        Тест проводится для оценкци соответствия интервала долбления: подошва-кровля ≥ выносу в метрах
             Required data:
                Кровля интервала отбора; Подошва интервала отбора; Вынос керна
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
                and self.__check_data(self.core_removal_in_meters, "Вынос керна",
                                      "test coring depths second"):
            wrong_values = []
            result = True
            for i in range(len(self.top)):
                if self.bottom[i] - self.top[i] < self.core_removal_in_meters[i]:
                    wrong_values.append(i)
                    result = False
            self.dict_of_wrong_values["test_coring_depths_second"] = [{"Кровля интервала отбора": wrong_values,
                                                                       "Подошва интервала отбора": wrong_values,
                                                                       "Вынос керна": wrong_values
                                                                       },
                                                                      "разница между подошвой и кровлей меньше выноса в м"]
            if result:
                report_text = self.generate_report_text(
                    f"Все данные корректны", 1)
            else:
                report_text = self.generate_report_text(
                    f"Индексы выпадающих значений{wrong_values}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_third(self, get_report=True):
        """
        Тест оценивает соответствие значений выноса керна в метрах и в процентах
             Required data:
                Вынос керна; Вынос керна, %; Кровля интервала отбора; Подошва интервала отбора
            Args:
                self.intervals (array[[int/float]]): массив с массивамими,
                                                    содержашими начало интервала и конец интервала
                self.percent_core_removal (array[int/float]): массив со значениями выноса в процентах
                self.outreach_in_meters(array[int/float]): массив со значениями выноса в метрах

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.outreach_in_meters, "Вынос керна", "test coring depths third") \
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
                                                                      "Вынос керна": wrong_values,
                                                                      },
                                                                     "вынос керна в процентах и метрах не совпадает"]

            if result:
                report_text = self.generate_report_text(
                    f"Все данные корректны", 1)
            else:
                report_text = self.generate_report_text(
                    f"Индексы выпадающих значений{wrong_values}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_coring_depths_four(self, get_report=True):
        """
        Тест проводится с целью соответствия глубин отбора образцов с глубинами выноса керна
            Required data:
                Вынос керна; Глубина отбора, м; Подошва интервала отбора
            Args:
                self.core_removal_in_meters (array[int/float]): массив с выносом керна в метрах
                self.core_sampling (array[int/float]): массив с глубинами отбора образцов
                self.bottom(array[int/float]): массив с подошвой отбора образцов

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.__check_data(self.core_removal_in_meters, "Вынос керна", "test coring depths four") \
                and self.__check_data(self.core_sampling, "Глубина отбора, м", "test coring depths four") \
                and self.__check_data(self.bottom, "Подошва интервала отбора", "test coring depths four"):
            result = True
            wrong_values = []
            for i in range(len(self.core_removal_in_meters)):
                if self.core_removal_in_meters[i] + self.core_sampling[i] > self.bottom[i]:
                    result = False
                    wrong_values.append(i)

            self.dict_of_wrong_values["test_coring_depths_four"] = [{"Глубина отбора, м": wrong_values,
                                                                     "Вынос керна": wrong_values,
                                                                     "Подошва интервала отбора": wrong_values,
                                                                     }, "глубина отбора ниже фактического выноса керна"]
            if result:
                report_text = self.generate_report_text(
                    f"Все данные корректны", 1)
            else:
                report_text = self.generate_report_text(
                    f"Индексы выпадающих значений{wrong_values}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_data_tampering(self, get_report=True):
        """
        Тест выполняется с целью фиксации подлога измерений.
        Подлог заключается в том, что существуют значения параметров,
        схожие вплоть до 3-его знака после запятой
            Required data:
                    Кпр_газ; Кп откр; Кво; Параметр пористости(F);
                    Плотность абсолютно сухого образца; Sw; Параметр_насыщенности(RI)
            Args:
                self.kpr (array[int/float]): массив с данными коэффициент проницаемости для сравнения
                self.porosity_open (array[int/float]): массив с данными Кп откр для сравнения
                self.sw_residual (array[int/float]): массив с данными коэффициент остаточной водонасыщенности для сравнения
                self.rp (array[int/float]): массив с данными Параметр пористости(F) для сравнения
                self.density (array[int/float]): массив с данными всех плотностей для сравнения
                self.rn (array[int/float]): массив с данными параметр насыщенности для сравнения
                self.water_permeability (array[int/float]): массив с данными коэффициент водонасыщенности для сравнения

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """
        result = True
        wrong_values_klickenberg_permeability = []
        wrong_values_parallel_permeability = []
        wrong_values_water_permeability = []
        wrong_values_kpr = []
        wrong_values_porosity_open = []
        wrong_values_poroHe = []
        wrong_values_porosity_kerosine = []
        wrong_values_sw_residual = []
        wrong_values_rp = []
        wrong_values_density = []
        wrong_values_rn = []

        if self.__check_data(self.kpr, "Кпр_газ(гелий)", "test data tampering"):
            unique, indices, counts = np.unique(self.kpr, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.kpr == dup)[0]
                wrong_values_kpr.extend(dup_indices)

        if self.__check_data(self.water_permeability, "Газопроницаемость по воде", "test data tampering"):
            unique, indices, counts = np.unique(self.water_permeability, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.water_permeability == dup)[0]
                wrong_values_water_permeability.extend(dup_indices)

        if self.__check_data(self.klickenberg_permeability, "Газопроницаемость по Кликенбергу", "test data tampering"):
            unique, indices, counts = np.unique(self.klickenberg_permeability, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.klickenberg_permeability == dup)[0]
                wrong_values_klickenberg_permeability.extend(dup_indices)

        if self.__check_data(self.parallel_permeability, "Кпр абс", "test data tampering"):
            unique, indices, counts = np.unique(self.parallel_permeability, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.parallel_permeability == dup)[0]
                wrong_values_parallel_permeability.extend(dup_indices)

        if self.__check_data(self.porosity_open, "Кп откр", "test data tampering"):
            unique, indices, counts = np.unique(self.porosity_open, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.porosity_open == dup)[0]
                wrong_values_porosity_open.extend(dup_indices)

        if self.__check_data(self.sw_residual, "Кво", "test data tampering"):
            unique, indices, counts = np.unique(self.sw_residual, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.sw_residual == dup)[0]
                wrong_values_sw_residual.extend(dup_indices)

        if self.__check_data(self.poroHe, "Открытая пористость по газу", "test data tampering"):
            unique, indices, counts = np.unique(self.poroHe, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.poroHe == dup)[0]
                wrong_values_poroHe.extend(dup_indices)

        if self.__check_data(self.porosity_kerosine, "Открытая пористость по керосину", "test data tampering"):
            unique, indices, counts = np.unique(self.porosity_kerosine, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.porosity_kerosine == dup)[0]
                wrong_values_porosity_kerosine.extend(dup_indices)

        if self.__check_data(self.rp, "Параметр пористости(F)", "test data tampering"):
            unique, indices, counts = np.unique(self.rp, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.rp == dup)[0]
                wrong_values_rp.extend(dup_indices)

        if self.__check_data(self.density, "Плотность абсолютно сухого образца", "test data tampering"):
            unique, indices, counts = np.unique(self.density, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.density == dup)[0]
                wrong_values_density.extend(dup_indices)

        if self.__check_data(self.rn, "Параметр насыщенности(RI)", "test data tampering"):
            unique, indices, counts = np.unique(self.rn, return_inverse=True, return_counts=True)
            duplicates = unique[counts > 1]

            for dup in duplicates:
                result = False
                dup_indices = np.where(self.rn == dup)[0]
                wrong_values_rn.extend(dup_indices)

        wrong_values = []
        wrong_values.append(wrong_values_kpr)
        wrong_values.append(wrong_values_porosity_open)
        wrong_values.append(wrong_values_klickenberg_permeability)
        wrong_values.append(wrong_values_parallel_permeability)
        wrong_values.append(wrong_values_water_permeability)
        wrong_values.append(wrong_values_sw_residual)
        wrong_values.append(wrong_values_density)
        wrong_values.append(wrong_values_rp)
        wrong_values.append(wrong_values_rn)
        wrong_values.append(wrong_values_poroHe)
        wrong_values.append(wrong_values_porosity_kerosine)
        self.dict_of_wrong_values["test_data_tampering"] = [{"Кпр_газ(гелий)": wrong_values_kpr,
                                                             "Газопроницаемость по воде": wrong_values_water_permeability,
                                                             "Газопроницаемость по Кликенбергу": wrong_values_klickenberg_permeability,
                                                             "Кпр абс": wrong_values_parallel_permeability,
                                                             "Кп откр": wrong_values_porosity_open,
                                                             "Открытая пористость по газу": wrong_values_poroHe,
                                                             "Открытая пористость по керосину": wrong_values_porosity_kerosine,
                                                             "Кво": wrong_values_sw_residual,
                                                             "Параметр пористости(F)": wrong_values_rp,
                                                             "Параметр насыщенности(RI)": wrong_values_rn,
                                                             "Плотность абсолютно сухого образца": wrong_values_density},
                                                            "схожие значения"]
        if result:
            report_text = self.generate_report_text(
                f"Все данные корректны", 1)
        else:
            report_text = self.generate_report_text(
                f"Индексы выпадающих значений{wrong_values}", 0)

        self.update_report(report_text)
        if get_report:
            print(report_text)
        return {"result": result, "wrong_values": wrong_values, "report_text": self.report_text, "date": self.dt_now}

    def test_kp_in_surface_and_reservoir_conditions(self, get_report=True):
        """
        Тест выполянется для сравнения коэффициентов пористости, оцененных пластовых
        и атмосферных условиях. Кп в атмосферных условиях всегда больше чем Кп в пластовых условиях.

        Required data:
            Кп откр TBU; Кп откр

        Args:
            self.porosity_open (array[int/float]): массив с данными Кп откр
                                                в пластовых условиях для проверки
            self.poro_tbu (array[int/float]): массив с данными Кп откр
                                                в атмосферных условиях для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        result_array = []
        wrong_values = []
        for poro in self.porosity_array:
            self.porosity_open = self.porosity_array[poro]
            poro_name = self.poro_name_dic[poro]
            if self.__check_data(self.porosity_open, f"{poro_name}",
                                 f"test {poro_name} and Кп откр TBU") and \
                    self.__check_data(self.poro_tbu, "Кп откр TBU",
                                      f"test {poro_name} and Кп откр TBU"):

                result = True
                for i in range(len(self.porosity_open)):
                    if self.poro_tbu[i] <= self.porosity_open[i]:
                        result = False
                        wrong_values.append(i)

                self.dict_of_wrong_values[f"test {poro_name} and Кп откр TBU"] = [
                    {f"{poro_name}": wrong_values,
                     }, f"{poro_name} больше или равно Кп откр TBU"]

                if result:
                    report_text = self.generate_report_text(
                        f"Все данные корректны", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Индексы выпадающих значений{wrong_values}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
        return {"result": result_array, "wrong_values": wrong_values, "report_text": self.report_text,
                "date": self.dt_now}

    def test_table_notes(self, get_report=True):
        """
        Тест проводится с целью ранее установленных несоответствий/аномалий
        при лаборатоном анализе керна, указанных в “примечаниях”

        Required data:
            Примечание(в керне);
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
            self.dict_of_wrong_values["test_table_notes"] = [{"Примечание(в керне)": indexes},
                                                             "присутствует неисправность"]

            report_text = self.generate_report_text(f"Индексы выпадающих значений{indexes}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            return indexes
        except:
            self.dict_of_wrong_values["test_table_notes"] = [{"Примечание(в керне)": [0]}, "пустой"]

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
                report_text = self.generate_report_text(
                    f"Все данные корректны", 1)
            else:
                report_text = self.generate_report_text(
                    f"Индексы выпадающих значений{wrong_values}", 0)

            self.update_report(report_text)
            if get_report:
                print(report_text)
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_correctness_of_p_sk_kp(self, get_report=True):
        """
            Тест выполянется с целью оценки соответствия плотности/карбонатности
            и пористости образцов в зависимости от направления измерений
            (перпендикулярный или параллельный образец). Разница не должна превышать
            0.5 % абсолютных Например, 7.5 и 7.1 будут схожими замерами по пористости,
            а 7.5 и 6.9 уже непохожими
            Required data:
                 Плотность абсолютно сухого образца; Карбонатность; Кп откр
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
            "Кп откр": wrong_den,
            "Карбонатность": wrong_carb,
            "Плотность абсолютно сухого образца": wrong_poro
        }, "расхождение параметров выше 5%"]

        if result:
            report_text = self.generate_report_text(
                f"Все данные корректны", 1)
        else:
            report_text = self.generate_report_text(
                f"Индексы выпадающих значений{wrong_values}", 0)

        self.update_report(report_text)
        if get_report:
            print(report_text)
        return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": self.dt_now}

    def test_dependence_kno_kpr(self, get_report=True):
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
        result_array = []
        for kpr_elem in self.kpr_array:
            kpr = self.kpr_array[kpr_elem]
            kpr_name = self.kpr_name_dic[kpr_elem]
            parsed_kpr, parsed_kno, index_dic = remove_nan_pairs(kpr, self.kno)
            if self.__check_data(parsed_kno, "Кно", "test dependence kno kpr") and \
                    self.__check_data(parsed_kpr, f"{kpr_name}", "test dependence kno kpr"):

                r2 = self.test_general_dependency_checking(parsed_kno, parsed_kpr, "test dependence kno kpr",
                                                           "Коэффициент остаточной нефтенасыщенности",
                                                           f"{kpr_name}")["r2"]

                coefficients = np.polyfit(self.kno, np.exp(parsed_kpr), 1)
                a, b = coefficients[0], coefficients[1]
                result = True
                if a >= 0 or b <= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = logarithmic_function_visualization(parsed_kno, parsed_kpr, a, b, r2,
                                                                                  get_report,
                                                                                  "Коэффициент остаточной нефтенасыщенности",
                                                                                  f"{kpr_name}",
                                                                                  f"test_dependence_kno_{kpr_name}")
                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                self.dict_of_wrong_values[f"test_dependence_kno_{kpr_name}"] = [{"Кно": wrong_values1,
                                                                                 f"{kpr_name}": wrong_values2},
                                                                                "выпадает из линии тренда"]
                result_array.append(result)

                return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_dependence_kpr_kgo(self, get_report=True):
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
        result_array = []
        for kpr_elem in self.kpr_array:
            kpr = self.kpr_array[kpr_elem]
            kpr_name = self.kpr_name_dic[kpr_elem]
            parsed_kpr, parsed_kgo, index_dic = remove_nan_pairs(kpr, self.kgo)
            if self.__check_data(parsed_kgo, "Кго", f"test dependence {kpr_name} kgo") and \
                    self.__check_data(parsed_kpr, f"{kpr_name}", f"test dependence {kpr_name} kgo"):

                r2 = self.test_general_dependency_checking(parsed_kgo, parsed_kpr, f"test dependence kno {kpr_name}",
                                                           "Коэффициент остаточной нефтенасыщенности",
                                                           f"{kpr_name}")["r2"]

                coefficients = np.polyfit(parsed_kpr, np.exp(parsed_kgo), 1)
                a, b = coefficients[0], coefficients[1]
                result = True
                if a >= 0 or b <= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = logarithmic_function_visualization(parsed_kgo, parsed_kpr, a, b, r2,
                                                                                  get_report,
                                                                                  "Cвязанная газонасыщенность",
                                                                                  f"{kpr_name}",
                                                                                  f"test_dependence_kno_{kpr_name}")

                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                self.dict_of_wrong_values[f"test_dependence_kno_{kpr_name}"] = [{"Sgl": wrong_values1,
                                                                                 f"{kpr_name}": wrong_values2},
                                                                                "выпадает из линии тренда"]
                result_array.append(result)

        return {"result": result_array, "report_text": self.report_text, "date": self.dt_now}

    def test_dependence_kpr_knmng(self, get_report=True):
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
        result_array = []
        for kpr in self.kpr_array:
            kpr = self.kpr_array[kpr]
            kpr_name = self.kpr_name_dic[kpr]
            parsed_knmng, parsed_kpr, index_dic = remove_nan_pairs(self.knmng, kpr)
            if self.__check_data(parsed_knmng, "Кнмнг", f"test dependence {kpr_name} knmng") and \
                    self.__check_data(parsed_kpr, f"{kpr_name}", f"test dependence {kpr_name} knmng"):

                r2 = self.test_general_dependency_checking(parsed_knmng, parsed_kpr, f"test dependence kno {kpr_name}",
                                                           "Коэффициент остаточной нефтенасыщенности",
                                                           f"{kpr_name}")["r2"]

                coefficients = np.polyfit(parsed_kpr, np.exp(parsed_knmng), 1)
                a, b = coefficients[0], coefficients[1]
                result = True
                if a >= 0 or b <= 0 or r2 < 0.7:
                    result = False

                wrong_values1, wrong_values2 = logarithmic_function_visualization(parsed_knmng, parsed_kpr, a, b, r2,
                                                                                  get_report,
                                                                                  "Кно(Sowcr)",
                                                                                  f"{kpr_name}",
                                                                                  f"test_dependence_{kpr_name}_knmng")

                wrong_values1 = remap_wrong_values(wrong_values1, index_dic)
                wrong_values2 = remap_wrong_values(wrong_values2, index_dic)

                if result:
                    report_text = self.generate_report_text(
                        f"Зависимость выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 1)
                else:
                    report_text = self.generate_report_text(
                        f"Зависимость не выполняется. Выпадающие точки {wrong_values1, wrong_values2}", 0)

                self.update_report(report_text)
                if get_report:
                    print(report_text)
                self.dict_of_wrong_values[f"test_dependence_kno_{kpr_name}"] = [{"Кно(Sowcr)": wrong_values1,
                                                                                 f"{kpr_name}": wrong_values2},
                                                                                "выпадает из линии тренда"]
                result_array.append(result)

        return {"result": result_array, "file_name": self.file_name, "date": self.dt_now}

    def start_tests(self, list_of_tests: list):
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
