import datetime
import inspect
import os
from typing import List, Optional


class QA_QC_main():
    def __init__(self):
        self.report_text = ""
        self.ident = ' ' * 5  # отступ при формировании отчета
        self.delimeter = '\n------------------------------------------------------------------------------------'

    def get_list_of_tests(self) -> list:
        """
        Метод для получения списка тестов для данных реализованных в классе

        Returns:
            list: список с названиями методов реализующих тесты
        """
        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def get_method_description(self, method_name: str) -> str:
        """
        Метод для получение описания теста по его названию

         Args:
             method_name(str) - название теста
        Returns:
             str - описание теста
         """
        method = getattr(self, method_name, None)
        if method is not None:
            return method.__doc__
        else:
            return "Метод не найден."

    def start_tests(self, list_of_tests: List[str], get_report=True) -> List[dict]:
        """
        Метод который запускает все тесты, которые переданы в виде списка list_of_tests

        Args:
            list_of_tests (list): список названий тестов которые должны быть проведены
            get_report (bool, optional): _description_. Defaults to True.

        Returns:
            dict: результаты выбранных тестов
        """
        results = []
        for method_name in list_of_tests:

            try:
                method = getattr(self, method_name)

                try:
                    report = method(get_report=get_report)
                except TypeError as _:
                    report = method()
                results.append({**report, 'test_name': method_name})

            except Exception as e:
                results.append({"test_name": method_name, 'error': str(e)})
                print(method_name, e)

        return results

    def start_tests_debug(self, list_of_tests: List[str], get_report=True) -> List[dict]:
        results = []

        for method_name in list_of_tests:
            method = getattr(self, method_name)

            try:
                report = method(get_report=get_report)
            except TypeError as _:
                report = method()

            results.append(report)
        return results

    def start_tests_with_filters(self, list_of_tests: List[tuple[str, Optional[List[dict]]]], get_report=True) \
            -> List[dict]:
        """
        Метод который запускает все тесты вместе с фильтрами, которые переданы в виде списка list_of_tests
        List[tuple[str- (имя теста), List[dict] - фильтры теста]]

        Args:
            list_of_tests (list): список названий тестов которые должны быть проведены
            get_report (bool, optional): _description_. Defaults to True.

        Returns:
            dict: результаты выбранных тестов
        """
        results: List[dict] = []
        for method_name, filters in list_of_tests:
            try:
                method = getattr(self, method_name)
                results.append({"test": method(get_report=get_report, filters=filters), "filters": filters})
            except Exception as e:
                results.append({"test_name": method_name, "filters": filters, 'error': str(e)})
                print(e)

        return results

    def generate_test_report(self, file_name='test_report', file_path='report', data_name=None):
        """
        Метод для генерации отчета в виде текстового файла

        Args:
            file_name (str, optional): название файла с отчетом. Defaults to 'test_report'.
            file_path (str, optional): директория в которую следует сохранить отчет. Defaults to 'report'.
            data_name (str, optional): название данных который подвергались тестированию. 
                                       Данное название отобразится в итоговом отчете. Defaults to self.file_name.
        """
        if not os.path.isdir(file_path):
            os.mkdir(file_path)

        data_name = self.file_name if not data_name else data_name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        report = f"Отчет о тестировании от {timestamp}{self.ident}Название тестируемого файла: '{data_name}'\n\n{self.report_text} "
        with open(f"{file_path}/{file_name}.txt", "w") as file:
            file.write(report)

    def update_report(self, report_text):
        caller_method_name = inspect.stack()[1][3]  # получаем имя метода из которого вызван текущий (название теста)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / {caller_method_name}:\n{report_text}\n\n"

    def generate_report_text(self, text, status):
        status_dict = {0: 'Тест не пройден.',
                       1: 'Тест пройден успешно.',
                       2: 'Тест не был запущен.'}

        report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}"
        return report_text
