class KernConsts:
    def __init__(self) -> None:
        self.general_dependency_checking_wrong ="Высокая дисперсия данных"
        self.general_dependency_checking_accepted ="Низкая дисперсия данных"
        self.check_data_empty_wrong = "Передан пустой массив"
        self.check_data_not_array_wrong = "Передан не массив"
        self.check_data_not_int_wrong = "Содержит не числовое значение"
        self.check_data_has_nan_wrong = "Содержит nan"
        self.monotony_wrong = "Нарушена монотонность глубины"
        self.monotony_accepted = "Глубина монотонно возрастает"
        self.dependency_wrong = "Зависимость не выполняется. Выпадающие точки"
        self.dependency_accepted = "Зависимость выполняется. Выпадающие точки"