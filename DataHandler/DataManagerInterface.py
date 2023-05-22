class DataManager:
    def __init__(self):
        pass
    
    def load_data(self, data_path):
        """
        Загружает данные из указанного пути.
        """
        pass
    
    def save_data(self, data, save_path):
        """
        Сохраняет данные по указанному пути.
        """
        pass

    def clean_data(self, data):
        """
        Очищает данные от некорректных или пропущенных значений.
        """
        pass
    
    def split_data(self, data, train_size=0.8):
        """
        Разделяет данные на обучающую и тестовую выборки.
        """
        pass

    def normalize_data(self, data):
        """
        Нормализует данные.
        """
        pass

    def select_features(self, data):
        """
        Отбирает наиболее важные признаки.
        """
        pass