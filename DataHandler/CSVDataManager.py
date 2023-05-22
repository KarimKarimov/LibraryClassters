import DataManagerInterface as DataManager

class CSVDataManager(DataManager):
    def __init__(self):
        super().__init__()
    
    def load_data(self, data_path):
        """
        Загружает данные из CSV-файла.
        """
        return pd.read_csv(data_path)
    
    def save_data(self, data, save_path):
        """
        Сохраняет данные в CSV-файл.
        """
        data.to_csv(save_path, index=False)