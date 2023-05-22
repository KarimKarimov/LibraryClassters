import DataManagerInterface as DataManager

class ExcelDataManager(DataManager):
    def __init__(self):
        super().__init__()
    
    def load_data(self, data_path):
        """
        Загружает данные из Excel-файла.
        """
        return pd.read_excel(data_path)
    
    def save_data(self, data, save_path):
        """
        Сохраняет данные в Excel-файл.
        """
        data.to_excel(save_path, index=False)