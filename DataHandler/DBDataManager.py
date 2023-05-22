import DataManagerInterface as DataManager

class DBDataManager(DataManager):
    def __init__(self, db_config):
        super().__init__()
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """
        Устанавливает соединение с базой данных.
        """
        self.conn = sqlite3.connect(**self.db_config)
    
    def disconnect(self):
        """
        Закрывает соединение с базой данных.
        """
        if self.conn is not None:
            self.conn.close()
    
    def load_data(self, query):
        """
        Загружает данные из базы данных по заданному запросу.
        """
        if self.conn is None:
            self.connect()
        return pd.read_sql_query(query, self.conn)
    
    def save_data(self, data, table_name):
        """
        Сохраняет данные в таблицу базы данных.
        """
        if self.conn is None:
            self.connect()
        data.to_sql(table_name, self.conn, if_exists='replace', index=False)