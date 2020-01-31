import pandas as pd
from sqlalchemy import text


class ReadDatabase(object):
    def __init__(self, engine, db):
        """
        Interface for reading database tables from a given database. Only simple selections
        can be made using this class. Any processing should be encapsulated elsewhere.

        :param engine: sqlalchemy engine object.
        :param db: string, db name.
        """
        self.engine = engine
        self.db = db

    def get_data(self, sql_query_columns, table, filter=None, index=None):
        """
        Read a particular table from a database, implementing filters and setting the index column.

        :param sql_query_columns: single string with the columns selected from the table separated
         by commas, like 'A, B, C'.
        :param table: string, table name.
        :param filter: string, corresponding to the clause after `WHERE` in a SQL filter. Default `None`.
        :param index: string, name of the index column from the selected columns. Default `None`.
        :return: pandas dataframe with the selected columns filtered and indexed.
        """
        connection = self.engine.connect()
        if filter is None:
            filter_text = ''
        else:
            filter_text = f'WHERE {filter}'
        query = text(f'SELECT {sql_query_columns} FROM {self.db}.{table} {filter_text}')
        print('-'*70 + '\n'
              + f'reading columns {sql_query_columns} from table {table} from database {self.db}')
        df = pd.read_sql(
            query,
            connection,
            index_col=index)
        connection.close()
        return df
