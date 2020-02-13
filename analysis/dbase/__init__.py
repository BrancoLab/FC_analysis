from analysis.dbase.utils.dj_config import start_connection, dbname

try:
    schema = start_connection()
except Exception as e:
    print("Could not load database: ", e)