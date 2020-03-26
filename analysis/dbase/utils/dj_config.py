import matplotlib.pyplot as plt
try:
    import datajoint as dj
except Exception as e:
    print("Could not import datajoint: {}".format(e))
    pass

import sys
if sys.platform == "darwin":
    # ip = "192.168.241.87"
    ip = "localhost"
else:
    ip = "localhost"

dbname = 'Locomotion'    # Name of the database subfolder with data

def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    """

    if dj.config['database.user'] != "root":
        try:
            dj.config['database.host'] = ip
        except Exception as e:
            print("Could not connect to database: ", e)
            return None, None

        dj.config['database.user'] = 'root'
        dj.config['database.password'] = 'fede'
        dj.config['database.safemode'] = True
        dj.config['safemode']= True

        dj.config["enable_python_native_blobs"] = True

        dj.conn()

    schema = dj.schema(dbname)
    return schema


def print_erd():
    schema = start_connection()
    # dj.ERD(schema).draw()
    dj.Diagram(schema).draw()
    plt.show()


if __name__ == "__main__":
    start_connection()
    print_erd()
