import datajoint as dj


from fcutils.file_io.io import load_excel_file

from analysis.misc.paths import mice_log
from analysis.dbase.dj_config import start_connection, dbname

schema = start_connection()


# ---------------------------------------------------------------------------- #
#                                     MOUSE                                    #
# ---------------------------------------------------------------------------- #
@schema
class Mouse(dj.Manual):
    definition = """
		# Mouse table lists all the mice used and the relevant attributes
		mouse_id: varchar(128)                        # unique mouse id
		---
		strain:   varchar(128)                        # genetic strain
		sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
	"""

    def pop(self):  
        mice_data = load_excel_file(mice_log)

        a = 1


if __name__ == '__main__':
    mouse = Mouse()
    mouse.pop()