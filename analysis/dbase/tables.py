import datajoint as dj
import pandas as pd

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
		id: smallint auto_increment
		---
        mouse_id: varchar(128)                        # unique mouse id
		strain:   varchar(128)                        # genetic strain
		sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
	"""

    def pop(self):  
        mice_data = load_excel_file(mice_log)

        in_table = list(self.fetch("mouse_id"))
        for mouse in mice_data:
            if mouse['ID'] in in_table:
                continue
            
            key = dict(mouse_id = mouse['ID'].upper(), strain=mouse['MouseLine'].upper(),
                        sex=mouse['Gender'].upper())
            self.insert1(key)
        
        print("Finished populating mouse table: ")
        print(pd.DataFrame(self.fetch()).tail())


# ---------------------------------------------------------------------------- #
#                                  EXPERIMENT                                  #
# ---------------------------------------------------------------------------- #
@schema
class Experiment(dj.Manual):
    definition = """
        name: varchar(512)                        
        ---
        arena: varchar(64)
    """
    def pop(self, exp, arena):
        names_in_table = list(self.fetch('name'))
        if exp not in names_in_table:
            key = dict(name=exp, arena=arena)
            self.insert1(key)  


@schema
class Subexp(dj.Manual):
    definition = """
    -> Experiment
    subname: varchar(512)
    """
    exp_table = Experiment()

    def pop(self, name, subname):
        key = dict(name=name, subname=subname)
        try:
            self.insert1(key)
        except Exception as e:
            if 'Duplicate entry' in e.args[0]:
                pass # Just a duplicate warning
            else:
                raise ValueError(e)

    def show(self):
        print("Entries in experiments table: ")
        for exp,arena in zip(*self.exp_table.fetch('name', 'arena')):
            print("Experiment {} -  in arena {}".format(exp,arena))
            print((self & "name='{}'".format(exp)))
            print("\n\n")
        return ""



if __name__ == '__main__':
