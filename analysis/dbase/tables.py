import datajoint as dj
import pandas as pd

from fcutils.file_io.io import load_excel_file, load_yaml

from analysis.misc.paths import *
from analysis.dbase.dj_config import start_connection, dbname

schema = start_connection()


def manual_insert_skip_duplicate(table, key):
    try:
        table.insert1(key)
    except Exception as e:
        if isinstance(e, dj.errors.DuplicateError):
            pass # Just a duplicate warning
        else:
            raise ValueError(e)
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
        exp_name: varchar(512)                        
        ---
        arena: varchar(64)
    """
    def pop(self):
        exps = load_yaml("analysis\dbase\populate\experiments.yml")
        for exp in exps.keys():
            key = dict(exp_name=exp, arena=exps[exp]['arena'])
            manual_insert_skip_duplicate(self, key)


@schema
class Subexp(dj.Manual):
    definition = """
    -> Experiment
    subname: varchar(512)
    """
    exp_table = Experiment()

    def pop(self):
        exps = load_yaml("analysis\dbase\populate\experiments.yml")
        for exp in exps.keys():
            for subexp in exps[exp]['subexps']:
                key=dict(exp_name=exp, subname=subexp)
                manual_insert_skip_duplicate(self, key)

    def show(self):
        print("Entries in experiments table: ")
        for exp,arena in zip(*self.exp_table.fetch('exp_name', 'arena')):
            print(  "Experiment {} -  in arena {}".format(exp,arena))
            print((self & "exp_name='{}'".format(exp)))
            print("\n\n")


# ---------------------------------------------------------------------------- #
#                                    SESSION                                   #
# ---------------------------------------------------------------------------- #
@schema
class Session(dj.Manual):
    definition = """
    session_id: smallint auto_increment
    -> Mouse
    -> Subexp
    date: date
    """

    def pop(self):
        session_data = load_excel_file(sessions_log)
        for session in session_data:
            if not session['Mouse']: continue
            key = dict(
                mouse_id=session['Mouse'].upper(),
                exp_name=session['Experiment'],
                subname=session['Subexperiment'],
                date=session['Date'].strftime("%Y-%m-%d"),                
            )
            manual_insert_skip_duplicate(self, key)