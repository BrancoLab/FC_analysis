import datajoint as dj
import pandas as pd
import datetime

# from behaviour.tracking.tracking import compute_body_segments
from fcutils.file_io.io import load_excel_file, load_yaml
from fcutils.file_io.utils import listdir

from analysis.misc.paths import *
from analysis.misc.paths import experiments_file, surgeries_file

from analysis.dbase.utils.dj_config import start_connection, dbname

schema = start_connection()


def manual_insert_skip_duplicate(table, key):
	try:
		table.insert1(key)
		return True
	except Exception as e:
		if isinstance(e, dj.errors.DuplicateError):
			return False # Just a duplicate warning
		elif isinstance(e, dj.errors.IntegrityError):
			raise ValueError("Could not insert in table, likely missing a reference to foreign key in parent table!\n{}\n{}".format(table.describe(), e))
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

	class Injection(dj.Part):
		definition = """
			-> Mouse
			target: varchar(64)
			compound: varchar(64)
		"""

	def pop(self):  
		# Populate the main table
		mice_data = load_excel_file(mice_log)
		in_table = list(self.fetch("mouse_id"))
		for mouse in mice_data:
			if mouse['ID'] in in_table:
				continue
			
			key = dict(mouse_id = mouse['ID'].upper(), strain=mouse['MouseLine'].upper(),
						sex=mouse['Gender'].upper())
			self.insert1(key)

		# Populate the surgeries tables
		surgery_data = load_yaml(surgeries_file)
		in_table = list(self.fetch("mouse_id"))

		for mouse in surgery_data.keys():
			if mouse not in in_table: continue
			surgery = surgery_data[mouse]

			for inj in [k for k in list(surgery.keys()) if 'injection' in k]:
				part_key = dict(mouse_id = mouse,
								target = surgery[inj]['target'],
								compound = surgery[inj]['compound'])
				
				manual_insert_skip_duplicate(self.Injection, part_key)


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
		exps = load_yaml(experiments_file)
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
		exps = load_yaml(experiments_file)
		for exp in exps.keys():
			for subexp in exps[exp]['subexps']:
				key=dict(exp_name=exp, subname=list(subexp.keys())[0])
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

	class Metadata(dj.Part):
		definition = """
			-> Session
			---
			naive: int
			lights: int
			shelter: int
		"""

	class IPinjection(dj.Part):
		definition = """
			-> Session
			---
			injected: varchar(12)
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

			# Check if an entry for this session exists already
			intable = (self & f"mouse_id='{key['mouse_id']}'" & f"date='{key['date']}'").fetch(as_dict=True)
			if not intable:
				manual_insert_skip_duplicate(self, key)

		clean_session_data = [s for s in session_data if s['Mouse']]
		self.pop_metadata(clean_session_data)
		self.pop_ip_injections(clean_session_data)

	def pop_metadata(self, clean_session_data):
		# Populate subtable 
		for key in self.fetch(as_dict=True):
			datalog_entry = [s for s in clean_session_data \
								if s['Date'].strftime("%Y-%m-%d")==key['date'].strftime("%Y-%m-%d") \
								and s['Mouse']==key['mouse_id']][0]

			key['naive'] = datalog_entry['naive']
			key['lights'] = datalog_entry['lights']
			key['shelter'] = datalog_entry['shelter']
			manual_insert_skip_duplicate(self.Metadata, key)

	def pop_ip_injections(self, clean_session_data):
		for key in self.fetch(as_dict=True):
			datalog_entry = [s for s in clean_session_data \
					if s['Date'].strftime("%Y-%m-%d")==key['date'].strftime("%Y-%m-%d") \
					and s['Mouse']==key['mouse_id']][0]

			if datalog_entry['IP_injection']:
				key['injected'] = datalog_entry['IP_injection'].upper()


	# ----------------------------------- UTILS ---------------------------------- #
	
	def get_files_for_session(self, session_id=None, mouse_id=None, date=None):
		# Get session given args and check all went okay
		if session_id is not None:
			session = (self & f"session_id={session_id}").fetch(as_dict=True)
		elif mouse_id is not None and date is not None:
			session = (self & f"mouse_id='{mouse_id}'" & f"date='{date}'").fetch(as_dict=True)
		else:
			raise ValueError("Need to pass either session id or mouse id + date")

		if not session:
			print("Couldn't find any session with args: {} - {} - {}".format(session_id, mouse_id, date))
			return None
		elif len(session) > 1:
			raise ValueError("Found too many sessions")
		else:
			session = session[0]
		
		# Create name
		sessname = session['date'].strftime("%y%m%d")+f"_{session['mouse_id']}"

		# Get files
		video_tdms = [f for f in listdir(raw_video_fld) if sessname in f and f.endswith(".tdms")][0]
		metadata_tmds = [f for f in listdir(raw_metadata_fld) if sessname in f and f.endswith(".tdms")][0]
		inputs_tmds = [f for f in listdir(raw_analog_inputs_fld) if sessname in f and f.endswith(".tdms")][0]

		try:
			converted_vid = [f for f in listdir(raw_video_fld) if sessname in f and f.endswith(".mp4")][0]
		except:
			converted_vid = None

		try:
			tracking = [f for f in listdir(raw_tracking_fld) if sessname in f and f.endswith(".h5")][0]
		except:
			tracking = None

		files = dict(
			raw_video = video_tdms, 
			raw_metadata = metadata_tmds,
			raw_inputs = inputs_tmds,
			converted_video = converted_vid,
			trackingdata_file = tracking,
		)
		return files


# ---------------------------------------------------------------------------- #
#                                   TRACKING                                   #
# ---------------------------------------------------------------------------- #
@schema
class Tracking(dj.Imported):
	definition = """
		-> Session
	"""

	bparts = ['snout', 'left_ear', 'right_ear', 'neck', 'body', 'tail']
	bsegments = [('snout', 'left_ear'), ('snout', 'right_ear'),
				('left_ear', 'neck'), ('right_ear', 'neck'),
				('neck', 'body'), ('body', 'tail')]

	class BodyPartTracking(dj.Part):
		definition = """
			-> Tracking
			bp: varchar(64)
			---
			x: longblob
			y: longblob
			speed: longblob
			dir_of_mvmt: longblob
			angular_velocity: longblob
		"""

	class BodySegmentTracking(dj.Part):
		definition = """
			-> Tracking
			bp1: varchar(64)
			bp2: varchar(64)
			---
			orientation: longblob
			angular_velocity: longblob
		"""

	# Populate method
	def _make_tuples(self, key):
		# Insert entry into main class
		self.insert1(key)

		# Insert into the bodyparts tracking
		for bp in self.bparts:
			bp_key = key.copy()
			bp_key['bp'] = bp
			
			# TODO load the relevant tracking data

			# TODO organize the tracking into a new key

			self.BodyPartTracking.isert1(bp_key)

		# Insert into the body segment data
		for (bp1, bp2) in self.bsegments:
			segment_key = key.copy()
			segment_key['bp1'] = bp1
			segment_key['bp2'] = bp2

			# TODO get relevant data with:
			# compute_body_segments # <- need to finish writing this

			# TODO turn it into a key

			self.BodySegmentTracking.insert1(segment_key)

