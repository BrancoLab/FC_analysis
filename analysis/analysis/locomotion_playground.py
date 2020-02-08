# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.dbase.tables import Session, Tracking

# %%
# Get data
experiment = 'Circarena'
subexperiment = 'baseline'
bpart = 'body'

entries = Session * Tracking.BodyPartTracking & f"exp_name='{experiment}''" \
                & f"subname='{subexperiment}'" & f"bp='{bpart}'"
tracking_data = pd.DataFrame(entries.fetch())

# TODO inspect how data are organize

# %%
# Few generic plots

