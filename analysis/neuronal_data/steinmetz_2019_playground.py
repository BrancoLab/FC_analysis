# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oneibl.onelight import ONE


# %%
one = ONE()
one.set_figshare_url("https://figshare.com/articles/steinmetz/9974357")
sessions = one.search(['spikes'])

# %%
# Get data
sc_channels, pag_channels = pd.DataFrame(), pd.DataFrame()
for sess_n, session in enumerate(sessions):
    # ------------------------ Get channels in SC ------------------------------- #
    channels = one.load_object(session, 'channels')

    channels_data = channels.brainLocation
    channels_data['probe_idx'] = channels.probe.astype(np.int32)
    channels_data['raw_data_row'] = channels.rawRow
    channels_data['session'] = [session for i in range(len(channels_data))]
    channels_data['site'] = channels.site.ravel()
    channels_data['session_idx'] = [sess_n for i in range(len(channels_data))]
    
    sc_channels = channels_data.loc[channels_data.allen_ontology == 'SCm']
    if not len(sc_channels): continue

    # ---------------------------- Get clusters in SC ---------------------------- #
    clusters = one.load_object(session, 'clusters')

    clusters_data = pd.DataFrame(dict(phy_annot=list(clusters._phy_annotation.ravel()),
                                        ch=list(clusters.peakChannel.ravel().astype(np.int32)),
                                        probe=list(clusters.probes.ravel().astype(np.int32))))
    clusters_data = clusters_data.loc[clusters_data.phy_annot > 2]
    n_probes = np.max(clusters_data.probe) + 1

    sc_clusters = pd.DataFrame()
    for probe in range(n_probes):
        probe_sc_channels = sc_channels.loc[sc_channels.probe_idx == probe]
        probe_clusters = clusters_data.loc[clusters_data.probe==probe]

        sc_clusters = pd.concat([sc_clusters, probe_clusters.loc[probe_clusters.ch.isin(probe_sc_channels.site)]])


    # ----------------------- Get spikes for clusters in SC ---------------------- #
    spikes = one.load_object(session, 'spikes')
    
    spikes_data = pd.DataFrame(dict(cluster = spikes.clusters.ravel().astype(np.int32),
                                        time=spikes.times.ravel()))
    spikes_data = spikes_data.loc[spikes_data.cluster.isin(sc_clusters.ch)]
    print(f'Found {len(set(spikes_data.cluster))} clusters for session {session}')

    if session == 'nicklab/Subjects/Richards/2017-10-30/001':
        break
    
# %%
# --------------------------- reorganize spike data -------------------------- #

clusters = list(set(spikes_data.cluster))
n_clusters = len(clusters)
# max_t = np.int32(np.max(spikes_data.time) * 1000)
max_t = 3000000

samples_per_bin = 1000
n_bins = int((max_t)/samples_per_bin)

spikes = np.zeros((max_t, n_clusters))
binned_spikes = np.zeros((int(max_t/samples_per_bin), n_clusters))

for n, cluster in enumerate(clusters):
    times = np.int32(spikes_data.loc[spikes_data.cluster == cluster].time.values * 1000)
    spikes[times, n] = 1

    binned_spikes[:, n] = np.mean(spikes[:, n].reshape((samples_per_bin, -1)), axis=0)


# %%
plt.plot(binned_spikes[:, 0], binned_spikes[:, 2])


# %%
