from Utils.imports import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
col = [0, 0, 0]
# col = [.8, .8, .8]
matplotlib.rc('axes', edgecolor=col)
matplotlib.rcParams['text.color'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['axes.labelcolor'] = col
matplotlib.rcParams['xtick.color'] = col
matplotlib.rcParams['ytick.color'] = col

plt.rcParams.update(params)

import platform
import math
import random
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from Plotting.Plotting_utils import make_legend, save_all_open_figs
from Utils.Data_rearrange_funcs import flatten_list
from Utils.maths import line_smoother

from Config import cohort_options

arms_colors = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))


class MazeCohortProcessor:
    def __init__(self, cohort=None, store_tracking=False, load=False):

        if 'Windows' in platform.system():
            fig_save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\181017_graphs'
            self.data_fld = "D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis"
        else:
            fig_save_fld = '/Users/federicoclaudi/desktop'
            self.data_fld = "/Users/federicoclaudi/desktop"

        self.save_name = 'alltrials_pdf'
        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2], center=[.3, .7, .2],
                           central=[.3, .7, .2], shelter='c', threat='y')
        if not load:
            name = cohort_options['name']
            print(colored('Maze processing cohort {}'.format(name), 'green'))

            metad = cohort.Metadata[name]
            tracking_data = cohort.Tracking[name]
            self.from_trials_to_dataframe(metad, tracking_data, store_tracking)
        else:
            name = 'all_trials'
            self.load_dataframe()

        # self.plot_trajectories_at_roi(roi=['Threat_platform', 'Left_med_platform', 'Right_med_platform'])
        if store_tracking:
            self.plot_escape_trajectories()
        self.plot_status_stim()

        self.logreg_analaysis()

        self.plot_bootstrapped_distributions()

        save_all_open_figs(target_fld=fig_save_fld, name=name, format='svg')
        plt.show()
        a = 1


########################################################################################################################

    def probR_given(self):
        def calc_probs(data, ntrials):
            rcount_origin, rcount_xpos, rcount_orientation = 0, 0, 0
            count_xpos_given_origin, count_orientation_give_origin = 0, 0
            for trn in range(ntrials):
                tr = data.iloc[trn]
                ori = tr['atstim'][1]['Orientation']
                while ori > 360: ori -= 360
                if ori > 180: rcount_orientation += 1
                if 'right' in tr['origin'].lower():
                    rcount_origin += 1
                    if tr['atstim'][1]['adjusted x'] <= 0: count_xpos_given_origin += 1
                    if ori > 180: count_orientation_give_origin += 1
                if tr['atstim'][1]['adjusted x'] <= 0: rcount_xpos += 1
            print('p(R) given R origin: {}'.format(round(rcount_origin / ntrials, 2)))
            print('p(R) given X pos: {}'.format(round(rcount_xpos / ntrials, 2)))
            print('p(R) given orientation: {}'.format(round(rcount_orientation / ntrials, 2)))
            print('p(xpos) given origin {}'.format(round(count_xpos_given_origin / rcount_origin, 2)))
            print('p(xpos) given origin {}'.format(round(count_orientation_give_origin / rcount_origin, 2)))


        """ Calculate the probability of going right given a number of factors... """
        flipflop = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                     (self.triald_df['configuration'] != 'Left')]
        flipflop = flipflop[(flipflop['escape'] == 'Right_TtoM_platform') | (flipflop['escape'] == 'Right_TtoM_bridge')]
        ntrials_ff = len(flipflop)
        print('Processing FLIPFLOP maze, {} R trials'.format(ntrials_ff))
        calc_probs(flipflop, ntrials_ff)

        squared = self.triald_df.loc[(self.triald_df['experiment'] == 'Square Maze') |
                                     (self.triald_df['experiment'] == 'TwoAndahalf Maze')]
        squared = squared[squared['escape'] != 'Central_TtoX_bridge']
        squared = squared[squared['escape'] != 'Central_TtoX_bridge']
        squared = squared[squared['escape'] != 'Shelter_platform']
        ntrials_sq = len(squared)
        print('\nProcessing SQUARED maze, {} R trials'.format(ntrials_sq))
        calc_probs(squared, ntrials_sq)



    def from_trials_to_dataframe(self, metad, tracking_data, store_tracking):
        t = namedtuple('trial', 'name session origin escape stimulus experiment '
                                'configuration atstim tracking')
        data = []
        for trial in tracking_data.trials:
            outcome = trial.processing['Trial outcome']['trial_outcome']
            if not outcome:
                print(trial.name, ' no escape')
                continue
            else:
                tr_sess_id = int(trial.name.split('-')[0])
                for session in metad.sessions_in_cohort:
                    id = session[1].session_id
                    exp = None
                    if id == tr_sess_id:
                        exp = session[1].experiment
                        break
                if store_tracking:
                    d = t(trial.name, tr_sess_id, trial.processing['Trial outcome']['threat_origin_arm'],
                          trial.processing['Trial outcome']['threat_escape_arm'], trial.metadata['Stim type'], exp,
                          trial.processing['Trial outcome']['maze_configuration'],
                          trial.processing['status at stimulus'], trial)
                else:
                    d = t(trial.name, tr_sess_id, trial.processing['Trial outcome']['threat_origin_arm'],
                          trial.processing['Trial outcome']['threat_escape_arm'], trial.metadata['Stim type'], exp,
                          trial.processing['Trial outcome']['maze_configuration'],
                          trial.processing['status at stimulus'], None)
                data.append(d)
        self.triald_df = pd.DataFrame(data, columns=t._fields)

    def plot_trajectories_at_roi(self, roi='Threat'):
        """ plot the tracjectory (adjusted to shelter position) during the escape as the mouse crosses the rois
         listed"""
        if not isinstance(roi, list):
            roi = [roi]

        flipflop = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze')&
                                    (self.triald_df['configuration'] == 'Right')]
        f, axarr = plt.subplots(len(roi), 1, facecolor=[.1, .1, .1])
        for r_index, r in enumerate(roi):
            for tr in flipflop['tracking']:
                threat_loc = tr.processing['Trial outcome']['maze_rois']['Threat_platform']

                threat_loc = (threat_loc.topleft[0] + (threat_loc.bottomright[0] - threat_loc.topleft[0]) / 2,
                              threat_loc.topleft[1] + (threat_loc.bottomright[1] - threat_loc.topleft[1]) / 2)
                roi_trajectory = tr.processing['Trial outcome']['trial_rois_trajectory'][1800:]
                tracking = (np.subtract(tr.dlc_tracking['Posture']['body']['x'][1800:], threat_loc[0]),
                            np.subtract(tr.dlc_tracking['Posture']['body']['y'][1800:], threat_loc[1]))
                inroi = [i for i, e in enumerate(roi_trajectory) if r.lower() in e.lower()]
                if not inroi: continue
                axarr[r_index].plot(tracking[0].values[inroi], tracking[1].values[inroi], alpha=0.75)
                axarr[r_index].set(title=r, facecolor=[.2, .2, .2])

        a = 1

    def plot_escape_trajectories(self):
        """ plot the escape trajectories for L trials for ff and 2arms trials """
        #
        # flipflop = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
        #                               (self.triald_df['configuration'] != 'Right')]
        # flipflop = flipflop[(self.triald_df['escape'] == 'Right_TtoM_bridge') |
        #                     (self.triald_df['escape'] == 'Right_TtoF_bridge')]
        # flipflop2 = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
        #                               (self.triald_df['configuration'] != 'Left')]
        # flipflop2 = flipflop2[(self.triald_df['escape'] == 'Left_TtoM_bridge') |
        #                     (self.triald_df['escape'] == 'Left_TtoF_bridge')]

        f, ax = plt.subplots(1, 1, facecolor=[.1, .1, .1])
        ax.set(ylim=[100, -600], facecolor=[.2, .2, .2])

        sides = ['Left', 'Right']
        for i, s in enumerate(sides):
            for ii, ss in enumerate(sides):
                print('Conf {}, side {}'.format(s, ss))
                flipflop = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                              (self.triald_df['configuration'] == s)]
                flipflop = flipflop[(flipflop['escape'] == '{}_TtoM_bridge'.format(ss))]
                for tr in flipflop['tracking']:
                    print(tr)
                    threat_loc = tr.processing['Trial outcome']['maze_rois']['Threat_platform']

                    threat_loc = (threat_loc.topleft[0] + (threat_loc.bottomright[0] - threat_loc.topleft[0]) / 2,
                                  threat_loc.topleft[1] + (threat_loc.bottomright[1] - threat_loc.topleft[1]) / 2)
                    roi_trajectory = tr.processing['Trial outcome']['trial_rois_trajectory'][1800:]
                    tracking = (np.subtract(tr.dlc_tracking['Posture']['body']['x'][1800:], threat_loc[0]),
                                np.subtract(tr.dlc_tracking['Posture']['body']['y'][1800:], threat_loc[1]))
                    at_shelter = roi_trajectory.index('Shelter_platform')
                    ax.plot(np.subtract(tracking[0].values[:at_shelter], 750*i),
                            tracking[1].values[: at_shelter], alpha=0.75, color=[.4 * i, .6, .5*ii])

    def plot_bootstrapped_distributions(self, num_iters=50000, normalised=True, noise=True, replacement=True):
        def bootstrap(data, num_iters, num_samples =False, tag='Right', noise=noise, replacement=replacement):
            if not num_samples: num_samples = len(data)
            noise_std = 1 / (math.sqrt(num_samples)) ** 2
            res = []
            for _ in tqdm(range(num_iters)):
                if not replacement:
                    sel_trials = random.sample(data, num_samples)
                else:
                    sel_trials = [random.choice(data) for _ in data]
                    sel_trials = sel_trials[:num_samples]
                calculated_probability = len([b for b in sel_trials if tag in b]) / num_samples
                if noise:
                    res.append(calculated_probability+np.random.normal(scale=noise_std))
                else:
                    res.append(calculated_probability)
            return res

        def coin_simultor(num_samples=38, num_iters=50000):
            probs = []
            noise_std = 1 / (math.sqrt(num_samples)) ** 2
            for _ in tqdm(range(num_iters)):
                data = [random.randint(0, 1) for _ in range(num_samples)]
                prob_one = len([n for n in data if n == 1]) / len(data) + np.random.normal(scale=noise_std)
                probs.append(prob_one)
            return probs

        """ Calculate bootstrapped probability of going R on Two arms maze and Flip Flop VS and FLip Flop AS"""
        ff_vis = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                               (self.triald_df['stimulus'] == 'visual')]
        vis_flipflop_bs = bootstrap(ff_vis['escape'].values, num_iters)
        vis_flipflop_mean = len([e for e in ff_vis['escape'].values if 'Right' in e])/len(ff_vis['escape'].values)

        ff_aud = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                               (self.triald_df['stimulus'] == 'audio') & (self.triald_df['configuration'] == 'Right')]
        audio_flipflop_bs = bootstrap(ff_aud['escape'].values, num_iters)
        audio_flipflop_mean = len([e for e in ff_aud['escape'].values if 'Right' in e])/len(ff_aud['escape'].values)

        all_ff = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                    (self.triald_df['configuration'] != 'Left')]
        all_ff_bs = bootstrap(all_ff['escape'].values, num_iters)
        all_ff_mean = len([e for e in ff_vis['escape'].values if 'Right' in e]) / len(ff_vis['escape'].values)

        twoarm = self.triald_df.loc[(self.triald_df['experiment'] == 'PathInt2')]
        twoarms_bs = bootstrap(twoarm['escape'].values, num_iters)
        twoarms_flipflop_mean = len([e for e in twoarm['escape'].values if 'Right' in e])/len(twoarm['escape'].values)


        """ calculate welche's t test [independant paired t test for samples with different N and SD]:
                 https://www.scipy-lectures.org/packages/statistics/index.html   """
        welch_ffvis_ffaud = stats.ttest_ind([1 if 'Right' in e else 0 for e in ff_vis['escape'].values],
                                            [1 if 'Right' in e else 0 for e in ff_aud['escape'].values],
                                            equal_var=False)
        welch_ffvis_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in ff_vis['escape'].values],
                                              [1 if 'Right' in e else 0 for e in twoarm['escape'].values],
                                              equal_var=False)
        welch_ffaud_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in ff_aud['escape'].values],
                                              [1 if 'Right' in e else 0 for e in twoarm['escape'].values],
                                              equal_var=False)
        welch_allff_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in all_ff['escape'].values],
                                              [1 if 'Right' in e else 0 for e in twoarm['escape'].values],
                                              equal_var=False)

        """ calculate probs for squared mazes """
        squared = self.triald_df.loc[(self.triald_df['experiment'] == 'Square Maze') |
                                     (self.triald_df['experiment'] == 'TwoAndahalf Maze')]
        squared_bs = bootstrap(squared['escape'].values, num_iters)
        squared_mean = len([e for e in squared['escape'].values if 'Right' in e]) / len(squared['escape'].values)
        coin_prob = coin_simultor(num_samples=40)

        """ PLOT """
        binz=60
        f, axarr = plt.subplots(2, 1, facecolor=[0.1, 0.1, 0.1])
        ax, ax2 = axarr[0], axarr[1]
        self.histogram_plotter(ax, audio_flipflop_bs, color=self.colors['right'], label='FF_aud',
                               bins=binz, normalised=normalised, alpha=.5)
        self.histogram_plotter(ax, vis_flipflop_bs, color=self.colors['left'], label='FF_vis',
                               bins=binz, normalised=normalised, alpha=.5)
        self.histogram_plotter(ax, twoarms_bs, color=self.colors['central'], label='2A_vis',
                               bins=binz,  normalised=normalised, alpha=.5)
        self.histogram_plotter(ax, all_ff_bs, color=[.6, .6, .6], label='all_FFs',
                               bins=binz,  normalised=normalised, alpha=.5)

        ax.axvline(vis_flipflop_mean, linewidth=2, color=self.colors['left'], alpha=.5)
        ax.axvline(audio_flipflop_mean, linewidth=2, color=self.colors['right'], alpha=.5)
        ax.axvline(twoarms_flipflop_mean, linewidth=2, color=self.colors['central'], alpha=.5)
        ax.axvline(all_ff_mean, linewidth=2, color=self.colors['central'], alpha=.5)

        self.histogram_plotter(ax2, squared_bs, color=[.7, .7, .7], label='Squared',
                               bins=binz, normalised=normalised, alpha=.5)
        self.histogram_plotter(ax2, coin_prob, color=[.2, .4, .2], label='coin',
                               bins=binz, normalised=normalised, alpha=.5)
        ax2.axvline(squared_mean, linewidth=2, color=[.7, .7, .7], alpha=.5)

        ax.set(facecolor=[0.2, 0.2, 0.2], xlim=[0, 1], xticks=np.arange(0, 1+0.1, 0.1))
        ax2.set(facecolor=[0.2, 0.2, 0.2], xlim=[0, 1], xticks=np.arange(0, 1+0.1, 0.1))
        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)
        make_legend(ax2, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        a = 1

    def plot_status_stim(self):
        ff_vis = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                    (self.triald_df['stimulus'] == 'visual')]
        ff_vis_ntrials = len(ff_vis)

        ff_aud = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                    (self.triald_df['stimulus'] == 'audio') &
                                    (self.triald_df['configuration'] == 'Right')]
        ff_aud_ntrials = len(ff_aud)

        twoarm = self.triald_df.loc[(self.triald_df['experiment'] == 'PathInt2')]
        twoarm_ntrials = len(twoarm)

        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2], center=[.3, .7, .2],
                           central=[.3, .7, .2], shelter='c', threat='y')
        colors = dict(
            Right_TtoM_platform=[.7, .3, .2],
            Right_TtoM_bridge=[.7, .3, .2],
            Left_TtoM_bridge=[.2, .3, .7],
            Left_TtoF_bridge=[.2, .3, .7]
        )

        datas = [ff_vis, ff_aud, twoarm]
        nn = [ff_vis_ntrials, ff_aud_ntrials, twoarm_ntrials]
        colorby = ['origin', 'origin', 'origin', 'escape', 'escape', 'escape',
                   'Orientation',  'Orientation',  'Orientation']
        f, axarr = plt.subplots(3, 3, facecolor=[0.1, 0.1, 0.1])
        axarr = axarr.flatten()
        for i, d in enumerate(datas):
            for ii, var in enumerate(colorby):
                var = colorby[i]
                if var == 'Orientation':
                    try:
                        data = np.asarray([s[1][var] for s in d['status']]).reshape(-1, 1)
                        while np.any(data[data > 360]):
                            data[data > 360] -= 360
                    except: continue
                else:
                    data = np.asarray([s for s in d[var]]).reshape(-1, 1)

                try:
                    xpos = np.asarray([s[1]['adjusted x'] for s in d['atstim']])[:]
                    ypos = np.asarray([s[1]['adjusted y'] for s in d['atstim']])[:]
                except:
                    xpos = np.asarray([s[1]['x'] for s in d['atstim']])[:]
                    ypos = np.asarray([s[1]['y'] for s in d['atstim']])[:]
                col = np.asarray([colors[o] for o in d[var].values])

                axarr[i].scatter(xpos, ypos, s=25, c=col)
                axarr[i].set(title='Colored by {}'.format(var), facecolor=[.2, .2, .2], xlim=[-50, 50], ylim=[400, 600])

        a = 1

    @staticmethod
    def histogram_plotter(ax, data, color='g', bins=50, label=None, normalised=False, alpha=0.75):
        if normalised:
            results, edges = np.histogram(data, bins=bins, normed=True)
            binWidth = edges[1] - edges[0]
            ax.bar(edges[:-1], results * binWidth, binWidth, color=color, label=label, alpha=alpha)
        else:
            ax.hist(data, color=color, label=label, alpha=alpha)

    def save_dataframe(self):
        import dill as pickle
        with open(os.path.join(self.data_fld, self.save_name), "wb") as dill_file:
            pickle.dump(self.triald_df, dill_file)

    def load_dataframe(self):
        self.triald_df = pd.read_pickle(os.path.join(self.data_fld, self.save_name))

    def logreg_analaysis(self):
        """ pool the data from squared mazes for non Center escape trials and use these data to train the
            logistic regression model """

        squared = self.triald_df.loc[(self.triald_df['experiment'] == 'Square Maze') |
                                     (self.triald_df['experiment'] == 'TwoAndahalf Maze')]
        squared = squared[squared['escape'] != 'Central_TtoX_bridge']
        squared = squared[squared['escape'] != 'Central_TtoX_bridge']
        squared = squared[squared['escape'] != 'Shelter_platform']

        ntrials = len(squared)
        variables = ['adjusted x', 'adjusted y', 'Orientation']
        esc = squared['escape'].values.reshape(-1, 1)
        f, axarr = plt.subplots(len(variables), 1, facecolor=[.1, .1, .1])

        for i, var in enumerate(variables):
            data = np.asarray([s[1][var] for s in squared['atstim']]).reshape(-1, 1)
            if var == 'Orientation':
                while np.any(data[data > 360]):
                    data[data > 360] -= 360

            # Split dataset into training and test datasets
            x_train, x_test, y_train, y_test = train_test_split(data, esc, test_size=0.25, random_state=0)

            # Create the mode, fit it to the test data and use the trained model to make predictions on the test set
            logisticRegr = LogisticRegression()
            logisticRegr.fit(x_train, y_train)
            predictions = logisticRegr.predict(x_test)
            predictions_probabilities = logisticRegr.predict_proba(x_test)

            axarr[i].plot(x_test, predictions_probabilities[:, 0])
            axarr[i].plot(x_test, predictions_probabilities[:, 1])
            axarr[i].set(title=var, facecolor=[.2, .2, .2])

        a = 1

        # Measure score and confusion matrix, plot confusion matrix
        # score = logisticRegr.score(x_test, y_test)
        # cm = metrics.confusion_matrix(y_test, predictions)
        #
        # plt.figure(figsize=(9, 9))
        # sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        # plt.ylabel('Actual label')
        # plt.xlabel('Predicted label')
        # all_sample_title = 'Accuracy Score: {0}'.format(score)
        # plt.title(all_sample_title, size=15)


if __name__ == "__main__":
    MazeCohortProcessor(store_tracking=False, load=True)


