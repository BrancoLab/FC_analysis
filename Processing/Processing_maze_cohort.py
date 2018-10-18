from Utils.imports import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('axes', edgecolor=[0.8, 0.8, 0.8])
params = {'legend.fontsize': 16,
          'legend.handlelength': 2}
col = [0, 0, 0]
# col = [.8, .8, .8]
matplotlib.rc('axes', edgecolor=col)
matplotlib.rcParams['text.color'] = [.8, .8, .8]
matplotlib.rcParams['axes.labelcolor']  =[.8, .8, .8]
matplotlib.rcParams['axes.labelcolor'] = [.8, .8, .8]
matplotlib.rcParams['xtick.color'] = [.8, .8, .8]
matplotlib.rcParams['ytick.color'] = [.8, .8, .8]

plt.rcParams.update(params)

import platform
import math
import random
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from Plotting.Plotting_utils import make_legend, save_all_open_figs, create_figure, show, ticksrange
from Utils.Data_rearrange_funcs import flatten_list
from Utils.maths import line_smoother

from Config import cohort_options

arms_colors = dict(left=(255, 0, 0), central=(0, 255, 0), right=(0, 0, 255), shelter=(200, 180, 0),
                        threat=(0, 180, 200))


class MazeCohortProcessor:
    def __init__(self, cohort=None, store_tracking=True, load=False):

        if 'Windows' in platform.system():
            fig_save_fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\181017_graphs'
            self.data_fld = "D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis"
        else:
            fig_save_fld = '/Users/federicoclaudi/desktop'
            self.data_fld = "/Users/federicoclaudi/desktop"

        self.save_name = 'alltrials_pdf'
        self.colors = dict(left=[.2, .3, .7], right=[.7, .3, .2], centre=[.3, .7, .2], center=[.3, .7, .2],
                           central=[.3, .7, .2], shelter='c', threat='y')

        self.load = load
        if not load:
            name = cohort_options['name']
            print(colored('Maze processing cohort {}'.format(name), 'green'))

            metad = cohort.Metadata[name]
            tracking_data = cohort.Tracking[name]
            self.from_trials_to_dataframe(metad, tracking_data, store_tracking)
            self.adjust_xypos(save=False)
        else:
            name = 'all_trials'
            self.load_dataframe()

        self.for_labmeeting()

        save_all_open_figs(target_fld=fig_save_fld, name=name, format='svg')
        plt.show()
        a = 1

########################################################################################################################
    ###  DATA SEGMENTATION FUNCTIONS ####
########################################################################################################################
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

    def get_flipflop_data(self, only_r_escapes=False):
        flipflop_right = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                      (self.triald_df['configuration'] != 'Left')]
        if only_r_escapes:
            flipflop_right = flipflop_right[(flipflop_right['escape'] == 'Right_TtoM_platform') |
                                            (flipflop_right['escape'] == 'Right_TtoM_bridge')]
        ntrials_ff_right = len(flipflop_right)

        flipflop_left = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                      (self.triald_df['configuration'] == 'Left')]
        if only_r_escapes:
            flipflop_left = flipflop_left[(flipflop_left['escape'] == 'Right_TtoM_platform') |
                                            (flipflop_left['escape'] == 'Right_TtoM_bridge')]
        ntrials_ff_left = len(flipflop_left)

        return flipflop_right, ntrials_ff_right, flipflop_left, ntrials_ff_left

    def get_square_data(self, exclude_twoandhalf=False, only_r_escapes=False):
        if exclude_twoandhalf:
            squared = self.triald_df.loc[(self.triald_df['experiment'] == 'Square Maze')]
        else:
            squared = self.triald_df.loc[(self.triald_df['experiment'] == 'Square Maze') |
                                         (self.triald_df['experiment'] == 'TwoAndahalf Maze')]
        if only_r_escapes:
            squared = squared[squared['escape'] != 'Central_TtoX_bridge']
            squared = squared[squared['escape'] != 'Central_TtoX_bridge']
            squared = squared[squared['escape'] != 'Shelter_platform']
        ntrials_sq = len(squared)
        return squared, ntrials_sq

    def get_twoarms_data(self, only_r_escapes=False):
        twoarm = self.triald_df.loc[(self.triald_df['experiment'] == 'PathInt2')]
        if only_r_escapes:
            twoarm = twoarm[(twoarm['escape'] == 'Right_TtoM_bridge')|
                            (twoarm['escape'] == 'Right_TtoM_bridge')]
        n_trials_2a = len(twoarm)
        return  twoarm, n_trials_2a

    def get_threearm_data(self, only_r_escapes=False):
        threearm = self.triald_df.loc[(self.triald_df['experiment'] == 'PathInt')]
        if only_r_escapes:
            threearm = threearm[(threearm['escape'] == 'Right_TtoM_bridge')|
                            (threearm['escape'] == 'Right_TtoM_bridge')]
        n_trials_2a = len(threearm)
        return  threearm, n_trials_2a

    @staticmethod
    def get_probability_bymouse(data):
        mice = set(data['session'])
        probs, ntrials = [], []
        for mouse in mice:
            mouse_escapes = data[data['session'] == mouse]
            n_escapes = len(mouse_escapes)
            n_r_escapes = len([e for e in mouse_escapes['escape'].values if 'Right' in e])
            ntrials.append(n_escapes)
            probs.append(round(n_r_escapes/n_escapes, 2))
        return probs, ntrials

    def adjust_xypos(self, save=False):
        for n in range(len(self.triald_df)):
            tr = self.triald_df.iloc[n]
            threat_loc = tr['tracking'].processing['Trial outcome']['maze_rois']['Threat_platform']
            threat_loc = (threat_loc.topleft[0] + (threat_loc.bottomright[0] - threat_loc.topleft[0]) / 2,
                          threat_loc.topleft[1] + (threat_loc.bottomright[1] - threat_loc.topleft[1]) / 2)

            p = (tr['tracking'].processing['status at stimulus'][1]['x'],
                 tr['tracking'].processing['status at stimulus'][1]['y'])
            adjp = (p[0]-threat_loc[0], p[1]-threat_loc[1])

            tr['atstim'][1]['adjusted x'] = adjp[0]
            tr['atstim'][1]['adjusted y'] = adjp[1]
        if save:
            self.triald_df['tracking'] = None
            self.save_dataframe()

########################################################################################################################
    ###  FOR PRESENTATION ####
########################################################################################################################
    def ntrials_perexp(self):
        # Calculate n trials per experiment
        number_of_trials, number_of_trial_per_mouse = {}, {}
        for exp in self.experiments:
            ntr = []
            sess = self.data_byexp[exp]['session']
            for s in set(sess.values):
                if s is None: continue
                calculated = len(sess[self.data_byexp[exp]['session'] == s])
                if calculated: ntr.append(calculated)
            number_of_trials[exp] = np.sum(ntr)
            number_of_trial_per_mouse[exp] = ntr
        return number_of_trials, number_of_trial_per_mouse

    def pR_perexp(self, contingency_tables, number_of_trials):
        # Store p(R) for each experiment and calculate the binomial distribution C.I.:
        # https://www.thomasjpfan.com/2015/08/statistical-power-of-coin-flips/
        print('\n\n\nCalculating C.I.')
        alpha = 0.05
        pr_by_exp = {}
        for exp in self.experiments:
            pr = contingency_tables[exp]['All']['right']
            pr_by_exp[exp] = pr

            ntr = number_of_trials[exp]
            fair_interval = stats.binom.interval(1 - alpha, ntr, 0.5)
            print(exp, fair_interval, ' -- ', round(pr, 2))
        return pr_by_exp

    def pR_permouse_inexp(self):
        # Calculate p(R) for each mouse in each experiment
        prob_right_bymouse = {}
        for exp in self.experiments:
            probs = []
            sessions = set(self.data_byexp[exp]['session'].values)
            for s in sessions:
                if s is None: continue
                if np.isnan(s): continue
                exp_data = self.data_byexp[exp][self.data_byexp[exp]['session'] == s]
                cont_table = pd.crosstab(exp_data['escape'], exp_data['origin'], margins=True, normalize='all')
                if 'right' in cont_table['All'].keys():
                    probs.append(round(cont_table['All']['right'], 2))
                else:
                    probs.append(0)
            prob_right_bymouse[exp] = probs
        return prob_right_bymouse

    def pR_byXpos_perexp(self):
        # Calculate p(R) as a factor of X position
        binned_pr, binz = {}, {}
        for exp in self.experiments:
            positions_outcomes = []
            for s in range(len(self.data_byexp[exp]['atstim'].values)):
                pos = self.data_byexp[exp]['atstim'].iloc[s]
                out = self.data_byexp[exp]['escape'].iloc[s]
                try:
                    if np.isnan(out): continue
                except:
                    pass
                if pos is None: continue
                positions_outcomes.append((pos[0], out))

            all_pos = [p[0] for p in positions_outcomes]
            resc_pos = [p[0] for p in positions_outcomes if p[1] == 'right']

            nbinz = 5
            binned_all, bins = np.histogram(sorted(all_pos), bins=nbinz)
            binned_resc = np.histogram(sorted(resc_pos), bins=nbinz)[0]
            binned_prob = np.divide(binned_resc, binned_all)
            binned_pr[exp] = binned_prob
            binz[exp] = bins
        return binned_pr, binz

    def contingencies_bystim_perexp(self):
        # Show contingencies tables for stimulus vs escape arm
        res = {}
        for exp in self.experiments:
            print('='*60)
            print('='*60)

            cont_table = pd.crosstab(self.data_byexp[exp]['escape'], self.data_byexp[exp]['stimulus'], margins=True,
                                     normalize='columns')
            print('\n\n\n\nCont. table {}\n'.format(exp), cont_table)
            cont_table = pd.crosstab(self.data_byexp[exp]['escape'], self.data_byexp[exp]['stimulus'], margins=False)
            print('\n', cont_table)

            print('\n\n\n')
            print('='*60)
            print('='*60)
            res[exp] = cont_table
        return res

    def chisq_onproportions(self, exp_to_compare=None, data=None):
        # Calculate chi-squared test for proportions
        # https://sites.ualberta.ca/~lkgray/uploads/7/3/6/2/7362679/slides_-_binomialproportionaltests.pdf
        # create a dataframe with exp id on one column and
        print('\n\n\nChi2 testing ---')
        dff = []
        if exp_to_compare is None: exp_to_compare = ['square', 'flipflop']
        if data is None: data = self.data_byexp
        for exp in exp_to_compare:
            good_escapes = [e for e in data[exp]['escape'].values if e is not None]
            good_escapes = [e for e in good_escapes if str(e) != 'nan']
            good_experiments = [exp for _ in good_escapes]
            dic = {
                'experiment': good_experiments,
                'escape': good_escapes,
            }
            df = pd.DataFrame.from_dict(dic, orient='index').transpose()
            dff.append(df)
        exp_outcome_df = pd.concat(dff)

        obs = pd.crosstab(exp_outcome_df['escape'], exp_outcome_df['experiment'])
        print('\n', obs, '\n')
        stat, p, dof, expected = stats.chi2_contingency(obs)
        # interpret test-statistic
        prob = 0.95
        critical = stats.chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            print('Dependent (reject H0) -- chi: {} -- p: {} -- '.format(stat, p))
        else:
            print('Independent (fail to reject H0) -- chi: {} -- p: {} -- '.format(stat, p))

        # Do a pairwise chi square test with bonferroni correction
        if len(exp_to_compare) > 2:
            dummies = pd.get_dummies(exp_outcome_df['experiment'])
            adjusted_p = 0.05 / len(exp_to_compare)

            for series in dummies:
                crosstab = pd.crosstab(dummies[series], exp_outcome_df['escape'])
                print('\n\n', crosstab)
                chi2, p, dof, expected = stats.chi2_contingency(crosstab)
                print('Chi2: {}, p: {}, DoF: {}'.format(chi2, p, dof))
                if p < adjusted_p:
                    print('Significant')
                else:
                    print('Not significant')

    def get_escape_durations(self):
        # loop over each trial in each experiment
        durations = namedtuple('d', 'left right centre')
        durs_byexp = {}
        for exp in self.experiments:
            d = self.data_byexp[exp]
            durs = durations([], [], [])
            for n in range(len(d)):
                trial = d.iloc[n]
                if trial['escape'] is None or str(trial['escape']) == 'nan': continue
                else:
                    # calculate escape duration
                    rois = trial['tracking'].processing['Trial outcome']['trial_rois_trajectory'][1800:]
                    atshelter = rois.index('Shelter_platform')
                    if atshelter < 10: continue  # there was an error
                    if trial['escape'] == 'right':
                        durs[1].append(atshelter)
                    elif trial['escape'] == 'centre':
                        durs[2].append(atshelter)
                    else:
                        durs[0].append(atshelter)

            # There should be no escape that lasts more than 9 seconds
            if np.max(durs[0])>9*30 or np.max(durs[1])>9*30:
                raise Warning('sneaky sneak')

            durs_byexp[exp] = durs

            all_durs = [d for d in durs.left] + [d for d in durs.right]+ [d for d in durs.centre]
            print('\n\nMedian escape duration for exp {}: {} -- L escape: {} -- R escape: {}'.format(
                exp, np.median(all_durs), np.median(durs.left), np.median(durs.right) ))
        return durs_byexp

    def for_labmeeting(self):
        cols = dict(
            flipflop=[233/250, 150/255, 122/255],
            twoarms=[173/255, 216/250, 230/255],
            square=[216/255, 191/255, 216/255],
            threearms=[147 / 255, 112 / 255, 119 / 255],
            coin=[1.0, 1.0, 1.0]
        )
        fcol = [.2, .2, .2]

        """
        ADD: relevant plots for 3arms maze (e.g. duration of escape on short arm)
        P(R) as a factor of X position... 
        """

        """ set up data and calc contingency tables """
        # get data
        ff_r, ntr_ff_r, _, _ = self.get_flipflop_data()
        ta, ntr_ta = self.get_twoarms_data()
        sq, ntr_sq = self.get_square_data()
        threea, ntr_threea = self.get_threearm_data()

        # Calculate contingencies tables for each experiment + create multi-layer dataframe
        experiments = ['square', 'flipflop', 'twoarms', 'threearms']
        datas = [sq, ff_r, ta, threea]
        contingency_tables, contingency_tables_nomargins = {}, {}
        dff = []
        for exp, dat in zip(experiments, datas):
            # Preperare data to restructure in data frame
            temp_escs = [(i, 'left') if 'Left' in e else (i, 'right') if 'Right' in e
                        else (i, 'centre') if 'Central' in e else (None, None)
                        for i, e in enumerate(dat['escape'].values)]
            included_indexes = [e[0] for e in temp_escs]
            escs = [e[1] for e in temp_escs]
            oris = ['left' if 'Left' in e else 'right' if 'Right' in e
                    else 'centre' if 'Central' in e else None for e in dat['origin'].values]
            n = [n if i in included_indexes else None for i, n in enumerate(dat['name'].values) ]
            sess = [n if i in included_indexes else None for i, n in enumerate(dat['session'].values)]
            stims = [n if i in included_indexes else None for i, n in enumerate(dat['stimulus'].values)]
            exps = [n if i in included_indexes else None for i, n in enumerate(dat['experiment'].values)]
            confs = [n if i in included_indexes else None for i, n in enumerate(dat['configuration'].values)]
            atstims = [(round(n[1]['adjusted x'],2), round(n[1]['adjusted y'],2)) if i in included_indexes else None
                       for i, n in enumerate(dat['atstim'].values)]
            trackings = [t if i in included_indexes else None for i, t in enumerate(dat['tracking'].values)]

            dic = {
                'origin': oris,
                'escape': escs,
                'name': n,
                'session': sess,
                'stimulus': stims,
                'experiment': exps,
                'configuration': confs,
                'atstim': atstims,
                'tracking': trackings
            }
            # Store the results, we will concatenate the dataframes out of the loop
            df = pd.DataFrame.from_dict(dic, orient='index').transpose()
            dff.append(df)

            # Print out the normalised cont. table
            cont_table = pd.crosstab(df['escape'], df['origin'], margins=True, normalize='columns')
            nm_cont_table = pd.crosstab(df['escape'], df['origin'], margins=False)
            contingency_tables[exp] = cont_table
            contingency_tables_nomargins[exp] = nm_cont_table
            print('\n', 'Norm. contingency table for {}\n\n'.format(exp), cont_table)

        # Plot p(R) given arm of origin
        f, axarr = create_figure(ncols=len(experiments))
        for i, exp in enumerate(experiments):
            axarr[i].set(facecolor=fcol, ylim=[0, 1])
            ct = contingency_tables[exp]
            pp = ct.iloc[1]
            axarr[i].bar(np.linspace(0, len(pp), len(pp)), [p for p in pp], color=cols[exp])



            ct = contingency_tables_nomargins[exp]
            stat, p, dof, expected = stats.chi2_contingency(ct)
            # interpret test-statistic
            prob = 0.95
            critical = stats.chi2.ppf(prob, dof)
            if abs(stat) >= critical:
                print('EXP {} -- Dependent (reject H0) -- chi: {} -- p: {} -- '.format(exp, stat, p))
            else:
                print('EXP {} -- Independent (fail to reject H0) -- chi: {} -- p: {} -- '.format(exp, stat, p))

        data_byexp = pd.concat(dff, axis=1, keys=experiments)  # <-- all data for good trials organised by experiment

        self.data_byexp = data_byexp
        self.experiments = experiments

        """ if we loaded the whole dataset (inc.tracking),
            do analysis on the exploration [TODO] and on the  escape duration  """
        if not self.load:
            escapedurs_byxp = self.get_escape_durations()

        """ Calculte a whole bunch of other stuff """
        # Calculate n trials per experiment
        number_of_trials, number_of_trial_per_mouse = self.ntrials_perexp()

        # Calculate n trials per mouse for all experiments (this includes "bad" trials)
        sessions = set(self.triald_df['session'])
        ntr_all = [len(self.triald_df[self.triald_df['session'] == s]) for s in sessions]

        # Store p(R) for each experiment and calculate the binomial distribution C.I.:  # TODO fix binomial range
        # https://www.thomasjpfan.com/2015/08/statistical-power-of-coin-flips/
        pR_byexp = self. pR_perexp(contingency_tables, number_of_trials)

        # Calculate p(R) as a factor of X position
        pR_byXpos_perexp, pr_Xpos_binz = self.pR_byXpos_perexp()

        # Calculate p(R) for each mouse in each experiment
        pR_bymouse_perexp = self.pR_permouse_inexp()

        # Show contingencies tables for p(R) by stim
        cont_byexp = self.contingencies_bystim_perexp()
        stat, p, dof, expected = stats.chi2_contingency(cont_byexp['flipflop'])
        # interpret test-statistic
        prob = 0.95
        critical = stats.chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            print('Dependent (reject H0) -- chi: {} -- p: {} -- '.format(stat, p))
        else:
            print('Independent (fail to reject H0) -- chi: {} -- p: {} -- '.format(stat, p))

        f, ax = create_figure()
        ax.bar(0, 0.821, color=np.subtract(cols['flipflop'], .2))
        ax.bar(0.8, 0.771, color=cols['flipflop'])
        ax.set(facecolor=[.2, .2, .2], ylim=[0,1])

        # Calculate chi-squared test for proportions
        self.chisq_onproportions()

        """ NOW PLOT THE RESULTS  """
        # Plot duration of escape [scatter + boxplot]
        if not self.load:
            f, axarr = create_figure(ncols=2)
            for i, exp in enumerate(experiments):
                esd = [escapedurs_byxp[exp].right, escapedurs_byxp[exp].left, escapedurs_byxp[exp].centre]
                for idx, e in enumerate(esd):
                    axarr[0].scatter([0.75 * i - 0.25 * idx + np.random.normal(scale=0.025) for _ in e],
                                     np.divide(e, 30),
                                     color=np.subtract(cols[exp], .2 * idx), s=75, alpha=0.5)

            lbls = ['square', 'square', 'square',
                    'flipflop', 'flipflop', 'flipflop',
                    'twoarms', 'twoarms', 'twoarms',
                    'threearms', 'threearms', 'threearms']
            all_durations = []
            for exp in experiments:
                all_durations.append(np.divide(escapedurs_byxp[exp].left, 30))
                all_durations.append(np.divide(escapedurs_byxp[exp].centre, 30))
                all_durations.append(np.divide(escapedurs_byxp[exp].right, 30))

            bplot = axarr[1].boxplot(all_durations,
                                      vert=True,  # vertical box alignment
                                      patch_artist=True,  # fill with color
                                      labels=lbls)  # will be used to label x-ticks

            for patch, colname in zip(bplot['boxes'], lbls):
                patch.set_facecolor(cols[colname])
            for ax in axarr: ax.set(facecolor=fcol)

            # Plot Velocoties aligned to stim onset and aligned to detection onset
            self.velocity_plotter(cols)

        # Plot # trial per mouse by experiment
        f, axarr = create_figure(ncols=2)
        for i, exp in enumerate(experiments):
            tr = number_of_trial_per_mouse[exp]
            axarr[0].scatter([0.2 * i + np.random.normal(scale=0.02) for _ in tr], tr,
                             color=cols[exp], s=500, alpha=0.5)
        bplot = axarr[1].boxplot(number_of_trial_per_mouse.values(),
                                 vert=True,  # vertical box alignment
                                 patch_artist=True,  # fill with color
                                 labels=experiments)  # will be used to label x-ticks
        for patch, colname in zip(bplot['boxes'], experiments):
            patch.set_facecolor(cols[colname])
        for ax in axarr: ax.set(facecolor=fcol)

        # plot p(R) by experiment
        f, ax = create_figure()
        for i, exp in enumerate(experiments):
            ax.bar(i-0.1*i, pR_byexp[exp], color=cols[exp])
        ax.set(facecolor=fcol, xticks=[],  yticks=ticksrange(0, 1, .1))

        # plot p(R) by mouse
        f, axarr = create_figure(nrows=len(experiments))
        for i, exp in enumerate(experiments):
            pr = pR_bymouse_perexp[exp]
            axarr[i].bar(np.linspace(0, len(pr), len(pr)), sorted(pr), color=cols[exp])
            axarr[i].set(facecolor=fcol, xticks=[], yticks=ticksrange(0, 1, .1))

        # plot p(R) by x position
        f, axarr = create_figure(nrows=len(experiments))
        for i, exp in enumerate(experiments):
            pr = pR_byXpos_perexp[exp]
            axarr[i].axhline(pR_byexp[exp], color='w', ls=':')
            axarr[i].bar(pr_Xpos_binz[exp][0:-1][::-1], pr, color=cols[exp], width=10)
            axarr[i].set(facecolor=fcol, xticks=ticksrange(np.max(pr_Xpos_binz[exp]), np.min(pr_Xpos_binz[exp]), 10),
                         yticks=ticksrange(0, 1, .1))

        # self.plot_escape_trajectories(cols)

        a = 1
        show()

########################################################################################################################
    ###  PROBABILITY FUNCTIONS ####
########################################################################################################################
    @staticmethod
    def bootstrap(data, num_iters, num_samples=False, tag='Right', noise=True, replacement=True):
        if not num_samples: num_samples = len(data)
        noise_std = 1 / (math.sqrt(num_samples)) ** 2
        res = []
        for i in tqdm(range(num_iters)):
            if not replacement:
                sel_trials = random.sample(data, num_samples)
            else:
                sel_trials = [random.choice(data) for _ in data]
                sel_trials = sel_trials[:num_samples]
            calculated_probability = len([b for b in sel_trials if tag in b]) / num_samples
            if noise:
                res.append(calculated_probability + np.random.normal(scale=noise_std))
            else:
                res.append(calculated_probability)
        return res

    @staticmethod
    def coin_simultor(num_samples=38, num_iters=50000, noise=True):
        if not num_samples: num_samples = 100
        probs = []
        noise_std = 1 / (math.sqrt(num_samples)) ** 2
        for _ in tqdm(range(num_iters)):
            data = [random.randint(0, 1) for _ in range(num_samples)]
            if noise:
                prob_one = len([n for n in data if n == 1]) / len(data) + np.random.normal(scale=noise_std)
            else:
                prob_one = len([n for n in data if n == 1]) / len(data)
            probs.append(prob_one)
        return probs

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

########################################################################################################################
###  PLOTTING FUNCTIONS ####
########################################################################################################################
    @staticmethod
    def histogram_plotter(ax, data, color='g', just_outline=False, bins=50, label=None, normalised=False, alpha=0.75):
        if just_outline:
            edgecol = color
            color = (0.2, 0.2, 0.1, 1)
            alpha=0.25
        else:
            edgecol = None
        lwidth = 1.25

        if normalised:
            results, edges = np.histogram(data, bins=bins, normed=True)
            binWidth = edges[1] - edges[0]
            ax.bar(edges[:-1], results * binWidth, binWidth, edgecolor=edgecol, fc=color,
                   label=label, alpha=alpha, lw=lwidth)
        else:
            ax.hist(data,  edgecolor=edgecol, fc=color, bins=bins, label=label, alpha=alpha, lw=lwidth)

    @staticmethod
    def scatter_plotter(ax, data, key, cols, coltag=None):
        c1, c2, c3 = [.98, .98, .98], [.5, .5, .5], [.65, .65, .65]
        cc = [c3, c2, c1]

        if not 'Orientation' in key:
            col_keys = list(set([string.split('_')[0] for string in data[key]
                                 if not 'Central' in string and not 'Threat' in string and not 'Shelter' in string]))
            keys = set(data[key])
            for i, k in enumerate(keys):
                if 'Central' in k or 'Threat' in k or 'Shelter' in k: continue
                col_idx = col_keys.index(k.split('_')[0])
                d = data[data[key]==k]
                pos = [(s.status['adjusted x'], s.status['adjusted y']) for s in d['atstim']]
                if col_keys[col_idx] =='Right':
                    color = cols[coltag]
                else:
                    color = cc[col_idx]
                ax.scatter([p[0] for p in pos], [p[1] for p in pos], s=35, alpha=.5, color=color,
                           label=col_keys[col_idx])
            ylim = [50, -75]
        else:
            d = data[(data['escape'] == 'Right_TtoM_bridge') | (data['escape'] == 'Right_TtoM_platform')]
            pos = [(s.status['adjusted x'], s.status['adjusted y']) for s in d['atstim']]
            oris = np.asarray([s[1]['Orientation'] for s in d['atstim']])
            while np.any(oris[oris>360]): oris[oris>360] -= 360
            norm = matplotlib.colors.Normalize(vmin=0, vmax=360)
            sc = ax.scatter([p[0] for p in pos], [p[1] for p in pos], s=35, alpha=.5, norm=norm, c=oris, cmap='bwr')
            plt.colorbar(sc, orientation='horizontal', ax=ax)
            ylim = [50, -65]

        ax.set(title='Colored by {}'.format(key), facecolor=[.2, .2, .2], xlim=[-65, 65], ylim=ylim,
               xlabel='adjusted x pos', ylabel='adjusted y pos')
        make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

    def velocity_plotter(self, cols):
        # Get all the sessions for which we have the detection time
        filepath = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis\\det_times.txt'
        with open(filepath, 'r') as f:
            sessions, timeshift = [], {}
            for line in f:
                sessions.append(int(line.split(' ')[1].split('_')[0]))
                timeshift[line.split(' ')[4]] = int(line.split(' ')[-1])
            sessions = set(sessions)

        f, axarr = create_figure(ncols=3, nrows=2)
        axarr = axarr.flatten()
        for ax in axarr:
            ax.axvline(15, color='w', linewidth=1.5)
            ax.set(facecolor=[.2, .2, .2])

        exps = ['twoarms', 'flipflop']
        wnd = [1785, 1940]
        for exp in exps:
            d = self.data_byexp[exp]
            container = [[], [], []]
            shifted_cont = [[], [], []]
            for n in range(len(d)):
                tr = d.iloc[n]
                if tr['origin'] is None or str(tr['origin']) == 'nan': continue
                if tr.session not in sessions: continue

                velocity = tr.tracking.dlc_tracking['Posture']['body']['Velocity'].values
                blength = tr.tracking.dlc_tracking['Posture']['body']['Body length'].values
                angvel = np.abs(line_smoother(tr.tracking.dlc_tracking['Posture']['body']['Body ang vel'].values,
                                              order=10, window_size=25))

                bl_atstim = blength[1800]
                velocity = np.multiply(np.divide(velocity, bl_atstim), 30)  # convert to bl/sec
                blength = np.divide(blength, bl_atstim)

                varz = [velocity,blength, angvel]
                adjusted_varz = []
                for i, v in enumerate(varz):
                    while len(v)<3600: v = np.append(v, 0)
                    container[i].append(v)
                    adjusted_varz.append(v)

                axarr[0].plot(velocity[wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)
                axarr[1].plot(blength[wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)
                axarr[2].plot(angvel[wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)

                try:
                    delay = timeshift[tr['name']]
                except:
                    pass
                adjusted = []
                for i, v in enumerate(adjusted_varz):
                    v = v[delay-1800-3:]
                    v = np.append(v, np.zeros(delay))
                    shifted_cont[i].append(v)
                    adjusted.append(v)

                axarr[3].plot(adjusted[0][wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)
                axarr[4].plot(adjusted[1][wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)
                axarr[5].plot(adjusted[2][wnd[0]:wnd[1]], color=np.subtract(cols[exp], .25), linewidth=0.75, alpha=.25)

            for i,c in enumerate(container):
                avg = np.mean(c, axis=0)
                sem = np.std(c, axis=0) / math.sqrt(len(c))
                axarr[i].errorbar(x=np.linspace(0, len(avg[wnd[0]:wnd[1]]), len(avg[wnd[0]:wnd[1]])),
                                  y=avg[wnd[0]:wnd[1]], yerr=sem[wnd[0]:wnd[1]],
                                  color=cols[exp], linewidth=2, elinewidth=1.25)

            for i, c in enumerate(shifted_cont):
                avg = np.mean(c, axis=0)
                sem = np.std(c, axis=0) / math.sqrt(len(c))
                axarr[3+i].errorbar(x=np.linspace(0, len(avg[wnd[0]:wnd[1]]), len(avg[wnd[0]:wnd[1]])),
                                  y=avg[wnd[0]:wnd[1]], yerr=sem[wnd[0]:wnd[1]],
                                  color=cols[exp], linewidth=2, elinewidth=1.25)

        axarr[0].set(title='Velocity', xlim=[0, 135], ylim=[0, 9], xticks=ticksrange(0, 135, 15))
        axarr[1].set(title='Body length', xlim=[0, 135], ylim=[0.4, 1.25], xticks=ticksrange(0, 135, 15))
        axarr[2].set(title='Ang. vel', xlim=[0, 135], ylim=[0, 13], xticks=ticksrange(0, 135, 15))
        axarr[3].set(title='Velocity', xlim=[0, 135], ylim=[0, 9], xticks=ticksrange(0, 135, 15))
        axarr[4].set(title='Body length', xlim=[0, 135], ylim=[0.4, 1.25], xticks=ticksrange(0, 135, 15))
        axarr[5].set(title='Ang. vel', xlim=[0, 135], ylim=[0, 13], xticks=ticksrange(0, 135, 15))

        a = 1

    def plot_escape_trajectories(self, cols):
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
            if s != 'Right': continue
            for ii, ss in enumerate(sides):
                print('Conf {}, side {}'.format(s, ss))
                flipflop = self.triald_df.loc[(self.triald_df['experiment'] == 'FlipFlop Maze') &
                                              (self.triald_df['configuration'] == s)]
                flipflop = flipflop[(flipflop['escape'] == '{}_TtoM_bridge'.format(ss))]
                for tr in flipflop['tracking']:
                    threat_loc = tr.processing['Trial outcome']['maze_rois']['Threat_platform']

                    threat_loc = (threat_loc.topleft[0] + (threat_loc.bottomright[0] - threat_loc.topleft[0]) / 2,
                                  threat_loc.topleft[1] + (threat_loc.bottomright[1] - threat_loc.topleft[1]) / 2)
                    roi_trajectory = tr.processing['Trial outcome']['trial_rois_trajectory'][1800:]
                    tracking = (np.subtract(tr.dlc_tracking['Posture']['body']['x'][1800:], threat_loc[0]),
                                np.subtract(tr.dlc_tracking['Posture']['body']['y'][1800:], threat_loc[1]))
                    at_shelter = roi_trajectory.index('Shelter_platform')
                    print(at_shelter/30)
                    ax.plot(np.subtract(tracking[0].values[:at_shelter], 750*i),
                            tracking[1].values[: at_shelter], alpha=0.75, color=cols['flipflop'])


########################################################################################################################
###  LOADING and SAVING FUNCTIONS ####
########################################################################################################################

    def save_dataframe(self):
        import dill as pickle
        with open(os.path.join(self.data_fld, self.save_name), "wb") as dill_file:
            pickle.dump(self.triald_df, dill_file)

    def load_dataframe(self):
        self.triald_df = pd.read_pickle(os.path.join(self.data_fld, self.save_name))

########################################################################################################################
###  OLD ####
########################################################################################################################
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


