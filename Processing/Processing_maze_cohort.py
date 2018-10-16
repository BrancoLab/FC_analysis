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

from Plotting.Plotting_utils import make_legend, save_all_open_figs
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
        if not load:
            name = cohort_options['name']
            print(colored('Maze processing cohort {}'.format(name), 'green'))

            metad = cohort.Metadata[name]
            tracking_data = cohort.Tracking[name]
            self.from_trials_to_dataframe(metad, tracking_data, store_tracking)

            self.adjust_xypos()
        else:
            name = 'all_trials'
            self.load_dataframe()


        self.for_labmeeting()

        # self.plot_trajectories_at_roi(roi=['Threat_platform', 'Left_med_platform', 'Right_med_platform'])
        # if store_tracking:
        #     self.plot_escape_trajectories()
        # self.plot_status_stim()
        #
        # self.logreg_analaysis()
        #
        # self.plot_bootstrapped_distributions()

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

    def adjust_xypos(self):
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
        self.triald_df['tracking'] = None
        self.save_dataframe()

########################################################################################################################
    ###  FOR PRESENTATION ####
########################################################################################################################

    def for_labmeeting(self):
        cols = dict(
            flipflop=[233/250, 150/255, 122/255],
            twoarms=[173/255, 216/250, 230/255],
            square=[216/255, 191/255, 216/255],
            coin=[1.0, 1.0, 1.0]
        )

        """
        plot:
            * p(R) per experiment  --> statistical test to check for significance
            * $$ --- # trials per experiment
            * $$ --- # trials in total
            * $$ --- # p(R) per mouse
        
        factors that could be influencing p(R):
            * duration of escape --> avg time to shelter per arm
            * exploration --> avg time spend in each platform
            * $$ --- # arm of origin --> contingency tables: p(escape R) vs p(origin R)
            * $$ --- # x position --> p(R) binned by X position
            * previous trial --> ???
            * stimulus --> 
        
        """

        """ set up data and calc contingency tables """
        # get data
        ff_r, ntr_ff_r, _, _ = self.get_flipflop_data()
        ta, ntr_ta = self.get_twoarms_data()
        sq, ntr_sq = self.get_square_data()

        # Calculate contingencies tables for each experiment + create multi-layer dataframe
        experiments = ['square', 'flipflop', 'twoarms']
        datas = [sq, ff_r, ta]
        contingency_tables, contingency_tables_nomargins = {}, {}
        dff = []
        for exp, dat in zip(experiments, datas):
            temp_escs = [(i, 'left') if 'Left' in e else (i, 'right') if 'Right' in e else (None, None)
                    for i, e in enumerate(dat['escape'].values)]
            included_indexes = [e[0] for e in temp_escs]
            escs = [e[1] for e in temp_escs]
            oris = ['left' if 'Left' in e else 'right' if 'Right' in e else None for e in dat['origin'].values]

            n = [n if i in included_indexes else None for i, n in enumerate(dat['name'].values) ]
            sess = [n if i in included_indexes else None for i, n in enumerate(dat['session'].values)]
            stims = [n if i in included_indexes else None for i, n in enumerate(dat['stimulus'].values)]
            exps = [n if i in included_indexes else None for i, n in enumerate(dat['experiment'].values)]
            confs = [n if i in included_indexes else None for i, n in enumerate(dat['configuration'].values)]
            atstims = [(round(n[1]['adjusted x'],2), round(n[1]['adjusted y'],2)) if i in included_indexes else None
                       for i, n in enumerate(dat['atstim'].values)]

            dic = {
                'origin': oris,
                'escape': escs,
                'name': n,
                'session': sess,
                'stimulus': stims,
                'experiment': exps,
                'configuration': confs,
                'atstim': atstims
            }
            df = pd.DataFrame.from_dict(dic, orient='index').transpose()
            dff.append(df)
            cont_table = pd.crosstab(df['escape'], df['origin'], margins=True, normalize='all')
            nm_cont_table = pd.crosstab(df['escape'], df['origin'], margins=False)
            print('\n', cont_table)
            contingency_tables[exp] = cont_table
            contingency_tables_nomargins[exp] = nm_cont_table

        data_byexp = pd.concat(dff, axis=1, keys=experiments)  # <-- all data for good trials organised by experiment

        # Calculate n trials per experiment
        number_of_trials, number_of_trial_per_mouse = {}, {}
        for exp in experiments:
            ntr = []
            sess = data_byexp[exp]['session']
            for s in set(sess.values):
                if s is None: continue
                calculated = len(sess[data_byexp[exp]['session'] == s])
                if calculated: ntr.append(calculated)
            number_of_trials[exp] = np.sum(ntr)
            number_of_trial_per_mouse[exp] = ntr

        # Store p(R) for each experiment and calculate the binomial distribution C.I.:
        # https://www.thomasjpfan.com/2015/08/statistical-power-of-coin-flips/
        print('\n\n\nCalculating C.I.')
        alpha = 0.05
        pr_by_exp = {}
        for exp in experiments:
            pr = contingency_tables[exp]['All']['right']
            pr_by_exp[exp] = pr

            ntr = number_of_trials[exp]
            fair_interval = stats.binom.interval(1-alpha, ntr, 0.5)
            print(exp, fair_interval, ' -- ', round(pr, 2))

        # Calculate n trials per mouse for all mice
        sessions = set(self.triald_df['session'])
        ntr_all = [len(self.triald_df[self.triald_df['session'] == s]) for s in sessions]

        # Calculate p(R) for each mouse in each experiment
        prob_right_bymouse = {}
        for exp in experiments:
            probs = []
            sessions = set(data_byexp[exp]['session'].values)
            for s in sessions:
                if s is None: continue
                if np.isnan(s): continue
                exp_data = data_byexp[exp][data_byexp[exp]['session']==s]
                cont_table = pd.crosstab(exp_data['escape'], exp_data['origin'], margins=True, normalize='all')
                if 'right' in cont_table['All'].keys():
                    probs.append(round(cont_table['All']['right'], 2))
                else:
                    probs.append(0)
            prob_right_bymouse[exp] = probs

        # Calculate p(R) as a factor of X position
        # TODO this need to be checked
        # binned_pr = {}
        # bin_ranges = np.arange(-70, 70, 10)
        # for exp in experiments:
        #     positions_outcomes = []
        #     for s in range(len(data_byexp[exp]['atstim'].values)):
        #         pos = data_byexp[exp]['atstim'].iloc[s]
        #         out = data_byexp[exp]['escape'].iloc[s]
        #         try:
        #             if np.isnan(out): continue
        #         except: pass
        #         if pos is None: continue
        #         positions_outcomes.append((pos['adjusted x'], out))
        #
        #     ranged_p = []
        #     for i, r in enumerate(bin_ranges):
        #         if i == 0:
        #             good_outcomes = [o for p, o in positions_outcomes if p <= r]
        #         else:
        #             good_outcomes = [o for p, o in positions_outcomes if bin_ranges[i-1] < p <= r]
        #
        #         if good_outcomes:
        #             pr = round(len([o for o in good_outcomes if o == 'right'])/len(good_outcomes), 2)
        #             ranged_p.append((r, pr))
        #         else:
        #             ranged_p.append((r, None))
        #
        #     binned_pr[exp] = ranged_p

        # Show contingencies tables for stimulus vs escape arm
        for exp in experiments:
            cont_table = pd.crosstab(data_byexp[exp]['escape'], data_byexp[exp]['stimulus'], margins=True,
                                     normalize='columns')
            print('\n\n\n\n', cont_table)
            cont_table = pd.crosstab(data_byexp[exp]['escape'], data_byexp[exp]['stimulus'], margins=True)
            print('\n', cont_table)


        # Calculate chi-squared test for proportions
        # https://sites.ualberta.ca/~lkgray/uploads/7/3/6/2/7362679/slides_-_binomialproportionaltests.pdf
        # create a dataframe with exp id on one column and
        print('\n\n\nChi2 testing ---')
        dff = []
        exp_to_compare = ['square', 'flipflop', 'twoarms']
        for exp in exp_to_compare:
            good_escapes = [e for e in data_byexp[exp]['escape'].values if e is not None]
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
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')

        # Do a pairwise chi square test with bonferroni correction
        if len(exp_to_compare)>2:
            dummies = pd.get_dummies(exp_outcome_df['experiment'])
            adjusted_p = 0.05/len(exp_to_compare)

            for series in dummies:
                crosstab = pd.crosstab(dummies[series], exp_outcome_df['escape'])
                print('\n\n', crosstab)
                chi2, p, dof, expected = stats.chi2_contingency(crosstab)
                print('Chi2: {}, p: {}, DoF: {}'.format(chi2, p, dof))
                if p < adjusted_p: print('Significant')
                else: print('Not significant')









        # Set up figure
        f = plt.figure(facecolor=[.1, .1, .1])
        f.tight_layout()
        axarr = []
        nrows, ncols = 4, 6
        facecolor = [.2, .2, .2]
        axarr.append(plt.subplot2grid((nrows, ncols), (0, 0), colspan=2))
        axarr.append(plt.subplot2grid((nrows, ncols), (0, 2), colspan=2))

        for i in range(3): axarr.append(plt.subplot2grid((nrows, ncols), (1, 2*i), colspan=2))
        for i in range(ncols): axarr.append(plt.subplot2grid((nrows, ncols), (2, i), colspan=1))
        for i in range(ncols): axarr.append(plt.subplot2grid((nrows, ncols), (3, i), colspan=1))
        axarr.append(plt.subplot2grid((nrows, ncols), (0, 4), colspan=1))
        axarr.append(plt.subplot2grid((nrows, ncols), (0, 5), colspan=1))



        # Perform chi squared for independence of variables:
        # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
        chisq_results = {name:stats.chi2_contingency(val) for name, val in contingency_tables_nomargins.items()}

        # plot the probability of going right for each experiment
        axarr[0].bar(1, contingency_tables['flipflop']['All']['right'], color=cols['flipflop'], label='flipflop {}tr'.format(ntr_ff_r))
        axarr[0].bar(2, contingency_tables['twoarms']['All']['right'], color=cols['twoarms'], label='two arms {}tr'.format(ntr_ta))
        axarr[0].bar(0, contingency_tables['square']['All']['right'], color=cols['square'], label='squared, {}tr'.format(ntr_sq))
        axarr[0].axhline(.5, color=[.8, .8, .8], linestyle=':', linewidth=.5)
        axarr[0].set(title='p(R)', ylabel='p(R)', ylim=[0, 1], xlim=[-0.5, 2.5], facecolor=facecolor)
        make_legend(axarr[0], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=12)

        # Plot the bootstrapped distributions
        skip = True
        if not skip:
            noise = True
            niter = 200000
            nsamples=39
            ff_r_bs = self.bootstrap(ff_r['escape'].values, niter, num_samples=nsamples, noise=noise)
            ta_bs = self.bootstrap(ta['escape'].values, niter, num_samples=nsamples, noise=noise)
            sq_bs = self.bootstrap(sq['escape'].values, niter, num_samples=nsamples, noise=noise)
            coin = self.coin_simultor(num_samples=100, num_iters=niter, noise=noise)

            binz, norm = 250, False
            self.histogram_plotter(axarr[1], coin, color=cols['coin'], label='binomial',
                                   bins=binz, normalised=norm, alpha=.35, just_outline=False)
            self.histogram_plotter(axarr[1], ff_r_bs, color=cols['flipflop'], label='flipflop',
                                   bins=binz, normalised=norm, alpha=.75)
            self.histogram_plotter(axarr[1], ta_bs, color=cols['twoarms'], label='two arms',
                                   bins=binz, normalised=norm, alpha=.75)
            self.histogram_plotter(axarr[1], sq_bs, color=cols['square'], label='squared',
                                   bins=binz, normalised=norm, alpha=.75)
            axarr[1].set(title='bootstrapped p(R)', ylabel='frequency', xlabel='p(R)', ylim=[0, 0.08],
                         xlim=[0,1],  xticks=np.arange(0, 1+0.1, 0.1), facecolor=facecolor)
            make_legend(axarr[1], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=12)

            f, axarr = plt.subplots(4, 1)
            self.histogram_plotter(axarr[0], coin, color=cols['coin'], label='binomial',
                                   bins=binz, normalised=norm, alpha=.5, just_outline=False)
            self.histogram_plotter(axarr[1], ff_r_bs, color=cols['flipflop'], label='flipflop',
                                   bins=binz, normalised=norm, alpha=.75)
            self.histogram_plotter(axarr[2], ta_bs, color=cols['twoarms'], label='two arms',
                                   bins=binz, normalised=norm, alpha=.75)
            self.histogram_plotter(axarr[3], sq_bs, color=cols['square'], label='squared',
                                   bins=binz, normalised=norm, alpha=.75)
            for ax in axarr:
                ax.set(title='bootstrapped p(R)', ylabel='frequency', xlabel='p(R)',
                             xlim=[0, 1], xticks=np.arange(0, 1 + 0.1, 0.1), facecolor=facecolor)
                make_legend(axarr[1], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=12)

        # Look at probs of individual mice in the squared dataset
        pr_ff_r_bymouse, ntr_ff_r_bymouse = self.get_probability_bymouse(ff_r)
        pr_ta_bymouse, ntr_ta_bymouse = self.get_probability_bymouse(ta)
        pr_sq_bymouse, ntr_sq_bymouse = self.get_probability_bymouse(sq)

        axarr[2].bar(np.linspace(0, len(pr_sq_bymouse), len(pr_sq_bymouse)), sorted(pr_sq_bymouse),
                     color=cols['square'], label='sq by mouse')
        axarr[3].bar(np.linspace(0, len(pr_ff_r_bymouse), len(pr_ff_r_bymouse)), sorted(pr_ff_r_bymouse),
                     color=cols['flipflop'], label='ff by mouse')
        axarr[4].bar(np.linspace(0, len(pr_ta_bymouse), len(pr_ta_bymouse)), sorted(pr_ta_bymouse),
                     color=cols['twoarms'], label='ta by mouse')

        axarr[2].set(title='p(R) by mouse', ylabel='p(R)', xlabel='mice', ylim=[0, 1.1], facecolor=facecolor,
                     xticks=[],  xlim=[-0.5, len(pr_sq_bymouse)+0.5])
        axarr[3].set(title='p(R) by mouse', ylabel='p(R)', xlabel='mice', ylim=[0, 1.1], facecolor=facecolor,
                     xticks=[], xlim=[-0.5, len(pr_ff_r_bymouse) + 0.5])
        axarr[4].set(title='p(R) by mouse', ylabel='p(R)', xlabel='mice', ylim=[0, 1.1], facecolor=facecolor,
                     xticks=[], xlim=[-0.5, len(pr_ta_bymouse) + 0.5])

        # Scatter plots of status at reaction
        self.scatter_plotter(axarr[7], ff_r,  'origin', cols, coltag='flipflop')
        self.scatter_plotter(axarr[8], ff_r,'escape',  cols, coltag='flipflop')
        self.scatter_plotter(axarr[14], ff_r, 'Orientation', cols, coltag='flipflop')

        self.scatter_plotter(axarr[9], ta, 'origin', cols, coltag='twoarms')
        self.scatter_plotter(axarr[10], ta, 'escape', cols, coltag='twoarms')
        self.scatter_plotter(axarr[12], ta, 'Orientation', cols, coltag='twoarms')

        self.scatter_plotter(axarr[5], sq, 'origin',  cols, coltag='square')
        self.scatter_plotter(axarr[6], sq, 'escape', cols, coltag='square')
        self.scatter_plotter(axarr[16], sq, 'Orientation', cols, coltag='square')

        # BOX plot of ntrials per mouse grouped by exp
        data = [ntr_sq_bymouse, ntr_ff_r_bymouse, ntr_ta_bymouse]
        labels = ['square', 'flipflop', 'twoarms']
        bplot = axarr[17].boxplot(data,
                                   vert=True,  # vertical box alignment
                                   patch_artist=True,  # fill with color
                                   labels=labels)  # will be used to label x-ticks

        for patch, colname in zip(bplot['boxes'], labels):
            patch.set_facecolor(cols[colname])

        axarr[17].set(title='n trials per mouse by exp', facecolor=facecolor, ylabel='num trials', xticks=[])
        make_legend(axarr[17], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=12)

        # scatterplot ntrial for all mice
        sessions = set(self.triald_df['session'])
        ntr_all = [len(self.triald_df[self.triald_df['session']==s]) for s in sessions]
        bplot = axarr[18].boxplot(ntr_all, vert=True, patch_artist=True, widths=5)

        axarr[18].scatter(np.linspace(5, len(ntr_all)+5, len(ntr_all)), sorted(ntr_all), s=55, color=[.6, .6, .6])

        axarr[18].set(title='n trials by mouse', facecolor=facecolor, ylabel='num trials', xlim=[-3,len(ntr_all)+7],
                      xticks=[])
        for patch, colname in zip(bplot['boxes'], labels):
            patch.set_facecolor([.8, .8, .8])

        # Probability of escape given origin
        self.prob_originescape(ff_r, axarr[13],cmap='Reds')
        self.prob_originescape(sq, axarr[11],cmap='Blues')
        self.prob_originescape(ta, axarr[15],cmap='Oranges')

        # Calulate Welche's t-test
        welch_ffvis_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in ff_r['escape'].values],
                                              [1 if 'Right' in e else 0 for e in ta['escape'].values],
                                              equal_var=False)
        welch_ffaud_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in ff_r['escape'].values],
                                              [1 if 'Right' in e else 0 for e in sq['escape'].values],
                                              equal_var=False)
        welch_allff_twoarms = stats.ttest_ind([1 if 'Right' in e else 0 for e in sq['escape'].values],
                                              [1 if 'Right' in e else 0 for e in ta['escape'].values],
                                              equal_var=False)



        # self.probR_given()
        f.tight_layout()
        plt.show()
        a = 1

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

    def prob_originescape(self, data, ax, cmap='hot'):
        for _ in range(5):
            p = dict(lori_lesc=0,
                     lori_resc=0,
                     rori_lesc=0,
                     rori_resc=0)

            for n in range(len(data)):
                d = data.iloc[n]
                if 'left' in d['origin'].lower():
                    if 'left' in d['escape'].lower(): p['lori_lesc'] += 1
                    else: p['lori_resc'] += 1
                else:
                    if 'left' in d['escape'].lower(): p['rori_lesc'] += 1
                    else: p['rori_resc'] += 1

        tot = len(data)

        probs = [[p['lori_lesc']/tot, p['rori_lesc']/tot],
                 [p['lori_resc']/tot, p['rori_resc']/tot]]
        sns.heatmap(probs, ax=ax, vmin=0, vmax=1, center=0.25, annot=True,  linewidths=.25, cmap=cmap,
                    xticklabels=['L origin', 'R origin'], yticklabels=['L escape', 'R escape'])

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


