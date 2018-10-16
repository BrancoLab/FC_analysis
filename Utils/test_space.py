
# Set up figure
f = plt.figure(facecolor=[.1, .1, .1])
f.tight_layout()
axarr = []
nrows, ncols = 4, 6
facecolor = [.2, .2, .2]
axarr.append(plt.subplot2grid((nrows, ncols), (0, 0), colspan=2))
axarr.append(plt.subplot2grid((nrows, ncols), (0, 2), colspan=2))

for i in range(3): axarr.append(plt.subplot2grid((nrows, ncols), (1, 2* i), colspan=2))
for i in range(ncols): axarr.append(plt.subplot2grid((nrows, ncols), (2, i), colspan=1))
for i in range(ncols): axarr.append(plt.subplot2grid((nrows, ncols), (3, i), colspan=1))
axarr.append(plt.subplot2grid((nrows, ncols), (0, 4), colspan=1))
axarr.append(plt.subplot2grid((nrows, ncols), (0, 5), colspan=1))

# Perform chi squared for independence of variables:
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
chisq_results = {name: stats.chi2_contingency(val) for name, val in contingency_tables_nomargins.items()}

# plot the probability of going right for each experiment
axarr[0].bar(1, contingency_tables['flipflop']['All']['right'], color=cols['flipflop'],
             label='flipflop {}tr'.format(ntr_ff_r))
axarr[0].bar(2, contingency_tables['twoarms']['All']['right'], color=cols['twoarms'],
             label='two arms {}tr'.format(ntr_ta))
axarr[0].bar(0, contingency_tables['square']['All']['right'], color=cols['square'],
             label='squared, {}tr'.format(ntr_sq))
axarr[0].axhline(.5, color=[.8, .8, .8], linestyle=':', linewidth=.5)
axarr[0].set(title='p(R)', ylabel='p(R)', ylim=[0, 1], xlim=[-0.5, 2.5], facecolor=facecolor)
make_legend(axarr[0], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=12)

# Plot the bootstrapped distributions
skip = True
if not skip:
    noise = True
    niter = 200000
    nsamples = 39
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
                 xlim=[0, 1], xticks=np.arange(0, 1 + 0.1, 0.1), facecolor=facecolor)
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
             xticks=[], xlim=[-0.5, len(pr_sq_bymouse) + 0.5])
axarr[3].set(title='p(R) by mouse', ylabel='p(R)', xlabel='mice', ylim=[0, 1.1], facecolor=facecolor,
             xticks=[], xlim=[-0.5, len(pr_ff_r_bymouse) + 0.5])
axarr[4].set(title='p(R) by mouse', ylabel='p(R)', xlabel='mice', ylim=[0, 1.1], facecolor=facecolor,
             xticks=[], xlim=[-0.5, len(pr_ta_bymouse) + 0.5])

# Scatter plots of status at reaction
self.scatter_plotter(axarr[7], ff_r, 'origin', cols, coltag='flipflop')
self.scatter_plotter(axarr[8], ff_r, 'escape', cols, coltag='flipflop')
self.scatter_plotter(axarr[14], ff_r, 'Orientation', cols, coltag='flipflop')

self.scatter_plotter(axarr[9], ta, 'origin', cols, coltag='twoarms')
self.scatter_plotter(axarr[10], ta, 'escape', cols, coltag='twoarms')
self.scatter_plotter(axarr[12], ta, 'Orientation', cols, coltag='twoarms')

self.scatter_plotter(axarr[5], sq, 'origin', cols, coltag='square')
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
ntr_all = [len(self.triald_df[self.triald_df['session'] == s]) for s in sessions]
bplot = axarr[18].boxplot(ntr_all, vert=True, patch_artist=True, widths=5)

axarr[18].scatter(np.linspace(5, len(ntr_all) + 5, len(ntr_all)), sorted(ntr_all), s=55, color=[.6, .6, .6])

axarr[18].set(title='n trials by mouse', facecolor=facecolor, ylabel='num trials', xlim=[-3, len(ntr_all) + 7],
              xticks=[])
for patch, colname in zip(bplot['boxes'], labels):
    patch.set_facecolor([.8, .8, .8])

# Probability of escape given origin
self.prob_originescape(ff_r, axarr[13], cmap='Reds')
self.prob_originescape(sq, axarr[11], cmap='Blues')
self.prob_originescape(ta, axarr[15], cmap='Oranges')

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
