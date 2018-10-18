import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, \
    roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import sys

class EscapePrediction:
    def __init__(self, data):
        self.included_params = ['adjusted x', 'Orientation']
        self.excluded_params = ['name', 'likelihood', 'maze_roi', 'x', 'y', 'Accelearation', 'Head ang acc']
        self.categoricals = ['stimulus', 'origin']

        # Prepare the dataframe
        self.data = data
        data_all = self.prep_status_atstim_data()
        data_all = self.remove_wrong_escapes_ors(data_all)

        # Create training and test datasets
        self.training_set, self.test_set = train_test_split(data_all, test_size=0.2, random_state=42)

        self.prepped_data = self.training_set.drop(columns=['escape'])
        self.escape_labels = self.training_set['escape']

        # Explore dta and plot stuff
        self.explore_data()

        # Clean up the data
        self.remove_categoricals()

        # Train the model
        self.train()

        # Test it
        self.test()

        a = 1
        sys.exit()

    @staticmethod
    def remove_wrong_escapes_ors(data):
        from copy import deepcopy
        def cleaner(val):
            good = ['right', 'left']
            for g in good:
                if g in val.lower(): return g
            return None

        idx_to_remove = []
        for n in range(len(data)):
            d = data.iloc[n]

            escape = cleaner(d['escape'])
            origin = cleaner(d['origin'])

            if escape is None or origin is None: idx_to_remove.append(n)
            else:
                d.at['escape'] = 'escape_{}'.format(escape)
                d.at['origin'] = 'origin_{}'.format(origin)
                data.iloc[n] = d

        data = data.drop(index=idx_to_remove)
        return data

    def prep_status_atstim_data(self):
        """  get for each trial the info at the stim onset and organise in a dataframe """
        # Extract the relevant information
        status_vars = self.data.iloc[0].atstim[1].keys()
        status = {v:[] for v in status_vars if v not in self.excluded_params}
        for s in self.data.atstim.values:
            for v, val in s[1].items():
                if v in self.excluded_params: continue
                if v not in self.included_params: continue
                else:
                    if v in ['Orientation', 'Head angle']:
                        while val > 360: val-= 360

                    status[v].append(val)
        status['origin'] = self.data['origin'].values
        status['escape'] = self.data['escape'].values
        status['stimulus'] = self.data['stimulus'].values

        final_status = {k:v for k,v in status.items() if np.any(v)}
        df = pd.DataFrame.from_dict(final_status)
        return df

    def explore_data(self, verbose=False, show_scatter_matrx=True):
        self.prepped_data.head()
        self.prepped_data.info()
        self.prepped_data.describe()
        self.prepped_data.hist()
        # self.prepped_data.plot(kind='scatter', x='adjusted x', y='adjusted y', alpha=.5)
        self.corrr_mtx = self.prepped_data.corr()

        if verbose:
            for k in self.corrr_mtx.keys():
                print('\n Correlation mtx for {}'.format(k))
                print(self.corrr_mtx[k])

        if show_scatter_matrx:
            self.show_scatter_matrix()

    def show_scatter_matrix(self):
        attributes = self.prepped_data.keys()
        scatter_matrix(self.prepped_data[attributes], alpha=.5)

    def remove_categoricals(self):
        # remove categoricals
        for cat in self.categoricals:
            encoder = OneHotEncoder(sparse=False)
            catdf = self.prepped_data[cat]
            catdf_encoded, categories = catdf.factorize()
            catdf_encoded_hot = encoder.fit_transform(catdf_encoded.reshape(-1, 1))

            for i,c in enumerate(categories):
                self.prepped_data[c] = catdf_encoded_hot[:, i]

            self.prepped_data = self.prepped_data.drop(columns=cat)

    def train(self, model_type='sgd', crossval=False):
        if model_type == 'sgd':
            self.model = SGDClassifier()
        elif model_type ==  'randomforest':
            self.model = RandomForestClassifier
        else:
            return

        if crossval:
            probs = cross_val_predict(self.model, self.prepped_data, self.escape_labels, cv=3, method='predict_proba')
            print('CV probability for model {}\n'.format(model_type), probs)
        else:
            self.model.fit(self.prepped_data, self.escape_labels)

    def test(self, crossval=True, conf_mtx=True, precision=True, curves=True, roc=True):
        pred = cross_val_predict(self.model, self.prepped_data, self.escape_labels, cv=3)

        if crossval:
            cv_score = cross_val_score(self.model, self.prepped_data, self.escape_labels, cv=3, scoring='accuracy')
            print('\n Crossval score: {}'.format(cv_score))
        if conf_mtx:
            confmtx = confusion_matrix(self.escape_labels, pred)
            print('Confusion matrix:\n {}'.format(confmtx))
        if precision:
            try:
                pc = precision_score(self.escape_labels, pred, pos_label='escape_right')
                rc = recall_score(self.escape_labels, pred, pos_label='escape_right')
                f1 = f1_score(self.escape_labels, pred, pos_label='escape_right')
                print('\nPrecision - {}\nRecall - {}\nF1 score - {}'.format(pc, rc, f1))
            except: print('couldnt calculate precision')
        if curves:
            try:
                prec, rec, thresh = precision_recall_curve(self.escape_labels, pred, pos_label='escape_right')
                plt.figure()
                plt.plot(thresh, prec[:-1], color='g')
                plt.plot(thresh, rec, color='m')
                plt.ylim([0, 1])
            except: print('Could not compute prcision-recall curve')
        if roc:
            try:
                fpr, tpr,thres = roc_curve(self.escape_labels, pred, pos_label='escape_right')
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1])
                plt.axis([0, 1, 0, 1])
            except: print('could not compute ROC curve')


        a = 1

