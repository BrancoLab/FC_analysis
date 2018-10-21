import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, \
    roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

import sys


class EscapePrediction:
    def __init__(self, data):
        all_params = ['adjusted x', 'Orientation', 'adjusted y']
        explore = False

        for nparams in range(len(all_params)):
            self.included_params = all_params[:nparams+1]
            self.excluded_params = ['name', 'likelihood', 'maze_roi', 'x', 'y', 'Accelearation', 'Head ang acc']
            self.categoricals = ['stimulus', 'origin']

            self.select_by_tag = 'experiment'
            self.select_by_values = ['Square Maze']

            # Prepare the dataframe
            self.data = data
            data_all = self.prep_status_atstim_data(select_subset=True, scale_data=True)
            data_all = self.remove_wrong_escapes_ors(data_all)

            # Create training and test datasets
            self.training_set, self.test_set = train_test_split(data_all, test_size=0.2, random_state=42)

            self.prepped_data = self.training_set.drop(columns=['escape'])
            self.escape_labels = self.training_set['escape']
            self.escape_labels = (self.escape_labels == 'escape_right').astype(np.int) # 1 for R - else 0

            self.colors = ['r' if v else 'g' for v in self.escape_labels.values]
        # TODO make it cleaner, facilitate comparison of same model across different dataset or for different params
        # TODO make it cleaner, to facilitate comparison across models
        # TODO improve model testing and results visualisation

        # Get data at stim onset 
        self.data = data
        processor = DataProcessing(self.data)
        data_all = processor.prep_status_atstim_data(dataset=self.data, scaledata=True, 
                                                     excluded_params=self.excluded_params,
                                                     included_params=self.included_params,
                                                     categoricals=self.categoricals)
        data_all = processor.remove_error_trials(data_all, keepC=False)

        # Create training and test dataset
        self.training_data, training_labels, _, complete_test = processor.get_training_test_datasets(data_all)
        # Remove categoricals
        self.training_data = processor.remove_categoricals(data=self.training_data, categoricals=self.categoricals)

            # Explore data and plot stuff
            if explore:
                self.explore_data(show_scatter_matrx=True)

            # Clean up the data
            self.remove_categoricals()

            # Train the model
            self.train(model_type='svm')

            # Test it
            self.test()
            print('Working with {} numerical parameters:'.format(nparams))
            print('     Numerical parameters: {}'.format(self.included_params))
            print('     Categorical parameters: {}'.format(self.categoricals))

        # Fine-tune the model
        # self.fine_tune()

        a = 1
        sys.exit()

    def select_data_subset(self, data, removebyconfig=False):
        data = data[data[self.select_by_tag].isin(self.select_by_values)]
        if removebyconfig:
            data = data[data['configuration'].isin(['Right'])]
        return data

    @staticmethod
    def remove_wrong_escapes_ors(data):
        def cleaner(val):
            good = ['right', 'left']
            for g in good:
                if g in val.lower(): return g
            return None

        idx_to_remove = []
        for n in range(len(data)):
            d = data.iloc[n]

            escape = cleaner(d['escape'])
            if 'origin' in d.keys():
                origin = cleaner(d['origin'])

                if escape is None or origin is None: idx_to_remove.append(n)
            else:
                if escape is None: idx_to_remove.append(n)
            d.at['escape'] = 'escape_{}'.format(escape)
            if 'origin' in d.keys():
                d.at['origin'] = 'origin_{}'.format(origin)
            data.iloc[n] = d

        data = data.drop(index=idx_to_remove)
        return data

    def prep_status_atstim_data(self, select_subset=False, scale_data=False):
        """  get for each trial the info at the stim onset and organise in a dataframe """
        # Select only relevant subset of trials
        self.data = self.select_data_subset(self.data)

        # Extract the relevant information
        status_vars = self.data.iloc[0].atstim[1].keys()
        statuses_names = ['atstim', 'atmediandetection', 'atpostmediandetection']

        statuses_list = []
        for status_name in statuses_names:
            temp_status = {'{}_{}'.format(status_name, v):[] for v in status_vars if v not in self.excluded_params}

            for s in self.data[status_name].values:
                for v, val in s[1].items():
                    if v in self.excluded_params: continue
                    if select_subset:
                        if v not in self.included_params: continue

                    if v in ['Orientation', 'Head angle']:
                        while val > 360: val-= 360
                    temp_status['{}_{}'.format(status_name, v)].append(val)
            statuses_list.append(temp_status)
        status = {**statuses_list[0], **statuses_list[1], **statuses_list[2]}

        # Scale the data
        if scale_data:
            for k,v in status.items():
                if not v: continue
                scaler = StandardScaler()
                v = np.asarray(v).reshape((-1, 1))
                scaled = scaler.fit_transform(v.astype(np.float64))
                status[k] = scaled.flatten()

        for name in self.categoricals:
            if select_subset:
                if name in self.included_params:
                    status[name] = self.data[name].values
                else:
                    status[name] = self.data[name].values

        status['escape'] = self.data['escape'].values
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
        self.training_data.head()
        self.training_data.info()
        self.training_data.describe()
        self.training_data.hist()
        self.training_data.plot(kind='scatter', x='adjusted x', y='adjusted y', alpha=.5)
        self.corrr_mtx = self.training_data.corr()

        if verbose:
            for k in self.corrr_mtx.keys():
                print('\n Correlation mtx for {}'.format(k))
                print(self.corrr_mtx[k])

        if show_scatter_matrx:
            self.show_scatter_matrix()

    def show_scatter_matrix(self):
        attributes = self.prepped_data.keys()
        scatter_matrix(self.prepped_data[attributes],  color=self.colors, diagonal='kde', s=200, alpha=0.75)

    def remove_categoricals(self):
        # remove categoricals
        for cat in self.categoricals:
            if not cat in self.prepped_data.keys(): continue
            encoder = OneHotEncoder(sparse=False)
            catdf = self.prepped_data[cat]
            catdf_encoded, categories = catdf.factorize()
            catdf_encoded_hot = encoder.fit_transform(catdf_encoded.reshape(-1, 1))

            for i,c in enumerate(categories):
                self.prepped_data[c] = catdf_encoded_hot[:, i]

            self.prepped_data = self.prepped_data.drop(columns=cat)
        attributes = self.training_data.keys()
        scatter_matrix(self.training_data[attributes], alpha=.5)

    def train(self, model_type='sgd', crossval=False):
        if model_type == 'sgd':
            self.model = SGDClassifier()
        elif model_type == 'randomforest':
            self.model = RandomForestRegressor()
        elif model_type == 'logreg':
            self.model = LogisticRegression()
        elif model_type == 'svm':
            self.model = SVC(kernel='poly', degree=3, coef0=.1, C=5)
        else:
            return

        if crossval:
            probs = cross_val_predict(self.model, self.prepped_data, self.escape_labels, cv=3, method='predict_proba')
            probs = cross_val_predict(self.model, self.training_data, self.training_labels, cv=3, method='predict_proba')
            print('CV probability for model {}\n'.format(model_type), probs)
        else:
            self.model.fit(self.prepped_data, self.escape_labels)
            self.model.fit(self.training_data, self.training_labels)

    def test(self, crossval=True, conf_mtx=True, precision=True, curves=True, roc=True):
        pred = cross_val_predict(self.model, self.prepped_data, self.escape_labels, cv=3)
        pred = cross_val_predict(self.model, self.training_data, self.training_labels, cv=3)

        if crossval:
            cv_score = cross_val_score(self.model, self.prepped_data, self.escape_labels, cv=3, scoring='accuracy')
            cv_score = cross_val_score(self.model, self.training_data, self.training_labels, cv=3, scoring='accuracy')
            print('\n Crossval score: {}'.format(cv_score))
        if conf_mtx:
            confmtx = confusion_matrix(self.escape_labels, pred)
            confmtx = confusion_matrix(self.training_labels, pred)

            # normalised it to highlight errors
            row_sums = confmtx.sum(axis=1, keepdims=True)
            norm_confmtx = confmtx / row_sums
            np.fill_diagonal(norm_confmtx, 0)

            plt.figure()
            plt.matshow(norm_confmtx)

            print('Confusion matrix:\n {}'.format(confmtx))
        if precision:
            try:
                pc = precision_score(self.escape_labels, pred)
                rc = recall_score(self.escape_labels, pred)
                f1 = f1_score(self.escape_labels, pred)
                pc = precision_score(self.training_labels, pred, pos_label='escape_right')
                rc = recall_score(self.training_labels, pred, pos_label='escape_right')
                f1 = f1_score(self.training_labels, pred, pos_label='escape_right')
                print('\nPrecision - {}\nRecall - {}\nF1 score - {}'.format(pc, rc, f1))
            except: print('couldnt calculate precision')
        if curves:
            try:
                prec, rec, thresh = precision_recall_curve(self.escape_labels, pred)
                plt.figure()
                plt.plot(thresh, prec[:-1], color='g')
                plt.plot(thresh, rec, color='m')
                plt.ylim([0, 1])
            except: print('Could not compute prcision-recall curve')
        if roc:
            try:
                fpr, tpr,thres = roc_curve(self.escape_labels, pred)
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1])
                plt.axis([0, 1, 0, 1])
            except: print('could not compute ROC curve')


        a = 1

    def fine_tune(self):
        # TODO work in progess for linear estimators
        random_search = RandomizedSearchCV(param_grid, cv=5, scoring='neg_mean_squared_error')


        param_grid = [{'max_features':[1, 2, 3, 4, 5, 6]},
                      {'bootstrap':[False],
                       'max_features':[1, 2, 3, 4, 5, 6]}]
        model = self.model
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.prepped_data, self.escape_labels)
        grid_search.fit(self.training_data, self.training_labels)


class DataProcessing:
    """ This class does a number of processing steps that are specific for maze experiments
      and it defines a number of sklearn pipelanes that can be used to prepare data for ML approaches
      """
    def __init__(self, data):
        self.data = data

    def subset_bytag(self, tag=None, data=None, values=None):
        """ Select part of a dataframe using a tag (column name) and a range of specific values
          :param tag = name of the column, if none the whole dataset is returned
          :param data = pandas df to process, if none the dataframe with which the class was initialised will be used
          :param values = iterable, range of permitted values for the selection of data

          :returns portion of the dataframe whose rows in column TAG have value in range VALUES
          """
        if data is None:
            data = self.data
        if tag is None:
            return data
        else:
            if values is None:
                data = self.data[self.data[tag]]
            else:
                data = self.data[self.data[tag].isin(values)]

            return data

    def remove_error_trials(self, data=None, keepC=False):
        """ Given a dataset, only keep the trials with R and L escapes/origin (and, optionally C trials)
        Also change the value of escape/origin to 'right'... to make it uniform across experiments

        :param keepC bool, set to false to discard trials with C origin or escapes """

        if keepC:
            good = ['right', 'left', 'centre']
        else:
            good = ['right', 'left']

        def cleaner(val):
            for g in good:
                if g == 'centre':
                    if g[:-1] in val.lower(): return g
                else:
                    if g in val.lower(): return g
            return None

        if data is None: data = self.data

        idx_to_remove = []
        for n in range(len(data)):
            d = data.iloc[n]

            escape = cleaner(d['escape'])
            if 'origin' in d.keys():
                origin = cleaner(d['origin'])
                if escape is None or origin is None: idx_to_remove.append(n)
            else:
                if escape is None: idx_to_remove.append(n)

            d.at['escape'] = 'escape_{}'.format(escape)
            if 'origin' in d.keys():
                d.at['origin'] = 'origin_{}'.format(origin)
            data.iloc[n] = d

        data = data.drop(index=idx_to_remove)
        return data

    def remove_categoricals(self, data=None, categoricals=None):
        """ Turn columns with categorical values (strings) in a DF to N columns with binary values where N is
         the number of set(values) in a categorical column: each new column A has 1 for entries that match
         set(values)[A] and zero otherwise.
          
          Relies on OneHotEncoder from sklearn """
        # TODO add possibility to detect categorical columns automatically without needing input from user
        if data is None: data = self.data
        if categoricals is None: return data

        # remove categoricals
        for cat in categoricals:
            if not cat in data.keys(): continue
            encoder = OneHotEncoder(sparse=False)  # <-- set as True to use spars matrix notation
            catdf = data[cat]
            catdf_encoded, categories = catdf.factorize()
            catdf_encoded_hot = encoder.fit_transform(catdf_encoded.reshape(-1, 1))
            for i,c in enumerate(categories):
                data[c] = catdf_encoded_hot[:, i]

            data = data.drop(columns=cat)
        return data

    @staticmethod
    def scale_data(d):
        scaler = StandardScaler()
        v = np.asarray(d).reshape((-1, 1))
        scaled = scaler.fit_transform(v.astype(np.float64))
        return scaled.flatten()

    def prep_status_atstim_data(self, dataset, scaledata=False, excluded_params=None, included_params=None,
                                categoricals=None):
        """  Given a dataset, for each trial extract the values corresponding to each statusat_... separately
         in different dataframes """
        # Extract the relevant information
        try:
            status_vars = dataset.iloc[0].atstim[1].keys()
        except:
            raise ValueError('Could not parse dataframe')
        
        if excluded_params is None: excluded_params = []
        if included_params is None: included_params = status_vars
        if categoricals is None: categoricals = []
        
        # Select the relevant data columns from the dataset df
        status = {v: [] for v in status_vars if v not in excluded_params}
        for s in dataset.atstim.values:
            for v, val in s[1].items():
                if v in excluded_params: continue
                if v not in included_params:
                    continue
                else:
                    if v in ['Orientation', 'Head angle']:
                        while val > 360: val -= 360
                    status[v].append(val)

        # Scale the data
        if scaledata:
            for k, v in status.items():
                if not v: continue
                else: status[k] = self.scale_data(v)

        for name in categoricals:
            if name in included_params:
                status[name] = dataset[name].values

        status['escape'] = dataset['escape'].values

        final_status = {k: v for k, v in status.items() if np.any(v)}
        df = pd.DataFrame.from_dict(final_status)
        return df

    @staticmethod
    def get_training_test_datasets(data):
        training_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        x = training_set.drop(columns=['escape'])
        y = training_set['escape']
        return x, y, training_set, test_set



