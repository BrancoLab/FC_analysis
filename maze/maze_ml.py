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
    def __init__(self, data, explore=False):
        self.data = data
        self.data_handling = DataProcessing(self.data)

        self.define_params()
        self.prepare_data()
        self.create_trainingvstest_data()

        if explore:
            self.explore_data(verbose=False, show_scatter_matrx=True)

        # define models
        # train
        # test
        # visualise

    ##################  SET UP

    def define_params(self):
        self.all_params = dict(
            included=['adjusted x', 'Orientation', 'adjusted y'],
            excluded=['name', 'likelihood', 'maze_roi', 'x', 'y', 'Accelearation', 'Head ang acc'],
            categoricals=['stimulus', 'origin']
        )

    def prepare_data(self):
        self.select_by_tag = 'experiment'
        self.select_by_values = ['Square Maze']

        # Prepare the dataframe
        self.prepped_data = self.data_handling.prep_status_atstim_data(self.data, scaledata=False,
                                                              excluded_params=None,
                                                              included_params=self.all_params['included'],
                                                              categoricals=self.all_params['categoricals'])
        # Clean up and remove categoricals
        self.prepped_data = self.data_handling.remove_error_trials(self.prepped_data)

    def create_trainingvstest_data(self):
        # Create training and test dataset
        self.training_data, self.training_labels, _, complete_test = self.data.get_training_test_datasets(self.data)
        self.colors = ['r' if v else 'g' for v in self.training_labels.values]

        # Remove categoricals
        self.training_data = self.data_handling.remove_categoricals(data=self.training_data,
                                                                    categoricals=self.all_params['categoricals'])

    ##################   EXPLORE DATA

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

    ##################  TRAIN MODELS

    def define_models(self):
        self.models = dict(
            sgd=SGDClassifier(),
            randomforest=RandomForestRegressor(),
            logreg=LogisticRegression(),
            svm=SVC(kernel='poly', degree=3, coef0=.1, C=5)
        )

    def train_model(self, model=None, crossval=None):
        if model is not None:
            models_to_train = self.models[model]
        else:
            models_to_train = [m for m in self.models.values()]

        for model in models_to_train:
            if crossval:
                probs = cross_val_predict(model, self.prepped_data, self.training_labels, cv=3, method='predict_proba')
                print('CV probability for model {}\n'.format(model), probs)
            else:
                model.fit(self.training_data, self.training_labels)

    ################### TEST MODELS

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

    # def fine_tune(self):
    #     # TODO work in progess for linear estimators
    #     random_search = RandomizedSearchCV(param_grid, cv=5, scoring='neg_mean_squared_error')
    #
    #
    #     param_grid = [{'max_features':[1, 2, 3, 4, 5, 6]},
    #                   {'bootstrap':[False],
    #                    'max_features':[1, 2, 3, 4, 5, 6]}]
    #     model = self.model
    #     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    #     grid_search.fit(self.prepped_data, self.escape_labels)
    #     grid_search.fit(self.training_data, self.training_labels)


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
        statuses = ['atstim', 'atmediandetection', 'atpostmediandetection']
        status = {}
        list_of_st_dict = []
        for i, st in enumerate(statuses):
            status_vars = dataset.iloc[0][st][1].keys()
            temp_status = {'{}_{}'.format(st, v): [] for v in status_vars if v not in excluded_params}
            for s in dataset[st].values:
                for v, val in s[1].items():
                    if v in excluded_params: continue
                    if v not in included_params:
                        continue
                    else:
                        if v in ['Orientation', 'Head angle']:
                            while val > 360: val -= 360
                        temp_status['{}_{}'.format(st, v)].append(val)
            list_of_st_dict.append(temp_status)

            # Scale the data
            if scaledata:
                for k, v in temp_status.items():
                    if not v: continue
                    else: temp_status[k] = self.scale_data(v)

        # Get categoricals and labels
        for name in categoricals:
            if name in included_params:
                status[name] = dataset[name].values
        status['escape'] = dataset['escape'].values

        # Clean up data and return
        status_df = pd.DataFrame.from_dict(status, orient='index')
        status_df = status_df.replace(to_replace='None', value=np.nan).dropna(how='all')
        status_df = status_df.transpose()
        temp = {**list_of_st_dict[0], **list_of_st_dict[1], **list_of_st_dict[2]}
        df = pd.DataFrame.from_dict(temp, orient='index')
        df = df.replace(to_replace='None', value=np.nan).dropna(how='all')
        df = df.transpose()
        for col in status_df.columns:
            d = status_df[col]
            df[col] = d

        return df

    @staticmethod
    def get_training_test_datasets(data):
        training_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        x = training_set.drop(columns=['escape'])
        y = training_set['escape']
        return x, y, training_set, test_set


