import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

from Utils.maths import calc_distance_2d, calc_angle_2d
from Plotting.Plotting_utils import create_figure, make_legend

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, \
    roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DecTreC
from sklearn.tree import export_graphviz
import suftware as sw

import sys


class EscapePrediction:
    def __init__(self, data, suft=True, explore=True):
        self.select_subset = True
        self.select_by_tag = 'experiment'
        self.select_by_values = ['TwoArms maze', 'FlipFlop Maze']

        self.data = data   # .iloc[280:]
        self.data_handling = DataProcessing(self.data)

        self.define_params()
        self.prepare_data()
        self.create_trainingvstest_data()

        # trying random tree classifier
        refined_data = None
        for rown in range(len(self.training_data)):
            print(rown)
            d = self.training_data.atstim.iloc[rown].status.drop('likelihood').drop('Head ang vel').drop('Head angle').drop('Head ang acc').drop('x').drop('y').drop('Body ang vel').drop('Accelearation').drop('Body length')
            d = pd.DataFrame(d).transpose()
            if refined_data is None:
                refined_data = d
            else:
                refined_data = refined_data.append(d)

        while np.any(refined_data.Orientation[refined_data.Orientation>360]):
            refined_data[refined_data.Orientation>360] -= 360


        refined_labels = pd.DataFrame(self.training_labels)

        tree = DecTreC(max_depth=8, min_samples_leaf=5) # max_depth=1, min_samples_leaf=1, max_leaf_nodes=4, max_features=1
        tree.fit(refined_data, refined_labels)

        export_graphviz(tree, out_file='/Users/federicoclaudi/desktop/tree_depth.dot', feature_names=refined_data.columns,
                        class_names=tuple(set(refined_labels.escape)), rounded=True, filled=True)

        if explore:
            self.explore_data(verbose=False, show_scatter_matrx=True)

        if suft:
            SUFT.calc_density_for_dataframe(self.prepped_data)

        # define models
        # train
        # test
        # visualise

    ##################  SET UP

    def define_params(self):
        self.all_params = dict(
            included=['x', 'y', 'adjusted x', 'Orientation', 'adjusted y'],
            excluded=['name', 'likelihood', 'maze_roi', 'Accelearation', 'Head ang acc'],
            categoricals=['stimulus', 'origin'],
            statuses=['atstim', 'atmediandetection', 'atpostmediandetection']
        )

    def prepare_data(self):
        # Select trials subset
        if self.select_subset:
            self.data = self.data_handling.subset_bytag(data=self.data, tag=self.select_by_tag,
                                                        values=self.select_by_values)

        # Prepare the dataframe
        self.prepped_data = self.data_handling.prep_status_atstim_data(self.data, scaledata=False,
                                                                       excluded_params=None,
                                                                       included_params=self.all_params['included'],
                                                                       categoricals=self.all_params['categoricals'],
                                                                       statuses=self.all_params['statuses'])

        # Clean up and remove categoricals
        self.prepped_data = self.data_handling.remove_error_trials(self.prepped_data)

        # Extract new metrics
        self.data_handling.calculate_new_metrics(self.prepped_data, timepoints=self.all_params['statuses'],
                                                 display=True)

    def create_trainingvstest_data(self):
        # Create training and test dataset
        self.training_data, self.training_labels, _, complete_test = self.data_handling.get_training_test_datasets(self.data)
        self.colors = ['r' if v else 'g' for v in self.training_labels.values]

        # Remove categoricals
        # self.training_data = self.data_handling.remove_categoricals(data=self.training_data,
        #                                                             categoricals=self.all_params['categoricals'])

    ##################   EXPLORE DATA

    def explore_data(self, verbose=False, show_scatter_matrx=True):
        self.training_data.head()
        self.training_data.info()
        self.training_data.describe()
        self.training_data.hist()
        # self.training_data.plot(kind='scatter', x='adjusted x', y='adjusted y', alpha=.5)
        self.corrr_mtx = self.training_data.corr()

        if verbose:
            for k in self.corrr_mtx.keys():
                print('\n Correlation mtx for {}'.format(k))
                print(self.corrr_mtx[k])

        if show_scatter_matrx:
            self.show_scatter_matrix()

    def show_scatter_matrix(self):
        attributes = self.prepped_data.keys()
        exclude_words = ['bridge', 'platform', 'BL']
        # TODO clean up attributes
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
                if not isinstance(values, list): values = list(values)
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
                                categoricals=None, statuses=None, include_maze_rois=True):
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
        if statuses is None: statuse = ['atstim', 'atmediandetection', 'atpostmediandetection']

        # Select the relevant data columns from the dataset df
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

        # Get position of maze rois
        # good names med/far platform TtoP/PtoS bridge
        goodnames = ['left_med_platform', 'left_far_platform', 'left_PtoS_bridge', 'left_TtoP_bridge',
                     'right_med_platform', 'right_far_platform', 'right_PtoS_bridge', 'right_TtoP_bridge',
                     'Threat_platform', 'Shelter_platform', 'meanBL']
        rois_list = []
        if include_maze_rois:
            for n in range(len(dataset)):
                good_rois = {n:None for n in goodnames}
                rois = dataset['tracking'].iloc[n].processing['Trial outcome']['maze_rois']
                for roi, pos in rois.items():
                    if roi in ['Central_TtoS_bridge', 'Central_TtoX_bridge']:
                        continue
                    elif roi == 'Threat_platform':
                        good_rois['Threat_platform'] = pos
                    elif roi == 'Shelter_platform':
                        good_rois['Shelter_platform'] = pos
                    elif 'Left' in roi:
                        if 'med' in roi: good_rois['left_med_platform'] = pos
                        elif 'far' in roi: good_rois['left_far_platform'] = pos
                        elif 'S' in roi: good_rois['left_PtoS_bridge'] = pos
                        elif 'T' in roi: good_rois['left_TtoP_bridge'] = pos
                        else: continue
                    elif 'Right: in roi':
                        if 'med' in roi: good_rois['right_med_platform'] = pos
                        elif 'far' in roi: good_rois['right_far_platform'] = pos
                        elif 'S' in roi: good_rois['right_PtoS_bridge'] = pos
                        elif 'T' in roi: good_rois['right_TtoP_bridge'] = pos
                        else: continue
                good_rois['meanBL'] = np.mean(dataset['tracking'].iloc[n].dlc_tracking['Posture']['body']['Body length'])
                rois_list.append(good_rois)
            roisdf = pd.DataFrame(rois_list)

        # Clean up data and return
        status_df = pd.DataFrame.from_dict(status, orient='index')
        status_df = status_df.replace(to_replace='None', value=np.nan).dropna(how='all')
        status_df = status_df.transpose()
        if len(list_of_st_dict) == 1:
            temp = {**list_of_st_dict[0]}
        else:
            temp = {**list_of_st_dict[0], **list_of_st_dict[1], **list_of_st_dict[2]}
        df = pd.DataFrame.from_dict(temp, orient='index')
        df = df.replace(to_replace='None', value=np.nan).dropna(how='all')
        df = df.transpose()

        # Now merge everything
        for col in status_df.columns:
            d = status_df[col]
            df[col] = d

        if include_maze_rois:
            for col in roisdf.columns:
                d = roisdf[col]
                df[col] = d
        return df

    @staticmethod
    def get_training_test_datasets(data):
        training_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        x = training_set.drop(columns=['escape'])
        y = training_set['escape']
        return x, y, training_set, test_set

    def calculate_new_metrics(self, data, timepoints=None, display=False):
        data = self.get_distance_to_Rescape_arm(data, timepoints=timepoints, visualise=display)
        data = self.get_Rescapelength_and_angletoRescape(data, timepoints=timepoints, visualise=display)


        # plt.show()
        a = 1

    @staticmethod
    def calc_roi_centre(roi):
        d = (int((roi.bottomright[0] - roi.topleft[0]) / 2),
             int((roi.bottomright[1] - roi.topleft[1]) / 2))
        d = (d[0]+roi.topleft[0], d[1]+roi.topleft[1])
        return d

    @staticmethod
    def add_col_to_df(df, lst, tag):
        temp = pd.DataFrame(lst)
        df[tag] = temp[0]
        return df

    def get_distance_to_Rescape_arm(self, data, timepoints=None, visualise=False):
        if timepoints is None: raise ValueError('incorrect input')
        # Get the distance to the arm of escape
        for tp in timepoints:
            distance_to_ebridge = []
            xpos, r_xpos, l_xpos = [], [], []
            re_distance_to_ebridge, le_distance_to_ebridge = [], []

            for n in range(len(data)):
                tr = data.iloc[n]
                pos = (tr['{}_x'.format(tp)], tr['{}_y'.format(tp)])

                xpos.append(tr['{}_adjusted x'.format(tp)])

                # Get distance of escape bridge
                esc_bridge_pos = self.calc_roi_centre(tr['right_TtoP_bridge'])

                dst = calc_distance_2d((pos, esc_bridge_pos), vectors=False)/tr['meanBL']
                distance_to_ebridge.append(dst)

                if 'right' in tr['escape']:
                    re_distance_to_ebridge.append(dst)
                    r_xpos.append(tr['{}_adjusted x'.format(tp)])
                else:
                    le_distance_to_ebridge.append(dst)
                    l_xpos.append(tr['{}_adjusted x'.format(tp)])

            if visualise:
                all_den = SUFT.calc_density_for_list(distance_to_ebridge)
                r_den = SUFT.calc_density_for_list(re_distance_to_ebridge)
                l_den = SUFT.calc_density_for_list(le_distance_to_ebridge)

                x_den = SUFT.calc_density_for_list(xpos)
                rx_den = SUFT.calc_density_for_list(r_xpos)
                lx_den = SUFT.calc_density_for_list(l_xpos)


                f, ax = create_figure(nrows=2)
                ax[0].plot(all_den.values, color='b', linewidth=2, label='All')
                ax[0].plot(r_den.values, color='r', alpha=.5, label='R')
                ax[0].plot(l_den.values, color='g', alpha=.5, label='L')
                ax[0].set(facecolor=[.2, .2, .2], xlim=[100, 0], title='distance to R arm - {}'.format(tp))
                make_legend(ax[0], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

                ax[1].plot(x_den.values, color='b', linewidth=2, label='All')
                ax[1].plot(rx_den.values, color='r', alpha=.5, label='R')
                ax[1].plot(lx_den.values, color='g', alpha=.5, label='L')
                ax[1].set(facecolor=[.2, .2, .2], xlim=[100, 0], title='Adjusted x pos - {}'.format(tp))
                make_legend(ax[1], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

            data = self.add_col_to_df(data, distance_to_ebridge, '{}_distance_to_ebridge'.format(tp))
        return data

    def get_Rescapelength_and_angletoRescape(self, data, timepoints=None, visualise=False):
        if timepoints is None: raise ValueError('incorrect input')

        normed_rpath_len, ang_to_r = [], []
        r_ang_to_r, l_ang_to_r = [], []
        for n in range(len(data)):
            tr = data.iloc[n]
            pos = (tr.atstim_x, tr.atstim_y)
            shelter = self.calc_roi_centre(tr['Shelter_platform'])
            threat = self.calc_roi_centre(tr['Threat_platform'])

            if tr.left_far_platform is not None:
                lplat = self.calc_roi_centre(tr['left_far_platform'])
            else:
                lplat = self.calc_roi_centre(tr['left_med_platform'])

            if tr.right_far_platform is not None:
                rplat = self.calc_roi_centre(tr['right_far_platform'])
            else:
                rplat = self.calc_roi_centre(tr['right_med_platform'])

            l_len = (calc_distance_2d((threat, lplat), vectors=False)
                     + calc_distance_2d((lplat, shelter), vectors=False))/tr['meanBL']
            r_len = (calc_distance_2d((threat, rplat), vectors=False)
                     + calc_distance_2d((rplat, shelter), vectors=False))/tr['meanBL']
            normed_rpath_len.append(round(r_len/l_len, 1))

            # calc angle to r bridge
            # TODO fix angles
            for i, tp in enumerate(timepoints):
                thetaR = calc_angle_2d(pos, rplat)
                theta = tr['{}_Orientation'.format(tp)]
                while theta > 360: theta -= 360
                ang = thetaR - theta
                ang_to_r.append(ang)
                esc = tr['escape']
                if 'right' in esc:
                    r_ang_to_r.append(ang)
                else:
                    l_ang_to_r.append(ang)

                data = self.add_col_to_df(data, ang_to_r, '{}_angle_to_rplatf'.format(tp))

        data = self.add_col_to_df(data, normed_rpath_len, 'normed_rpath_len')

        if visualise:
            f, ax = create_figure()
            ax.hist(normed_rpath_len, color='b')
            ax.set(facecolor=[.2, .2, .2], title='Normalised R path length')
            make_legend(ax, [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

            for i, tp in enumerate(timepoints):
                if i == 0:
                    f, ax = create_figure(nrows=len(timepoints))
                ax[i].hist(ang_to_r, color='b')
                ax[i].hist(r_ang_to_r, color='r', alpha=.5, label='R')
                ax[i].hist(l_ang_to_r, color='g', alpha=.5, label='L')
                ax[i].set(facecolor=[.2, .2, .2], title='Angle to R platform {}'.format(tp))
                make_legend(ax[1], [0.1, .1, .1], [0.8, 0.8, 0.8], changefont=16)

        return data


class SUFT:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def calc_density_for_dataframe(data, display=True):
        for col in data.columns:
            d = data[col].values

            try:
                density = sw.DensityEstimator(d)
            except:
                continue

            if display: density.plot(title=col)

    @staticmethod
    def calc_density_for_list(data):
        return sw.DensityEstimator(np.asarray(data))







