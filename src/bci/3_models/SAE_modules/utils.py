import numpy as np
import mne
import scipy
import torch
import sklearn
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset


def csp_score(signal, labels, cv_N = 5, classifier = False):
    
    # Set verbose to 0
    mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    
    if classifier:
        y_pred = classifier.predict(signal)
        acc = sklearn.metrics.accuracy_score(labels, y_pred)
        return acc
    
    else:
        # Assemble a classifier
        svm = sklearn.svm.SVC()
        lda = LinearDiscriminantAnalysis()
#         lda = sklearn.ensemble.RandomForestClassifier()
        csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=True)
        # Use scikit-learn Pipeline with cross_val_score function
        #clf = Pipeline([('CSP', csp), ('LDA', lda)])
        clf = Pipeline([('CSP', csp), ('SVM', svm)])
        scores = cross_val_score(clf, signal, labels, cv=cv_N, n_jobs=5)
        _ = clf.fit(signal, labels)
        return np.mean(scores), clf


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    ly += yerr[num1]
    ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh+0.05)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


class EEGDataSet_signal_by_day(Dataset):
    def __init__(self, EEGDict, days_range=[0,1]):
        
        # Concat dict      
        X, y, days_y = self.concat(EEGDict, days_range)
        
        # Convert from numpy to tensor
        self.X = torch.tensor(X)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.y = y
        self.days_y = days_y
        self.n_days_labels = days_range[1] - days_range[0]
        self.n_task_labels = y.shape[1]

        
    def __getitem__(self, index):
        return self.X[index].float(), self.y[index], self.days_y[index]
    
    def __len__(self):
        return self.n_samples
    
    def getAllItems(self):
        return self.X.float() , self.y, self.days_y
        
    def concat(self, EEGDict, days_range):
        X = []
        y = []
        days_y = []
        for day, d in enumerate(EEGDict[days_range[0]:days_range[1]]):
            X.append(d['segmentedEEG'])
            y.append(d['labels'])
            days_y.append(np.ones_like(d['labels']) * day)

        # Directly concatenate - don't use np.asarray first as it fails with 
        # different trial counts per day (inhomogeneous shapes)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y)
        days_y = np.concatenate(days_y)
        #  one hot encode days labels
        y_temp = np.zeros((days_y.size, days_y.max() + 1))
        y_temp[np.arange(days_y.size), days_y] = 1
        days_y = y_temp
        # One hot encode task labels
        y_temp = np.zeros((y.size, y.max() + 1))
        y_temp[np.arange(y.size), y] = 1
        y = y_temp
        return X, y, days_y


class EEGDataSet_signal(Dataset):
    def __init__(self, EEGDict, days_range=[0,1]):
        
        # Concat dict      
        X, y = self.concat(EEGDict, days_range)
        
        # Convert from numpy to tensor
        self.X = torch.tensor(X)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.y = y

    def __getitem__(self, index):
        return self.X[index].float(), self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def getAllItems(self):
        return self.X.float() , self.y
    
    def concat(self, EEGDict, days_range):
        X = []
        y = []
        for d in EEGDict[days_range[0]:days_range[1]]:
            X.append(d['segmentedEEG'])
            y.append(d['labels'])

        # Directly concatenate - don't use np.asarray first as it fails with 
        # different trial counts per day (inhomogeneous shapes)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y)
        return X, y


def remove_noisy_trials(dictListStacked, amp_thresh, min_trials):
    # Remove noisy trials using amplitude threshold
    new_dict_list = []
    for i, D in enumerate(dictListStacked):
        max_amp = np.amax(np.amax(D['segmentedEEG'], 2), 1)
        min_amp = np.amin(np.amin(D['segmentedEEG'], 2), 1)
        max_tr = max_amp > amp_thresh
        min_tr = min_amp < -amp_thresh
        noisy_trials = [a or b for a, b in zip(max_tr, min_tr)]
        D['segmentedEEG'] = np.delete(D['segmentedEEG'], noisy_trials,axis=0)
        D['labels'] = np.delete(D['labels'], noisy_trials,axis=0)
    #    # One hot the labels
    #     D['labels'][D['labels']==4] = 3
    #     D['labels'] = torch.as_tensor(D['labels']).to(torch.int64) - 1
    #     D['labels'] = F.one_hot(D['labels'], 3)
        if D['segmentedEEG'].shape[0] > min_trials:
                new_dict_list.append(D)

    return new_dict_list


def eegFilters(eegMat, fs, filterLim):
    eegMatFiltered = mne.filter.filter_data(eegMat, fs, filterLim[0], filterLim[1], verbose=0)
    return eegMatFiltered


