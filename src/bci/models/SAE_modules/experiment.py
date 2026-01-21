from .utils import *
import pickle 
import pytorch_lightning as pl

import os
import time
import torch
import random

from .models import convolution_AE         
from .properties import hyper_params as params
from .utils import EEGDataSet_signal_by_day
from .denoiser import Denoiser

class NoEEGDataException(Exception):
    pass

class Experiment():

    def __init__(self, exp_name: str, data_extractor, model_adjustments, train_days_range: list, n_iterations: int, mode = 'supervised'):

        self.experiment_name = exp_name
        self.data_extractor = data_extractor
        self.EEG_data = None
        self.subs = None
        self.train_days_range = train_days_range
        self.n_iterations = n_iterations
        self.task_file = None
        self.origin_file = None

        self.mode = mode
        self.model_adjustments = model_adjustments


    def extract_data(self):
        self.EEG_data = self.data_extractor.get_EEG_dict()
        self.subs = self.EEG_data.keys()

    def get_subs(self):
        return self.subs

    def _save_iteration_result(self, itr, task_result, origin_result):
        last_iteration_task_file = self.task_file
        last_iteration_origin_file = self.origin_file

        self.task_file = f'{self}results/task_{itr}_{self.experiment_name}.pickle'
        self.origin_file = f'{self}results/origin_{itr}_{self.experiment_name}.pickle'
        
        try:
            f_task = open(self.task_file, 'wb')
            f_origin = open(self.origin_file, 'wb')
            pickle.dump(task_result, f_task)
            pickle.dump(origin_result, f_origin)
        except Exception as e:
            print(e)
            print("Couldn't save to file")
        finally:
            f_task.close()
            f_origin.close()
            if last_iteration_task_file:
                os.remove(last_iteration_task_file)   
                os.remove(last_iteration_origin_file) 


    def _origin_day_clf(self, EEGdict, AE_denoiser):
        # Use day zero classifier for classifying the reconstructed eeg per day
        
        # get relevant data
        test_dataset = EEGDataSet_signal_by_day(EEGdict, [0, len(EEGdict)])
        orig_signal, _, labels = test_dataset.getAllItems()
        rec_signal = AE_denoiser.denoise(test_dataset)
        res_signal = orig_signal - rec_signal
        
        # change labels from 1hot to int
        labels = np.argmax(labels, axis=1)
        
        score_orig, _ = csp_score(np.float64(orig_signal.detach().numpy()), labels, cv_N = 5, classifier = False)
        score_rec, _ = csp_score(np.float64(rec_signal), labels, cv_N = 5, classifier = False)
        score_res, _ = csp_score(np.float64(res_signal), labels, cv_N = 5, classifier = False)
        return score_orig, score_rec, score_res


    def training_loop(self, train_days, sub):
        
        single_sub_EEG_data = self.EEG_data[sub]

        # check if enough train days exists
        if train_days[1] >= len(single_sub_EEG_data):
            raise Exception("Not enough training days")

        # Shuffle the days
        random.shuffle(single_sub_EEG_data)
        # Train Dataset
        train_dataset = EEGDataSet_signal_by_day(single_sub_EEG_data, train_days)
        train_x, y, days_y = train_dataset.getAllItems()
        y = np.argmax(y, -1)
        test_days = [train_days[1], len(single_sub_EEG_data)]

        # Create test Datasets
        test_dataset = EEGDataSet_signal_by_day(single_sub_EEG_data, test_days)

        # get data
        signal_test, y_test, _ = test_dataset.getAllItems()
        y_test = np.argmax(y_test, -1)

        # Fit AE_denoiser and use fitted model
        denoiser = Denoiser(self.model_adjustments, self.mode)
        denoiser.fit(train_dataset)
        denoised_signal = denoiser.denoise(test_dataset)
 
        ws_ae_train, day_zero_AE_clf = csp_score(np.float64(denoiser.denoise(train_dataset)), y, cv_N=5, classifier=False)
        ws_train, day_zero_bench_clf = csp_score(np.float64(train_x.detach().numpy()), y, cv_N=5, classifier=False)

        # Use models
        # within session cv on the test set (mean on test set)
        ws_test, _ = csp_score(np.float64(signal_test.detach().numpy()), y_test, cv_N=5, classifier = False)
        # Using day 0 classifier for test set inference (mean on test set)
        bs_test = csp_score(np.float64(signal_test.detach().numpy()), y_test, cv_N=5, classifier=day_zero_bench_clf)
        # Using day 0 classifier + AE for test set inference (mean on test set)
        bs_ae_test = csp_score(denoised_signal, y_test, cv_N=5, classifier=day_zero_AE_clf)
        
        return ws_train, ws_ae_train, ws_test, bs_test, bs_ae_test, denoiser



    def run_all_subs_multi_iterations(self):
        '''    
        This function runs multi iterations experiment over all subjects.
        The experiment runs all ranges of traning days from 0-`train_days_range[0]` to 0-`train_days_range[1]`.
        Every iteration models are trained for all ranges of training days and all subjects.
        the function saves 2 dictionaries to 2 files:
            task clasification results dictionary
            origin day clasification results dictionary
        The function returns the 2 file pathes
        '''
        if not self.EEG_data:
            raise NoEEGDataException('To run experiment you must first apply extract_data()')

        ts = time.strftime("%Y%m%d-%H%M%S")
        print(f'START EXPERIMENT!!! {ts}\n')

        task_iter_dict = {} # keys: iterations, vals: dict of dicts of dicts of scores for each sub
        origin_iter_dict = {} 
        
        for itr in range(self.n_iterations):
            task_days_range_dict = {} # keys: train days range, vals: dict of dicts of scores for each sub
            origin_days_range_dict = {}
            
            for last_train_day in range(self.train_days_range[0],self.train_days_range[1]):
                task_sub_dict = {} # keys: sub, vals: dict of list of the scores dicts for each sub
                origin_sub_dict = {}
                
                curr_days_rng=[0, last_train_day] # determine the current range for training days 
                rng_str = '-'.join(str(e) for e in curr_days_rng) # turn days range list to str to use as key name
                    
                for sub in list(self.subs):
                    print(f'Running {self.experiment_name}')
                    print(f'\niter: {itr}, last training day: {last_train_day}, sub: {sub}...\n')
                    
                    task_per_sub_scores_dict = {} # keys: method(ws,bs,AE), vals: scores
                    origin_per_sub_scores_dict = {} # keys: signal(orig,rec,res), vals: scores
                    
                    print('training model...\n')
                    try:
                        ws_train, ws_ae_train, ws_test, bs_test, ae_test, AE_denoiser = self.training_loop(curr_days_rng, sub)
                    except Exception as e:
                        print(f'Can\'t train a model for sub: {sub} with last training day: {last_train_day} because:')
                        print(e)
                        continue
                    
                    # Add task classification results
                    task_per_sub_scores_dict['ws_train'] = ws_train
                    task_per_sub_scores_dict['ae_train'] = ws_ae_train
                    task_per_sub_scores_dict['ws_test'] = ws_test
                    task_per_sub_scores_dict['bs_test'] = bs_test
                    task_per_sub_scores_dict['ae_test'] = ae_test
                    
                    # Day classfication using residuals original and recontrusted EEG
                    print('classifying origin day...')
                    orig_score, rec_score, res_score = self._origin_day_clf(self.EEG_data[sub], AE_denoiser)
                    origin_per_sub_scores_dict['orig'] = orig_score
                    origin_per_sub_scores_dict['rec'] = rec_score
                    origin_per_sub_scores_dict['res'] = res_score
                
                    task_sub_dict[sub] = task_per_sub_scores_dict
                    origin_sub_dict[sub] = origin_per_sub_scores_dict

                task_days_range_dict[rng_str] = task_sub_dict
                origin_days_range_dict[rng_str] = origin_sub_dict
                    
            task_iter_dict[itr] = task_days_range_dict
            origin_iter_dict[itr] = origin_days_range_dict
            
            # save to file
            print('save to file...')
            self._save_iteration_result(itr,task_iter_dict,origin_iter_dict)

            print(f'stopped after {itr+1} iterations')
        
    
    def run_experiment(self):

        if not self.EEG_data:
           self.extract_data() 
        
        self.run_all_subs_multi_iterations()