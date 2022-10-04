import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve


class MPMLearning:

    ## Cross validation (regression)
    def multi_k_cv_re(self, x, y, model_para, n_spilt=3, n_repeat=3, random_state=0, n_jobs=4, multioutput="uniform_average"):

        root = tk.Tk()
        root.withdraw()
        Folderpath = filedialog.askdirectory()

        Y_tests, Y_pres, MAE, MSE, RMSE, R2 = {}, {}, {}, {}, {}, {}
        RSK = RepeatedKFold(n_splits=n_spilt, n_repeats=n_repeat, random_state=random_state)
        cv = [(t, v) for (t, v) in RSK.split(x, y)]
        model_para = model_para

        process_fold = MPMLearning.process_allocation(self, n_spilt * n_repeat, n_jobs)
        pool = mp.Pool(n_jobs)
        tasks = [pool.apply_async(MPMLearning.fold_cv_training_re, args=(self, x, y, k, fold, cv, model_para, multioutput)) for k, fold in enumerate(process_fold)]
        pool.close()
        pool.join()

        results = [task.get() for task in tasks]
        for result in results:
            Y_tests.update((result[0]))
            Y_pres.update(result[1])
            MAE.update(result[2])
            MSE.update(result[3])
            RMSE.update(result[4])
            R2.update(result[5])

        Y_tests = sorted(Y_tests.items(), key=lambda x: x[0])
        Y_tests = [value[1] for value in Y_tests]
        Y_pres = sorted(Y_pres.items(), key=lambda x: x[0])
        Y_pres = [value[1] for value in Y_tests]
        MAE = sorted(MAE.items(), key=lambda x: x[0])
        MAE = [value[1] for value in MAE]
        MSE = sorted(MSE.items(), key=lambda x: x[0])
        MSE = [value[1] for value in MSE]
        RMSE = sorted(RMSE.items(), key=lambda x: x[0])
        RMSE = [value[1] for value in RMSE]
        R2 = sorted(R2.items(), key=lambda x: x[0])
        R2 = [value[1] for value in R2]

        mean_MAE = np.mean(MAE)
        mean_MSE = np.mean(MSE)
        mean_RMSE = np.mean(RMSE)
        mean_R2 = np.mean(R2)

        result_mkcvre = {'MAE':MAE, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
        result_mkcvre = pd.DataFrame(result_mkcvre)
        result_mkcvre_mean = {'MAE':mean_MAE, 'MSE': mean_MSE, 'RMSE': mean_RMSE, 'R2': mean_R2}
        result_mkcvre_mean = pd.DataFrame([result_mkcvre_mean], index=['Average'])
        result_mkcvre = pd.concat([result_mkcvre_mean, result_mkcvre], axis=0)
        result_mkcvre.to_csv(Folderpath + '/Results of multi kfold re.csv')
        print('Results of multi_k_cv_re are placed in the '+Folderpath)

        return result_mkcvre


    ## Cross validation (classification)
    def multi_k_cv(self, x, y, model_para, n_spilt=3, n_repeat=3, random_state=0, n_jobs=4, average='binary', pos_label=None, labels=None):

        root = tk.Tk()
        root.withdraw()
        Folderpath = filedialog.askdirectory()

        Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score = {}, {}, {}, {}, {}, {}
        RSK = RepeatedStratifiedKFold(n_splits=n_spilt, n_repeats=n_repeat, random_state=random_state)
        cv = [(t, v) for (t, v) in RSK.split(x, y)]
        model_para = model_para

        process_fold = MPMLearning.process_allocation(self, n_spilt * n_repeat, n_jobs)
        pool = mp.Pool(n_jobs)
        tasks = [pool.apply_async(MPMLearning.fold_cv_training, args=(self, x, y, k, fold, cv, model_para, average, pos_label, labels)) for k, fold in enumerate(process_fold)]
        pool.close()
        pool.join()

        results = [task.get() for task in tasks]
        for result in results:
            Y_tests.update((result[0]))
            Y_pres.update(result[1])
            Accuracy.update((result[2]))
            Precision.update(result[3])
            Recall.update(result[4])
            F1_score.update(result[5])

        Y_tests = sorted(Y_tests.items(), key=lambda x: x[0])
        Y_tests = [value[1] for value in Y_tests]
        Y_pres = sorted(Y_pres.items(), key=lambda x: x[0])
        Y_pres = [value[1] for value in Y_pres]
        Accuracy = sorted(Accuracy.items(), key=lambda x: x[0])
        Accuracy = [value[1] for value in Accuracy]
        Precision = sorted(Precision.items(), key=lambda x: x[0])
        Precision = [value[1] for value in Precision]
        Recall = sorted(Recall.items(), key=lambda x: x[0])
        Recall = [value[1] for value in Recall]
        F1_score = sorted(F1_score.items(), key=lambda x: x[0])
        F1_score = [value[1] for value in F1_score]

        mean_Accuracy = np.mean(Accuracy)
        mean_Precision = np.mean(Precision)
        mean_Recall = np.mean(Recall)
        mean_F1_score = np.mean(F1_score)

        result_mkcv = {'Accuracy':Accuracy, 'Precision':Precision, 'Recall':Recall, 'F1_score':F1_score}
        result_mkcv = pd.DataFrame(result_mkcv)
        result_mkcv_mean = {'Accuracy':mean_Accuracy, 'Precision': mean_Precision, 'Recall': mean_Recall, 'F1_score': mean_F1_score}
        result_mkcv_mean = pd.DataFrame([result_mkcv_mean], index=['Average'])
        result_mkcv = pd.concat([result_mkcv_mean, result_mkcv], axis=0)
        result_mkcv.to_csv(Folderpath + '/Results of multi kfold.csv')
        print('Results of multi_k_cv are placed in the '+Folderpath)

        return result_mkcv


    def fold_cv_training_re(self, x, y, k, fold, cv, model_para, multioutput):

        start_fold, end_fold = fold[0], fold[1]
        Y_tests, Y_pres, MAE, MSE, RMSE, R2 = {}, {}, {}, {}, {}, {}

        for i in range(start_fold, end_fold):
            (train_id, test_id) = cv[i]
            x_train, x_test, y_train, y_test = x[train_id], x[test_id], y[train_id], y[test_id]

            estimator = model_para
            estimator.fit(x_train, y_train)

            y_pre = estimator.predict(x_test)
            Y_tests[i] = y_test
            Y_pres[i] = y_pre
            MAE[i] = mean_absolute_error(y_test, y_pre, multioutput=multioutput)
            MSE[i] = mean_squared_error(y_test, y_pre, multioutput=multioutput)
            RMSE[i] = MSE[i]**0.5
            R2[i] = r2_score(y_test, y_pre, multioutput=multioutput)

        return Y_tests, Y_pres, MAE, MSE, RMSE, R2


    def fold_cv_training(self, x, y, k, fold, cv, model_para, average, pos_label, labels):

        start_fold, end_fold = fold[0], fold[1]
        Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score = {}, {}, {}, {}, {}, {}

        for i in range(start_fold, end_fold):
            (train_id, test_id) = cv[i]
            x_train, x_test, y_train, y_test = x[train_id], x[test_id], y[train_id], y[test_id]

            estimator = model_para
            estimator.fit(x_train, y_train)

            y_pre = estimator.predict(x_test)
            Y_tests[i] = y_test
            Y_pres[i] = y_pre
            Accuracy[i] = estimator.score(x_test, y_test)
            Precision[i] = precision_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)
            Recall[i] = recall_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)
            F1_score[i] = f1_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)

        return Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score


    def process_allocation(self, n_fold, n_jobs):

        process_fold_num = n_fold // n_jobs
        fold_legacy = n_fold % n_jobs
        first_fold = 0
        process_fold = []

        process_fold_nums = [process_fold_num for i in range(n_jobs)]
        for i in range(fold_legacy):
            process_fold_nums[i] += 1

        for i in range(len(process_fold_nums)):
            process_fold.append((first_fold, first_fold + process_fold_nums[i]))
            first_fold = first_fold + process_fold_nums[i]

        return process_fold


    ## Large scale training (regression)
    def multi_training_re(self, x, y, model_para, test_size=0.3, train_num=100, n_jobs=4, random_state=None, multioutput="uniform_average"):

        root = tk.Tk()
        root.withdraw()
        Folderpath = filedialog.askdirectory()

        Y_tests, Y_pres, MAE, MSE, RMSE, R2 = {}, {}, {}, {}, {}, {}
        SSS = ShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
        cv = [(t, v) for (t, v) in SSS.split(x, y)]
        model_para = model_para
        process_num = MPMLearning.process_rep_allocation(self, train_num, n_jobs)
        pool = mp.Pool(n_jobs)
        tasks = [pool.apply_async(MPMLearning.repeat_training_re, args=(self, x, y, cv, test_size, train_num, model_para, k, repeat, multioutput)) for k, repeat in
                 enumerate(process_num)]
        pool.close()
        pool.join()
        results = [task.get() for task in tasks]
        for result in results:
            Y_tests.update((result[0]))
            Y_pres.update(result[1])
            MAE.update(result[2])
            MSE.update(result[3])
            RMSE.update(result[4])
            R2.update(result[5])

        Y_tests = sorted(Y_tests.items(), key=lambda x: x[0])
        Y_tests = [value[1] for value in Y_tests]
        Y_pres = sorted(Y_pres.items(), key=lambda x: x[0])
        Y_pres = [value[1] for value in Y_pres]
        MAE = sorted(MAE.items(), key=lambda x: x[0])
        MAE = [value[1] for value in MAE]
        MSE = sorted(MSE.items(), key=lambda x: x[0])
        MSE = [value[1] for value in MSE]
        RMSE = sorted(RMSE.items(), key=lambda x: x[0])
        RMSE = [value[1] for value in RMSE]
        R2 = sorted(R2.items(), key=lambda x: x[0])
        R2 = [value[1] for value in R2]

        mean_MAE = np.mean(MAE)
        mean_MSE = np.mean(MSE)
        mean_RMSE = np.mean(RMSE)
        mean_R2 = np.mean(R2)

        result_mrtre = {'MAE':MAE, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
        result_mrtre = pd.DataFrame(result_mrtre)
        result_mrtre_mean = {'MAE':mean_MAE, 'MSE': mean_MSE, 'RMSE': mean_RMSE, 'R2': mean_R2}
        result_mrtre_mean = pd.DataFrame([result_mrtre_mean], index=['Average'])
        result_mrtre = pd.concat([result_mrtre_mean, result_mrtre], axis=0)
        result_mrtre.to_csv(Folderpath + '/Results of multi train re.csv')
        print('Results of multi_training_re are placed in the ' + Folderpath)

        return result_mrtre


    ## Large scale training (classification)
    def multi_training(self, x, y, model_para, test_size=0.3, train_num=100, n_jobs=4, random_state=None, average='binary', pos_label=None, labels=None):

        root = tk.Tk()
        root.withdraw()
        Folderpath = filedialog.askdirectory()

        Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score = {}, {}, {}, {}, {}, {}
        SSS = StratifiedShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
        cv = [(t, v) for (t, v) in SSS.split(x, y)]
        model_para = model_para
        process_num = MPMLearning.process_rep_allocation(self, train_num, n_jobs)
        pool = mp.Pool(n_jobs)
        tasks = [pool.apply_async(MPMLearning.repeat_training, args=(self, x, y, cv, test_size, train_num, model_para, k, repeat, average, pos_label, labels)) for k, repeat in
                 enumerate(process_num)]
        pool.close()
        pool.join()
        results = [task.get() for task in tasks]
        for result in results:
            Y_tests.update((result[0]))
            Y_pres.update(result[1])
            Accuracy.update((result[2]))
            Precision.update(result[3])
            Recall.update(result[4])
            F1_score.update(result[5])

        Y_tests = sorted(Y_tests.items(), key=lambda x: x[0])
        Y_tests = [value[1] for value in Y_tests]
        Y_pres = sorted(Y_pres.items(), key=lambda x: x[0])
        Y_pres = [value[1] for value in Y_pres]
        Accuracy = sorted(Accuracy.items(), key=lambda x: x[0])
        Accuracy = [value[1] for value in Accuracy]
        Precision = sorted(Precision.items(), key=lambda x: x[0])
        Precision = [value[1] for value in Precision]
        Recall = sorted(Recall.items(), key=lambda x: x[0])
        Recall = [value[1] for value in Recall]
        F1_score = sorted(F1_score.items(), key=lambda x: x[0])
        F1_score = [value[1] for value in F1_score]

        mean_Accuracy = np.mean(Accuracy)
        mean_Precision = np.mean(Precision)
        mean_Recall = np.mean(Recall)
        mean_F1_score = np.mean(F1_score)

        result_mrt = {'Accuracy':Accuracy, 'Precision':Precision, 'Recall':Recall, 'F1_score':F1_score}
        result_mrt = pd.DataFrame(result_mrt)
        result_mrt_mean = {'Accuracy':mean_Accuracy, 'Precision': mean_Precision, 'Recall': mean_Recall, 'F1_score': mean_F1_score}
        result_mrt_mean = pd.DataFrame([result_mrt_mean], index=['Average'])
        result_mrt = pd.concat([result_mrt_mean, result_mrt], axis=0)
        result_mrt.to_csv(Folderpath+'/Results of multi train.csv')
        print('Results of multi_training are placed in the ' + Folderpath)

        return result_mrt


    def repeat_training_re(self, x, y, cv, test_size, train_num, model_para, k, repeat, multioutput):

        start_repeat, end_repeat = repeat[0], repeat[1]
        Y_tests, Y_pres, MAE, MSE, RMSE, R2 = {}, {}, {}, {}, {}, {}

        for i in range(start_repeat, end_repeat):
            for train_id, test_id in cv:
                x_train, x_test = x[train_id], x[test_id]
                y_train, y_test = y[train_id], y[test_id]

                estimator = model_para
                estimator.fit(x_train, y_train)

                y_pre = estimator.predict(x_test)
                Y_tests[i] = y_test
                Y_pres[i] = y_pre
                MAE[i] = mean_absolute_error(y_test, y_pre, multioutput=multioutput)
                MSE[i] = mean_squared_error(y_test, y_pre, multioutput=multioutput)
                RMSE[i] = MSE[i] ** 0.5
                R2[i] = r2_score(y_test, y_pre, multioutput=multioutput)

        return Y_tests, Y_pres, MAE, MSE, RMSE, R2


    def repeat_training(self, x, y, cv, test_size, train_num, model_para, k, repeat, average, pos_label, labels):

        start_repeat, end_repeat = repeat[0], repeat[1]
        Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score = {}, {}, {}, {}, {}, {}

        for i in range(start_repeat, end_repeat):
            for train_id, test_id in cv:
                x_train, x_test = x[train_id], x[test_id]
                y_train, y_test = y[train_id], y[test_id]

                estimator = model_para
                estimator.fit(x_train, y_train)

                pre_y = estimator.predict_proba(x_test)[:, 1]
                y_pre = estimator.predict(x_test)
                Y_tests[i] = y_test
                Y_pres[i] = y_pre
                Accuracy[i] = estimator.score(x_test, y_test)
                Precision[i] = precision_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)
                Recall[i] = recall_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)
                F1_score[i] = f1_score(y_test, estimator.predict(x_test), average=average, pos_label=pos_label, labels=labels)

        return Y_tests, Y_pres, Accuracy, Precision, Recall, F1_score


    def process_rep_allocation(self, n_renum, n_jobs):

        process_re_num = n_renum // n_jobs
        renum_legacy = n_renum % n_jobs
        first_re = 0
        process_num = []

        process_re_nums = [process_re_num for i in range(n_jobs)]
        for i in range(renum_legacy):
            process_re_nums[i] += 1
        for i in range(len(process_re_nums)):
            process_num.append((first_re, first_re + process_re_nums[i]))
            first_re = first_re + process_re_nums[i]

        return process_num


    ## learning curve
    def learning_curve_plot(self, x, y, model_para=None, cv=None, n_jobs=4, train_step=np.linspace(.05, 1., 20),
                            verbose=0, plot=True, scoring=None):

        root = tk.Tk()
        root.withdraw()
        Folderpath = filedialog.askdirectory()

        train_step, train_scores, test_scores = learning_curve(model_para, x, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_step, verbose=verbose, scoring=scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if plot:

            max_train_mean = max(train_scores_mean)
            max_test_mean = max(test_scores_mean)
            max_train_std = max(train_scores_std)
            max_test_std = max(test_scores_std)

            a = max(max_train_mean, max_test_mean)
            b = max(max_train_std, max_test_std)
            c = min(max_train_mean, max_test_mean)

            ylim = (c - 3*b, a + 3*b)

            plt.figure()
            plt.title(u'Learning curve')
            plt.ylim(*ylim)
            plt.xlabel(u'train sample')
            plt.ylabel(u'score')
            plt.grid()
            plt.fill_between(train_step, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.2, color='#1E7DC1')
            plt.fill_between(train_step, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                             alpha=0.2, color='#CC4D65')
            plt.plot(train_step, train_scores_mean, 'o-', color='#1E7DC1', label=u'train score', markersize=5)
            plt.plot(train_step, test_scores_mean, 'o-', color='#CC4D65', label=u'cross validation score', markersize=5)
            plt.legend(loc='best')
            plt.draw()
            plt.savefig(Folderpath+'/learn_curve.pdf')
            plt.show()

        The_median_point_of_convergence = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
        The_width_of_convergence_interval = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

        result = {'Average (train score)':train_scores_mean, 'Std (train score)':train_scores_std, 'Average (cross validation score)':test_scores_mean, 'Std (cross validation score)':test_scores_std}
        result = pd.DataFrame(result)
        result.to_csv(Folderpath + '/Results of learning curve.csv')
        print('Result of learning_curve_plot are placed in the ' + Folderpath)

        return result

