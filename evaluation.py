import json
import logging
from os.path import join
import numpy as np
import pandas as pd
import json
import sys
import os
# from pycocotools.mask import *
import numpy as np
import time
import pandas as pd
from joblib import Parallel, delayed
import zipfile
import logging
import pdb

import os
def run_evaluation(standard_path_or_dict, submit_path_or_dict, require_detail=False):
    '''根据输入的gt路径（或字典）、预测结果路径（或字典），运行评测，返回得分'''
    if isinstance(standard_path_or_dict, str):
        standard_json = json.load(open(standard_path_or_dict))
    else:
        standard_json = standard_path_or_dict
    if isinstance(submit_path_or_dict, str):
        submit_json = json.load(open(submit_path_or_dict))
    else:
        submit_json = submit_path_or_dict
    anet = ANETdetection(standard_json, submit_json, verbose=True)
    anet.evaluate()
    per_iou_score = {}
    for i in [0, 5, 9]:
        thr = 0.5 + 0.05 * i
        per_iou_score['mAP@{:.2f}'.format(thr)] = float(anet.mAP[i]) * 100
    if not require_detail:
        return anet.average_mAP * 100, per_iou_score
    else:
        return anet.average_mAP * 100, per_iou_score, anet.ap


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
        + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i, :], candidate_segments)

    return tiou
class ANETdetection(object):

    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None

        self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            # print('[INIT] Loaded annotations from {} subset.'.format(self.subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(
                self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        if isinstance(ground_truth_filename, str):
            with open(ground_truth_filename, 'r') as fobj:
                data = json.load(fobj)
        else:
            print('already a json object')
            data = ground_truth_filename

        # Checking format

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst, difficult_lst = [], [], [], [], []
        for videoid, v in data.items():
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(ann['segment'][0])
                t_end_lst.append(ann['segment'][1])
                label_lst.append(activity_index[ann['label']])
                difficult_lst.append(ann['difficult'])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     'difficult': difficult_lst})
        return ground_truth, activity_index         #ground_truth是一个dataframe形式，activity_index是类别和数字的字典形式

    def _import_prediction(self, prediction_filename, keep_num=100):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        keep_num : int
            Number of predictions kept in each video.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        if isinstance(prediction_filename, str):
            with open(prediction_filename, 'r') as fobj:
                data = json.load(fobj)
        else:
            data = prediction_filename
        # Checking format...
        if 'results' in data:
            data = data['results']

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data.items():
            if videoid in self.blocked_videos:
                continue
            if keep_num and len(v) > 0:
                scores = [x['score'] for x in v]
                sort_idx = np.array(scores).argsort()[::-1][:keep_num]
                v = [v[idx] for idx in sort_idx]
            for result in v:
                try:
                    label = self.activity_index[result['label']]
                except:
                    print('1')
                video_lst.append(videoid)
                t_start_lst.append(result['segment'][0])
                t_end_lst.append(result['segment'][1])
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' %
                  label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(
                    cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in list(self.activity_index.items()))

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet detection task.')
            logging.warn('mAP {}'.format(self.mAP))
            logging.warn('\tAverage-mAP: {}'.format(self.average_mAP))
def compute_average_precision_detection(ground_truth, prediction, keep_num=None, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Ignore difficult instances.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = sum(ground_truth['difficult'] == 0)   # float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    if keep_num is not None:
        prediction = prediction[:keep_num]

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(
                this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                if this_gt.loc[jdx]['difficult'] == 0:

                    tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap


def run_evaluation(standard_path_or_dict, submit_path_or_dict, require_detail=False):
    '''根据输入的gt路径（或字典）、预测结果路径（或字典），运行评测，返回得分'''
    if isinstance(standard_path_or_dict, str):
        standard_json = json.load(open(standard_path_or_dict))
    else:
        standard_json = standard_path_or_dict
    if isinstance(submit_path_or_dict, str):
        submit_json = json.load(open(submit_path_or_dict))
    else:
        submit_json = submit_path_or_dict
    anet = ANETdetection(standard_json, submit_json, verbose=True)
    anet.evaluate()
    per_iou_score = {}
    for i in [0, 5, 9]:
        thr = 0.5 + 0.05 * i
        per_iou_score['mAP@{:.2f}'.format(thr)] = float(anet.mAP[i]) * 100
    if not require_detail:
        return anet.average_mAP * 100, per_iou_score
    else:
        return anet.average_mAP * 100, per_iou_score, anet.ap


error_msg = {
    1: "Bad input file.",
    2: "Wrong input file format.",
}


def run_evaluation_for_zip(standard_path, submit_path, accept_zip=True):
    '''这个是测评服务器上运行的版本，读取的是zip文件'''
    if accept_zip:
        # zip文件内不要包含文件夹，否走会报错
        with zipfile.ZipFile(standard_path) as standard_zip:
            standard_namelist = standard_zip.namelist()
            if len(standard_namelist) != 1:
                raise IOError(
                    'Zip file contains unexpected number of files '+str(standard_namelist))
            standard_json = json.load(standard_zip.open(standard_namelist[0]))
            print('open {} from zip file {}'.format(
                standard_namelist[0], standard_path))

        with zipfile.ZipFile(submit_path) as submit_zip:
            submit_namelist = submit_zip.namelist()
            if len(submit_namelist) != 1:
                raise IOError(
                    'Zip file contains unexpected number of files '+str(submit_namelist))
            submit_json = json.load(submit_zip.open(submit_namelist[0]))
            print('open {} from zip file {}'.format(
                submit_namelist[0], submit_path))
    else:
        standard_json = json.load(open(standard_path))
        submit_json = json.load(open(submit_path))

    anet = ANETdetection(standard_json, submit_json, verbose=True)
    anet.evaluate()
    per_iou_score = {}
    for i in [0, 5, 9]:
        thr = 0.5 + 0.05 * i
        per_iou_score['mAP@{:.2f}'.format(thr)] = float(anet.mAP[i])
    return anet.average_mAP, per_iou_score
if __name__ == "__main__":
    anno_dict = json.load(
                        open(join("./bmn/data/tianchi_annotation/", 'train_annotations.json')))
    this_set_video_ids = [x.strip() for x in open(
                        join('val_video_name.txt')).readlines()]
    anno_dict_for_this_set = {
                        k: anno_dict[k] for k in this_set_video_ids}
    logging.info('evaluation')
    with open("results.json","r") as f:
        res_dicts = json.load(f)
    run_evaluation(anno_dict_for_this_set, res_dicts)



# list1 = []
# f = open('val_video_name.txt', 'w')
# path = "/home/yanhao/pycharm/BMN_PGCM_bisaitijiaobanben/tcdata/i3d_feature/"
# for root, dirs, files in os.walk(path):
#     for file in files:
#         a = file.split(".")[0]
#         f.write(a)
#         f.write('\n')
