"""
Core evaluation logic for video object segmentation.
"""
import multiprocessing as mp
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
from tqdm import tqdm

from segmetric.dataset import Dataset
from segmetric.metrics import db_eval_iou, db_eval_boundary
from segmetric.results import Results
from segmetric import utils

def _evaluate_sequence(sequence_name: str, dataset_root: Path, gt_set: str, results_path: Path) -> Dict[str, Any]:
    """
    Evaluates a single sequence and computes its metrics.

    This function is designed to be called by a multiprocessing pool.
    """
    dataset = Dataset(root=dataset_root, subset=gt_set, sequences=[sequence_name])
    results = Results(root_dir=results_path)
    
    all_gt_masks, _, all_masks_id = dataset.get_all_masks(sequence_name, True)
    all_gt_masks = all_gt_masks[:, 1:-1, :, :]
    all_masks_id = all_masks_id[1:-1]
    
    if not all_masks_id or all_gt_masks.shape[0] == 0:
        return {}
        
    all_res_masks = results.read_masks(sequence_name, all_masks_id)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Evaluate Jaccard Index
        j_metrics_res = _evaluate_semisupervised(all_gt_masks, all_res_masks)
        
        # --- START: MODIFIED F-SCORE CALCULATION ---
        # Initialize an array to store F-scores for each object and frame
        f_metrics_res = np.zeros(all_gt_masks.shape[:2])
        # Iterate over each object before calling db_eval_boundary
        for i in range(all_gt_masks.shape[0]):
            # Pass a 3D tensor (frames, height, width) for each object
            f_metrics_res[i, :] = db_eval_boundary(all_gt_masks[i], all_res_masks[i])
        # --- END: MODIFIED F-SCORE CALCULATION ---

    # Initialize results dictionary
    sequence_results: Dict[str, Any] = {
        'J': {"M": [], "R": [], "D": [], "M_per_object": {}}, 'J_last': {"M": [], "R": [], "D": [], "M_per_object": {}},
        'F': {"M": [], "R": [], "D": [], "M_per_object": {}}, 'F_last': {"M": [], "R": [], "D": [], "M_per_object": {}},
    }

    # Calculate statistics
    num_eval_frames = len(all_masks_id)
    last_quarter_ind = int(floor(num_eval_frames * 0.75))

    for i in range(all_gt_masks.shape[0]):
        obj_name = f'{sequence_name}_{i+1}'
        
        # J-score stats
        jm, jr, jd = utils.db_statistics(j_metrics_res[i])
        sequence_results['J']["M"].append(jm); sequence_results['J']["R"].append(jr); sequence_results['J']["D"].append(jd)
        sequence_results['J']["M_per_object"][obj_name] = jm
        
        jm_last, jr_last, jd_last = utils.db_statistics(j_metrics_res[i][last_quarter_ind:])
        sequence_results['J_last']["M"].append(jm_last); sequence_results['J_last']["R"].append(jr_last); sequence_results['J_last']["D"].append(jd_last)
        sequence_results['J_last']["M_per_object"][obj_name] = jm_last

        # F-score stats
        fm, fr, fd = utils.db_statistics(f_metrics_res[i])
        sequence_results['F']["M"].append(fm); sequence_results['F']["R"].append(fr); sequence_results['F']["D"].append(fd)
        sequence_results['F']["M_per_object"][obj_name] = fm
        
        fm_last, fr_last, fd_last = utils.db_statistics(f_metrics_res[i][last_quarter_ind:])
        sequence_results['F_last']["M"].append(fm_last); sequence_results['F_last']["R"].append(fr_last); sequence_results['F_last']["D"].append(fd_last)
        sequence_results['F_last']["M_per_object"][obj_name] = fm_last
        
    return sequence_results

def _evaluate_semisupervised(all_gt_masks: np.ndarray, all_res_masks: np.ndarray) -> np.ndarray:
    """Helper to compute IoU for a sequence."""
    if all_res_masks.shape[0] < all_gt_masks.shape[0]:
        padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]), dtype=all_res_masks.dtype)
        all_res_masks = np.concatenate([all_res_masks, padding], axis=0)
    elif all_res_masks.shape[0] > all_gt_masks.shape[0]:
        print(f"\nWarning: Found more predicted objects than ground-truth objects. Truncating predictions.")
        all_res_masks = all_res_masks[:all_gt_masks.shape[0]]
        
    return db_eval_iou(all_gt_masks, all_res_masks)

class Evaluation:
    def __init__(self, dataset_root: Path, gt_set: str = 'val'):
        self.dataset_root = dataset_root
        self.gt_set = gt_set
        self.dataset = Dataset(root=self.dataset_root, subset=self.gt_set)
        print(f"Evaluating on dataset: {self.dataset_root}")

    def evaluate(self, res_path: Path) -> Dict[str, Any]:
        all_sequences = list(self.dataset.get_sequences())
        pool_args = [(seq, self.dataset_root, self.gt_set, res_path) for seq in all_sequences]
        
        metrics_res = {
            'J': {"M": [], "R": [], "D": [], "M_per_object": {}}, 'J_last': {"M": [], "R": [], "D": [], "M_per_object": {}},
            'F': {"M": [], "R": [], "D": [], "M_per_object": {}}, 'F_last': {"M": [], "R": [], "D": [], "M_per_object": {}},
        }

        print(f"Starting evaluation on {len(all_sequences)} sequences with {mp.cpu_count()} workers...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for seq_result in tqdm(pool.starmap(_evaluate_sequence, pool_args), total=len(all_sequences)):
                if not seq_result:
                    continue
                for key in ['J', 'J_last', 'F', 'F_last']:
                    metrics_res[key]["M"].extend(seq_result[key]["M"])
                    metrics_res[key]["R"].extend(seq_result[key]["R"])
                    metrics_res[key]["D"].extend(seq_result[key]["D"])
                    metrics_res[key]["M_per_object"].update(seq_result[key]["M_per_object"])
        return metrics_res
