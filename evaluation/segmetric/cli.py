#!/usr/bin/env python
import argparse
from pathlib import Path
from time import time
from typing import Tuple

import pandas as pd
import numpy as np

from segmetric.evaluation import Evaluation

def run_evaluation(results_path: Path, dataset_path: Path, subset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the evaluation and returns global and per-sequence results.
    """
    print("Evaluating sequences...")
    evaluator = Evaluation(dataset_root=dataset_path, gt_set=subset)
    metrics_res = evaluator.evaluate(results_path)

    # Process Jaccard and F-score results
    j_metrics = metrics_res.get('J', {})
    j_last_metrics = metrics_res.get('J_last', {})
    f_metrics = metrics_res.get('F', {})
    f_last_metrics = metrics_res.get('F_last', {})
    
    # --- GLOBAL RESULTS ---
    g_measures = [
        'J-Mean', 'J-Recall', 'J-Decay', 
        'F-Mean', 'F-Recall', 'F-Decay',
        'J_last-Mean', 'J_last-Recall', 'J_last-Decay',
        'F_last-Mean', 'F_last-Recall', 'F_last-Decay'
    ]
    g_res_data = [
        np.mean(j_metrics.get("M", np.nan)), np.mean(j_metrics.get("R", np.nan)), np.mean(j_metrics.get("D", np.nan)),
        np.mean(f_metrics.get("M", np.nan)), np.mean(f_metrics.get("R", np.nan)), np.mean(f_metrics.get("D", np.nan)),
        np.mean(j_last_metrics.get("M", np.nan)), np.mean(j_last_metrics.get("R", np.nan)), np.mean(j_last_metrics.get("D", np.nan)),
        np.mean(f_last_metrics.get("M", np.nan)), np.mean(f_last_metrics.get("R", np.nan)), np.mean(f_last_metrics.get("D", np.nan))
    ]
    table_g = pd.DataFrame([g_res_data], columns=g_measures)

    # --- PER-SEQUENCE RESULTS ---
    seq_names = list(j_metrics.get("M_per_object", {}).keys())
    
    j_per_obj_map = j_metrics.get("M_per_object", {})
    j_last_per_obj_map = j_last_metrics.get("M_per_object", {})
    f_per_obj_map = f_metrics.get("M_per_object", {})
    f_last_per_obj_map = f_last_metrics.get("M_per_object", {})

    seq_data = {
        'Sequence': seq_names,
        'J-Mean': [j_per_obj_map.get(name, np.nan) for name in seq_names],
        'F-Mean': [f_per_obj_map.get(name, np.nan) for name in seq_names],
        'J_last-Mean': [j_last_per_obj_map.get(name, np.nan) for name in seq_names],
        'F_last-Mean': [f_last_per_obj_map.get(name, np.nan) for name in seq_names],
    }
    
    table_seq = pd.DataFrame(seq_data)
    
    return table_g, table_seq

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VOS model on the VOST dataset.")
    parser.add_argument('--results_path', type=Path, required=True, help='Path to the folder containing the sequences folders.')
    parser.add_argument('--dataset_path', type=Path, default=Path('../aot_plus/datasets/VOST'), help='Path to the dataset folder.')
    parser.add_argument('--set', type=str, default='val', choices=['train', 'val', 'test', 'long_videos'], help='Subset to evaluate the results on.')
    parser.add_argument('--re', action='store_true', help='Force re-evaluation even if results CSV files exist.')
    args = parser.parse_args()

    time_start = time()
    csv_name_global = args.results_path / f'global_results-{args.set}.csv'
    csv_name_per_sequence = args.results_path / f'per-sequence_results-{args.set}.csv'

    print(f"Evaluating {args.results_path}")

    if not args.re and csv_name_global.exists() and csv_name_per_sequence.exists():
        print('Using pre-computed results...')
        table_g = pd.read_csv(csv_name_global)
        table_seq = pd.read_csv(csv_name_per_sequence)
    else:
        table_g, table_seq = run_evaluation(args.results_path, args.dataset_path, args.set)
        table_g.to_csv(csv_name_global, index=False, float_format="%.6f")
        print(f'Global results saved to {csv_name_global}')
        table_seq.to_csv(csv_name_per_sequence, index=False, float_format="%.6f")
        print(f'Per-sequence results saved to {csv_name_per_sequence}')

    print("\n" + "="*80)
    print(f" Global results for '{args.set}' ".center(80, " "))
    print("="*80)
    print(table_g.to_string(index=False))
    
    print("\n" + "="*80)
    print(f" Per-sequence results for '{args.set}' ".center(80, " "))
    print("="*80)
    print(table_seq.to_string(index=False))

    total_time = time() - time_start
    print(f'\nTotal time: {total_time:.2f} seconds')

if __name__ == '__main__':
    main()
