import argparse
import numpy as np
import os
import torch
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import TestData
from model import DeepGMR


def evaluate(model, loader, save_results=False, results_dir=None):
    model.eval()

    log_fmt = 'Test: inference time {:.3f}, preprocessing time {:.3f}, loss {:.4f}, ' + \
              'rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}'
    inference_time = 0
    preprocess_time = 0
    losses = 0
    r_errs = 0
    t_errs = 0
    rmses = 0
    N = 0

    if save_results:
        rotations = []
        translations = []
        rotations_gt = []
        translations_gt = []

    start = time()
    for step, (pts1, pts2, T_gt) in enumerate(tqdm(loader, leave=False)):
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            T_gt = T_gt.cuda()
        preprocess_time += time() - start
        N += pts1.shape[0]

        start = time()
        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
            inference_time += time() - start

        losses += loss.item()
        r_errs += r_err.mean().item()
        t_errs += t_err.mean().item()
        rmses += rmse.mean().item()

        if save_results:
            rotations.append(model.T_12[:, :3, :3].cpu().numpy())
            translations.append(model.T_12[:, :3, 3].cpu().numpy())
            rotations_gt.append(T_gt[:, :3, :3].cpu().numpy())
            translations_gt.append(T_gt[:, :3, 3].cpu().numpy())

        start = time()

    log_str = log_fmt.format(
        inference_time / N, preprocess_time / N,
        losses / N, r_errs / N, t_errs / N, rmses / N
    )
    print(log_str)

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'rotations.npy'), np.concatenate(rotations, 0))
        np.save(os.path.join(results_dir, 'translations.npy'), np.concatenate(translations, 0))
        np.save(os.path.join(results_dir, 'rotations_gt.npy'), np.concatenate(rotations_gt, 0))
        np.save(os.path.join(results_dir, 'translations_gt.npy'), np.concatenate(translations_gt, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file')
    parser.add_argument('--results_dir')
    parser.add_argument('--checkpoint')
    parser.add_argument('--save_results', action='store_true')
    # dataset
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', action='store_true')
    parser.add_argument('--use_tnet', action='store_true')
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    model = DeepGMR(args)
    if torch.cuda.is_available():
        model.cuda()

    test_data = TestData(args.data_file, args)
    test_loader = DataLoader(test_data, args.batch_size)

    model.load_state_dict(torch.load(args.checkpoint))
    evaluate(model, test_loader, args.save_results, args.results_dir)
