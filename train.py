'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter
from time import time
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import TrainData
from model import DeepGMR


def train_one_epoch(epoch, model, loader, writer, log_freq, plot_freq):
    model.train()

    log_fmt = 'Epoch {:03d} Step {:03d}/{:03d} Train: ' + \
              'batch time {:.2f}, data time {:.2f}, loss {:.4f}, ' + \
              'rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}'
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    rmses = []
    total_steps = len(loader)

    start = time()
    for step, (pts1, pts2, T_gt) in enumerate(loader):
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            T_gt = T_gt.cuda()
        data_time.append(time() - start)

        optimizer.zero_grad()
        loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
        loss.backward()
        optimizer.step()
        batch_time.append(time() - start)

        losses.append(loss.item())
        r_errs.append(r_err.mean().item())
        t_errs.append(t_err.mean().item())
        rmses.append(rmse.mean().item())

        global_step = epoch * len(loader) + step + 1

        if global_step % log_freq == 0:
            log_str = log_fmt.format(
                epoch+1, step+1, total_steps,
                np.mean(batch_time), np.mean(data_time), np.mean(losses),
                np.mean(r_errs), np.mean(t_errs), np.mean(rmses)
            )
            print(log_str)
            writer.add_scalar('train/loss', np.mean(losses), global_step)
            writer.add_scalar('train/rotation_error', np.mean(r_errs), global_step)
            writer.add_scalar('train/translation_error', np.mean(t_errs), global_step)
            writer.add_scalar('train/RMSE', np.mean(rmses), global_step)
            batch_time.clear()
            data_time.clear()
            losses.clear()
            r_errs.clear()
            t_errs.clear()
            rmses.clear()

        if global_step % plot_freq == 0:
            fig = model.visualize(0)
            writer.add_figure('train', fig, global_step)

        start = time()


def eval_one_epoch(epoch, model, loader, writer, global_step, plot_freq):
    model.eval()

    log_fmt = 'Epoch {:03d} Valid: batch time {:.2f}, data time {:.2f}, ' + \
              'loss {:.4f}, rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}'
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    rmses = []

    start = time()
    for step, (pts1, pts2, T_gt) in enumerate(tqdm(loader, leave=False)):
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            T_gt = T_gt.cuda()
        data_time.append(time() - start)

        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
            batch_time.append(time() - start)

        losses.append(loss.item())
        r_errs.append(r_err.mean().item())
        t_errs.append(t_err.mean().item())
        rmses.append(rmse.mean().item())

        if writer is not None:
            if (step+1) % plot_freq == 0:
                fig = model.visualize(0)
                writer.add_figure('valid/{:02d}'.format(step+1), fig, global_step)

        start = time()

    log_str = log_fmt.format(
        epoch+1, np.mean(batch_time), np.mean(data_time),
        np.mean(losses), np.mean(r_errs), np.mean(t_errs), np.mean(rmses)
    )
    print(log_str)
    writer.add_scalar('valid/loss', np.mean(losses), global_step)
    writer.add_scalar('valid/rotation_error', np.mean(r_errs), global_step)
    writer.add_scalar('valid/translation_error', np.mean(t_errs), global_step)
    writer.add_scalar('valid/RMSE', np.mean(rmses), global_step)

    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file')
    parser.add_argument('--log_dir')
    # dataset
    parser.add_argument('--max_angle', type=float, default=180)
    parser.add_argument('--max_trans', type=float, default=0.5)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--clean', action='store_true')
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', action='store_true')
    parser.add_argument('--use_tnet', action='store_true')
    parser.add_argument('--k', type=int, default=20)
    # train setting
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=30)
    parser.add_argument('--plot_freq', type=int, default=250)
    parser.add_argument('--save_freq', type=int, default=10)
    # eval setting
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_plot_freq', type=int, default=10)
    args = parser.parse_args()

    model = DeepGMR(args)
    if torch.cuda.is_available():
        model.cuda()

    data = TrainData(args.data_file, args)
    ids = np.random.permutation(len(data))
    n_val = int(args.val_fraction * len(data))
    train_data = Subset(data, ids[n_val:])
    valid_data = Subset(data, ids[:n_val])

    train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, args.eval_batch_size, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.n_epochs):
        train_one_epoch(epoch, model, train_loader, writer, args.log_freq, args.plot_freq)

        global_step = (epoch+1) * len(train_loader)
        test_loss = eval_one_epoch(epoch, model, valid_loader, writer, global_step, args.eval_plot_freq)

        scheduler.step(test_loss)
        for param_group in optimizer.param_groups:
            lr = float(param_group['lr'])
            break
        writer.add_scalar('train/learning_rate', lr, global_step)

        if (epoch+1) % args.save_freq == 0:
            filename = '{}/checkpoint_epoch-{:d}.pth'.format(args.log_dir, epoch+1)
            torch.save(model.state_dict(), filename)
