import argparse
import json
import logging
from datetime import datetime
import shutil
from itertools import chain
from os.path import join, exists
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from models import aae
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed
from utils.points import generate_points
from datasets import TxtDataset, Benchmark, collate_fn

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, 'vae', 'training')
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger('vae')

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == 'custom':
        dataset = TxtDataset(root_dir=config['data_dir'],
                             classes=config['classes'],
                             config=config)
    elif dataset_name == 'benchmark':
        dataset = Benchmark(root_dir=config['data_dir'],
                            classes=config['classes'],
                            config=config)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'], drop_last=True,
                                   pin_memory=True, collate_fn=collate_fn)

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder_pocket = aae.PocketEncoder(config).to(device)
    encoder_visible = aae.VisibleEncoder(config).to(device)

    hyper_network.apply(weights_init)
    encoder_pocket.apply(weights_init)
    encoder_visible.apply(weights_init)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        # from utils.metrics import earth_mover_distance
        # reconstruction_loss = earth_mover_distance
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    #
    # Optimizers
    #
    e_hn_optimizer = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer = e_hn_optimizer(chain(encoder_visible.parameters(), encoder_pocket.parameters(), hyper_network.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        hyper_network.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_G.pth')))
        encoder_pocket.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EP.pth')))
        encoder_visible.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EV.pth')))

        e_hn_optimizer.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGo.pth')))

        log.info("Loading losses...")
        losses_e = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E.npy')).tolist()
        losses_kld = np.load(join(metrics_path, f'{starting_epoch - 1:05}_KLD.npy')).tolist()
        losses_eg = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG.npy')).tolist()
    else:
        log.info("First epoch")
        losses_e = []
        losses_kld = []
        losses_eg = []

    if config['target_network_input']['normalization']['enable']:
        normalization_type = config['target_network_input']['normalization']['type']
        assert normalization_type == 'progressive', 'Invalid normalization type'

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)
        hyper_network.train()
        encoder_visible.train()
        encoder_pocket.train()

        total_loss_all = 0.0
        total_loss_r = 0.0
        total_loss_kld = 0.0
        for i, point_data in enumerate(points_dataloader, 1):
            # get only visible part of point cloud
            X = point_data['non-visible']
            X = X.to(device, dtype=torch.float)

            # get not visible part of point cloud
            X_visible = point_data['visible']
            X_visible = X_visible.to(device, dtype=torch.float)

            # get whole point cloud
            X_whole = point_data['cloud']
            X_whole = X_whole.to(device, dtype=torch.float)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)
                X_visible.transpose_(X_visible.dim() - 2, X_visible.dim() - 1)
                X_whole.transpose_(X_whole.dim() - 2, X_whole.dim() - 1)

            codes, mu, logvar = encoder_pocket(X)
            mu_visible = encoder_visible(X_visible)

            target_networks_weights = hyper_network(torch.cat((codes, mu_visible), 1))

            X_rec = torch.zeros(X_whole.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                if not config['target_network_input']['constant'] or target_network_input is None:
                    target_network_input = generate_points(config=config, epoch=epoch, size=(X_whole.shape[2], X_whole.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            loss_r = torch.mean(
                config['reconstruction_coef'] *
                reconstruction_loss(X_whole.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()

            loss_all = loss_r + loss_kld
            e_hn_optimizer.zero_grad()
            encoder_visible.zero_grad()
            encoder_pocket.zero_grad()
            hyper_network.zero_grad()

            loss_all.backward()
            e_hn_optimizer.step()

            total_loss_r += loss_r.item()
            total_loss_kld += loss_kld.item()
            total_loss_all += loss_all.item()

        log.info(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_ALL: {total_loss_all / i:.4f} '
            f'Loss_R: {total_loss_r / i:.4f} '
            f'Loss_E: {total_loss_kld / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )

        losses_e.append(total_loss_r)
        losses_kld.append(total_loss_kld)
        losses_eg.append(total_loss_all)

        #
        # Save intermediate results
        #
        X = X.cpu().numpy()
        X_whole = X_whole.cpu().numpy()
        X_rec = X_rec.detach().cpu().numpy()

        if epoch % config['save_frequency'] == 0:
            for k in range(min(5, X_rec.shape[0])):
                fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][1], X_rec[k][2], in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_reconstructed.png'))
                plt.close(fig)

                fig = plot_3d_point_cloud(X_whole[k][0], X_whole[k][1], X_whole[k][2], in_u_sphere=True, show=False,
                                          title=str(epoch))
                fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_real.png'))
                plt.close(fig)

                fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2], in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_visible.png'))
                plt.close(fig)

        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['save_frequency'] == 0:
            log.debug('Saving data...')

            torch.save(hyper_network.state_dict(), join(weights_path, f'{epoch:05}_G.pth'))
            torch.save(encoder_visible.state_dict(), join(weights_path, f'{epoch:05}_EV.pth'))
            torch.save(encoder_pocket.state_dict(), join(weights_path, f'{epoch:05}_EP.pth'))
            torch.save(e_hn_optimizer.state_dict(), join(weights_path, f'{epoch:05}_EGo.pth'))

            np.save(join(metrics_path, f'{epoch:05}_E'), np.array(losses_e))
            np.save(join(metrics_path, f'{epoch:05}_KLD'), np.array(losses_kld))
            np.save(join(metrics_path, f'{epoch:05}_EG'), np.array(losses_eg))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
