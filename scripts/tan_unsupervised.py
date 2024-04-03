# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.optim as optim

from random import SystemRandom
import models

import time

import mtan_utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--topic', type=str, help='Name of the topic of the data transfer that contains the alerts.')
parser.add_argument('--dim', type=int, help='dim value')
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    train_loader = torch.load('train_dataloader.pth')
    val_loader = torch.load('val_dataloader.pth')
    test_loader = torch.load('test_dataloader.pth')
    dim = args.dim

    # model
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, 128, learn_emb=args.learn_emb, device=device).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads, device=device).to(device)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 128, learn_emb=args.learn_emb, device=device).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads, device=device).to(device)


    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    print('parameters:', mtan_utils.count_parameters(rec), mtan_utils.count_parameters(dec))
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 1), device=device)
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 3), device=device)
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 10), device=device)
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 20), device=device)
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 30), device=device)
        print('Test MSE', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 50), device=device)

    best_val_mse = float('inf')
    total_time = 0.
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse = 0, 0, 0
        val_avg_reconst, val_avg_kl, val_mse, val_loss = 0, 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        start_time = time.time()

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = mtan_utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            # epsilon = torch.randn(qz0_mean.size()).to(device)
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            # compute loss
            logpx, analytic_kl = mtan_utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += mtan_utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len
        
        total_time += time.time() - start_time
        # Run validation
        # TODO: this means we validate using the MSE. Not sure if we should use the reconstruction loss, the KL loss or some combination of it instead. The mTAN paper used MSE as the test metric.
        val_mse = mtan_utils.evaluate(dim, rec, dec, val_loader, args, 1, device=device)
        if val_mse <= best_val_mse:
            best_val_mse = min(best_val_mse, val_mse)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            optimizer_state_dict = optimizer.state_dict()

            print(f'Saving the model at epoch {itr}')
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec_state_dict,
                'dec_state_dict': dec_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss': -loss,
            }, args.dataset + '_' + args.enc + '_' + args.dec + '_' + '.h5')

        # Validation end.
        scheduler.step(val_mse)

        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}, val_mse: {:.6f}'
                .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n, val_mse))
        if itr % 5 == 0:
            print('Test Mean Squared Error', mtan_utils.evaluate(dim, rec, dec, test_loader, args, 1, device=device))

        print(f'Time elapsed {total_time/60:.2f} min')

        #if itr % 10 == 0 and args.save:
        #    torch.save({
        #        'args': args,
        #        'epoch': itr,
        #        'rec_state_dict': rec_state_dict,
        #        'dec_state_dict': dec_state_dict,
        #        'optimizer_state_dict': optimizer_state_dict,
        #        'loss': -loss,
        #    }, args.dataset + '_' + args.enc + '_' + args.dec + '_' +
        #        str(itr) + '.h5')

