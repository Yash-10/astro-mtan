## Experiments using the original mTAN code on their data

```bash
Namespace(niters=500, lr=0.001, std=0.01, latent_dim=16, rec_hidden=64, gen_hidden=50, embed_time=128, k_iwae=5, save=1, enc='mtan_rnn', dec='mtan_rnn', fname=None, seed=0, n=8000, batch_size=32, quantization=0.016, classif=False, norm=True, kl=True, learn_emb=True, enc_num_heads=1, dec_num_heads=1, length=20, num_ref_points=64, dataset='physionet', enc_rnn=True, dec_rnn=True, sample_tp=0.9, only_periodic=None, dropout=0.0) 23792
Downloading https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt to data/physionet/PhysioNet/raw/Outcomes-a.txt
100% 79219/79219 [00:00<00:00, 247132.25it/s]
Downloading https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download to data/physionet/PhysioNet/raw/set-a.tar.gz?download
100% 6632372/6632372 [00:22<00:00, 295417.97it/s]
Processing set-a.tar.gz?download...
Downloading https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download to data/physionet/PhysioNet/raw/set-b.tar.gz?download
100% 6652690/6652690 [00:22<00:00, 302180.86it/s]
Processing set-b.tar.gz?download...
Done!
8000
torch.Size([6400, 203, 83]) torch.Size([1600, 190, 83])
parameters: 96594 70921
Iter: 1, avg elbo: 80.4483, avg reconst: 84.5458, avg kl: 0.4248, mse: 0.016993
Iter: 2, avg elbo: 28.7513, avg reconst: 31.6576, avg kl: 0.3066, mse: 0.006543
Iter: 3, avg elbo: 25.5648, avg reconst: 27.3990, avg kl: 0.4306, mse: 0.005789
Iter: 4, avg elbo: 22.6930, avg reconst: 24.1291, avg kl: 0.8148, mse: 0.005226
Iter: 5, avg elbo: 19.9423, avg reconst: 20.6898, avg kl: 1.4387, mse: 0.004650
Iter: 6, avg elbo: 17.6654, avg reconst: 18.1448, avg kl: 1.9683, mse: 0.004183
Iter: 7, avg elbo: 15.7095, avg reconst: 16.0011, avg kl: 2.9841, mse: 0.003776
Iter: 8, avg elbo: 14.1633, avg reconst: 14.4041, avg kl: 3.8953, mse: 0.003456
Iter: 9, avg elbo: 13.2250, avg reconst: 13.4721, avg kl: 4.8033, mse: 0.003276
Iter: 10, avg elbo: 12.4411, avg reconst: 12.6832, avg kl: 5.3704, mse: 0.003137
Test Mean Squared Error tensor(0.0030, device='cuda:0')
Iter: 11, avg elbo: 12.1235, avg reconst: 12.4866, avg kl: 5.5208, mse: 0.003050
Iter: 12, avg elbo: 11.6240, avg reconst: 12.0052, avg kl: 5.5749, mse: 0.002965
Iter: 13, avg elbo: 11.0661, avg reconst: 11.4515, avg kl: 5.4254, mse: 0.002834
Iter: 14, avg elbo: 10.4322, avg reconst: 10.7033, avg kl: 5.4734, mse: 0.002696
Iter: 15, avg elbo: 10.0510, avg reconst: 10.4880, avg kl: 5.3836, mse: 0.002605
Iter: 16, avg elbo: 9.6720, avg reconst: 10.1387, avg kl: 5.3402, mse: 0.002500
Iter: 17, avg elbo: 8.9524, avg reconst: 9.2457, avg kl: 5.4959, mse: 0.002380
Iter: 18, avg elbo: 8.5223, avg reconst: 8.8246, avg kl: 5.6128, mse: 0.002282
Iter: 19, avg elbo: 7.9927, avg reconst: 8.3589, avg kl: 5.4461, mse: 0.002160
Iter: 20, avg elbo: 7.3915, avg reconst: 7.7815, avg kl: 5.5068, mse: 0.002048
Test Mean Squared Error tensor(0.0020, device='cuda:0')
Iter: 21, avg elbo: 7.4114, avg reconst: 7.6975, avg kl: 5.3325, mse: 0.002018
Traceback (most recent call last):
  File "/content/mTAN/src/tan_interpolation.py", line 151, in <module>
    train_loss += loss.item() * batch_len
KeyboardInterrupt
```
We see that the ELBO, MSE, reconstruction loss, all decrease monotonically as training proceeds. Does this happen with our light curves?
