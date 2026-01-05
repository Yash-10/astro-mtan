#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -t 48:00:00

#python3 tan_unsupervised.py --niters 500 --lr 0.001 --rec-hidden 64 --latent-dim 1 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-04-17_832975_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-04-17_832975_copy --dim 2 --embed-time 128

#python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 1 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-04-26_572037_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-04-26_572037_copy --dim 2 --embed-time 128

# BELOW IS THE DEFAULT TRAINING CODE
#python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-05-27_433174_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-05-27_433174_copy --dim 2 --embed-time 128

#export WANDB_API_KEY=2694f9864be3a11448f361be00fce534845723b9

python3 tan_unsupervised.py --niters 300 --lr 0.0001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --dataset ftransfer_ztf_2025-05-31_430518 --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2025-05-31_430518 --dim 2 --embed-time 128 --ref-resolution-days 2 --train_val_test_min_max_times_filename train_val_test_min_max_times.npy

#python3 tan_unsupervised.py --niters 200 --lr 0.001 --rec-hidden 64 --latent-dim 2 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --learn-emb --kl --seed 42 --num-ref-points 160 --dataset ftransfer_ztf_2024-05-27_433174_copy --sample-tp 1.0 --save 1 --k-iwae 5 --std 0.01 --norm --topic ftransfer_ztf_2024-05-27_433174_copy --dim 2 --embed-time 128

