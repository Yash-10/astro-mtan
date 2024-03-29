from models import enc_mtan_rnn
from mtan_utils import subsample_timepoints
from sklearn import preprocessing
import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: Add option to pass these arguments as argument parsers.
num_ref_points = 64
latent_dim = 16
learn_emb = True
rec_hidden = 64
enc_num_heads = 1
sample_tp = 0.9
num_sample = 10
dim = 2

#train_loader = torch.load('train_dataloader.pth')
test_loader = torch.load('test_dataloader.pth')
final_objIds = np.load('final_objIds.npy')
final_objIds_encoded = np.load('final_objIds_encoded.npy')

le = preprocessing.LabelEncoder()
final_objIds_encoded_again  = le.fit_transform(final_objIds)  # use le.inverse_transform to get the string from the encoded value.
assert np.all(final_objIds_encoded == final_objIds_encoded_again)  # since the label encoding must be deterministic. So the values here and found during main_preprocessing must match.


rec = enc_mtan_rnn(
    dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden,
    embed_time=128, learn_emb=learn_emb, num_heads=enc_num_heads, device=device
).to(device)

outputs = []
with torch.no_grad():
    for batch in test_loader:
        test_batch = batch[0]  # batch[1] contains the objectIds in numerical form.
        test_batch = test_batch.to(device)
        observed_data, observed_mask, observed_tp = (
            test_batch[:, :, :dim],
            test_batch[:, :, dim: 2 * dim],
            test_batch[:, :, -1],
        )
        if sample_tp and sample_tp < 1:
            subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                observed_data.clone(), observed_tp.clone(), observed_mask.clone(), sample_tp)
        else:
            subsampled_data, subsampled_tp, subsampled_mask = \
                observed_data, observed_tp, observed_mask
        out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
        qz0_mean, qz0_logvar = (
            out[:, :, : latent_dim],
            out[:, :, latent_dim:],
        )
        epsilon = torch.randn(
            num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
        ).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        outputs.append((batch[1], z0))

outputs_condensed = np.array([o[1].cpu().detach().numpy() for o in outputs])  # this will be an array of shape (num_sample, 64, 16). The num_sample dimension can be averaged or compressed.
objIds = le.inverse_transform([o[0][0].cpu().detach().numpy() for o in outputs])
print(outputs_condensed.shape, len(objIds))

np.save('outputs_condensed.npy', outputs_condensed)
np.save('objIds.npy', objIds)

