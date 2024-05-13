from models import enc_mtan_rnn, dec_mtan_rnn
from mtan_utils import subsample_timepoints, mean_squared_error
from sklearn import preprocessing
import torch
import numpy as np

from torch.utils.data import DataLoader
from prepare_data import MyDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: Add option to pass these arguments as argument parsers. These values must match from training. So instead save a training parameter file and simply load it here.
num_ref_points = 160
latent_dim = 2
learn_emb = True
rec_hidden = 64
gen_hidden = 50
enc_num_heads = 1
dec_num_heads = 1
sample_tp = 1.0
num_sample = 1
embed_time = 128
dim = 2
seed = 42
store_decoded_lcs = True
# NOTE: Change the below two lines based on which dataset to evaluate the model on.
DATA_COMBINED_PATH = 'train_data_combined.pth'
DATA_IDS_PATH = 'train_objIds.npy'
model_file_path = 'ftransfer_ztf_2024-04-26_572037_copy_mtan_rnn_mtan_rnn_.h5'


# Set seed during testing as well since this script samples random values for the variable, epsilon.
torch.manual_seed(seed)
np.random.seed(seed)

if device == 'cuda':
    torch.cuda.manual_seed(seed)

data_combined = torch.load(DATA_COMBINED_PATH)
data_Ids = np.load(DATA_IDS_PATH)
dataset = MyDataSet(data_combined, data_Ids)
test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

#test_loader = torch.load('test_dataloader.pth') # NOTE: This script is only tested for dataloaders with batch size=1; for greater batch sizes, some bugs may be introduced. TODO: Fix this so can I also use train dataloader when needed to get results on the train set, for example.
#total_objIds = np.load('total_objIds.npy')
#total_objIds_encoded = np.load('total_objIds_encoded.npy')

#le = preprocessing.LabelEncoder()
#total_objIds_encoded_again  = le.fit_transform(total_objIds)  # use le.inverse_transform to get the string from the encoded value.
#assert np.all(total_objIds_encoded == total_objIds_encoded_again)  # since the label encoding must be deterministic. So the values here and found during main_preprocessing must match.


rec = enc_mtan_rnn(
    dim, torch.linspace(0, 1., num_ref_points), latent_dim, rec_hidden,
    embed_time=embed_time, learn_emb=learn_emb, num_heads=enc_num_heads, device=device
).to(device)

dec = dec_mtan_rnn(
    dim, torch.linspace(0, 1., num_ref_points), latent_dim, gen_hidden,
    embed_time=embed_time, learn_emb=learn_emb, num_heads=dec_num_heads, device=device).to(device)

model_file = torch.load(model_file_path)
rec.load_state_dict(model_file['rec_state_dict'])
rec.eval()
dec.load_state_dict(model_file['dec_state_dict'])
dec.eval()

outputs = []
objIds = []
test_n, mse = 0, 0.0
if store_decoded_lcs:
    decoded_lcs = []
with torch.no_grad():
    for batch in test_loader:
        test_batch = batch[0]
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

        if sample_tp == 1.:
            assert torch.all(observed_tp == subsampled_tp)
        assert subsampled_tp.max() <= 1
        ##query = torch.linspace(0, subsampled_tp.max(), num_ref_points)

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
        outputs.append(z0)
        objIds.append(batch[1])

        if store_decoded_lcs:
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch

            print(pred_x.shape, time_steps.shape, time_steps.unsqueeze(2).shape)
            decoded_lcs.append(
                np.vstack(
                    (pred_x.cpu().detach().numpy(), observed_data.cpu().detach().numpy(), observed_mask.cpu().detach().numpy())  # TODO: Also save time_steps: time_steps.unsqueeze(2).cpu().detach().numpy()
                )
            )

outputs_condensed = np.array([o.cpu().detach().numpy() for o in outputs])  # this will be an array of shape (num_test_examples, num_sample, num_ref_points, latent_dim). The num_sample dimension can be averaged or compressed somehow if it contains more than one entry.
print(outputs_condensed.shape)

from itertools import chain
objIds = list(chain.from_iterable(objIds))

print(f'MSE = {mse/test_n}')

np.save('evaluate_outputs_condensed.npy', outputs_condensed)
np.save('evaluate_objIds_dataloader.npy', objIds)

assert np.all(objIds == data_Ids)

if store_decoded_lcs:
    np.savez('evaluate_decoded_lcs.npz', *decoded_lcs)
