from models import enc_mtan_rnn, dec_mtan_rnn
from mtan_utils import subsample_timepoints, mean_squared_error
from sklearn import preprocessing
import torch
import numpy as np

from torch.utils.data import DataLoader
from prepare_data import MyDataSet
from torch.nn.utils.rnn import pad_sequence

import time

import utils

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
# TODO: Add option to pass these arguments as argument parsers. These values must match from training. So instead save a training parameter file and simply load it here.
ref_resolution_days = 2
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
# NOTE: Change the `SETTING` based on which dataset to evaluate the model on.
SETTING = 'test'   # 'test', 'train', or 'val', or 'OOD_test'. The last option is for totally new custom data. See `OOD_evaluate_unsupervised_prepare_data.py`.
DATA_COMBINED_PATH = f'{SETTING}_data_combined.pth'
DATA_IDS_PATH = f'{SETTING}_objIds.npy'
model_file_path = 'ftransfer_ztf_2025-05-31_430518_mtan_rnn_mtan_rnn_Jul27_298.h5'
batch_size = 1

if __name__ == '__main__':
    # Set seed during testing as well since this script samples random values for the variable, epsilon.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        print(torch.cuda.get_device_name(0))

    data_combined = torch.load(DATA_COMBINED_PATH, map_location=torch.device(device))
    data_Ids = np.load(DATA_IDS_PATH)
    dataset = MyDataSet(data_combined, data_Ids)
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    train_val_test_min_max_times_filename = 'train_val_test_min_max_times.npy'
    train_val_test_min_max_times = np.load(train_val_test_min_max_times_filename)
    train_max_time = train_val_test_min_max_times[1]
    delta_t = (ref_resolution_days * 24) / train_max_time

    #test_loader = torch.load('test_dataloader.pth') # NOTE: This script is only tested for dataloaders with batch size=1; for greater batch sizes, some bugs may be introduced.
    #total_objIds = np.load('total_objIds.npy')
    #total_objIds_encoded = np.load('total_objIds_encoded.npy')

    #le = preprocessing.LabelEncoder()
    #total_objIds_encoded_again  = le.fit_transform(total_objIds)  # use le.inverse_transform to get the string from the encoded value.
    #assert np.all(total_objIds_encoded == total_objIds_encoded_again)  # since the label encoding must be deterministic. So the values here and found during main_preprocessing must match.


    rec = enc_mtan_rnn(
        dim, latent_dim, rec_hidden,  # torch.linspace(0, 1., num_ref_points)
        embed_time=embed_time, learn_emb=learn_emb, num_heads=enc_num_heads, device=device
    ).to(device)

    dec = dec_mtan_rnn(
        dim, latent_dim, gen_hidden,
        embed_time=embed_time, learn_emb=learn_emb, num_heads=dec_num_heads, device=device).to(device)

    model_file = torch.load(model_file_path, map_location=torch.device(device))
    rec.load_state_dict(model_file['rec_state_dict'])
    rec.eval()
    dec.load_state_dict(model_file['dec_state_dict'])

    # Efficient decoder
    #dec = torch.compile(dec, mode="reduce-overhead")

    dec.eval()

    ########### Check model size ###########
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f'Encoder size [in MB]: {get_model_size(rec)}')
    print(f'Decoder size [in MB]: {get_model_size(dec)}')

    ########################################

    torch.set_grad_enabled(False)

    #start = time.time()
    times, num_obs_arr, ids = [], [], []
    for batch in test_loader:
        test_batch = batch[0]
        test_batch = test_batch.to(device)
        observed_data, observed_mask, observed_tp = (
            test_batch[:, :, :dim],
            test_batch[:, :, dim: 2 * dim],
            test_batch[:, :, -1],
        )
        subsampled_data, subsampled_tp, subsampled_mask = observed_data, observed_tp, observed_mask

        #if sample_tp == 1.:
        #    assert torch.all(observed_tp == subsampled_tp)

        _times = []
        # NOTE: This approach of iterating multiple times over a single batch can have first iteration
        # to be slower than the rest due to cache. Taking the median should tackle this problem.
        # An alternative is to run iterations over the entire test data multiple times rather than on
        # a single batch multiple times -- cutting and pasting `for _ in range(5):` before `for batch in test_loader:`.
        # I checked that this approach only added <=1 sec overhead, so likely the median calculation is solving that issue.
        for _ in range(5):
            start = time.time()
            query = utils.generate_query_matrix(subsampled_tp, delta_t, padding_value=-999)
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp, query)
            end = time.time()
            _times.append(end - start)
        median_exec_time = np.median(_times)

        num_obs = observed_mask.sum(-1).sum(-1)
        times.append(median_exec_time)
        num_obs_arr.append(num_obs.cpu().numpy())
        ids.append(batch[1])
        #times.append([batch[1], median_exec_time, num_obs.numpy()])

        qz0_mean, qz0_logvar = (
            out[:, :, : latent_dim],
            out[:, :, latent_dim:],
        )
        epsilon = torch.randn(
            num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
        ).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
        batch, seqlen = observed_tp.size()
        time_steps = (
            observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
        )
        pred_x = dec(z0, time_steps, query)

    #end = time.time()
    #print(end-start)
    # This is the time reported for the batch_size = 32 case in the paper
    print('Total time required = ', np.sum(times))
    if batch_size == 1:
        ids = [i[0] for i in ids]
        num_obs_arr = [float(i) for i in num_obs_arr]
    print(np.array(ids).shape, np.array(times).shape, np.array(num_obs_arr).shape)
    # NOTE: The below will give a shape mismatch error for any batch size other than 1
    # But that's understandable, so if using batch size > 1, see the time printed above.
    np.save(f'mtan_exec_time_vs_nobs_{batch_size}.npy', np.vstack((ids, times, num_obs_arr)))
