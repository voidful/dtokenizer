import gc
from collections import defaultdict
from functools import partial
from itertools import groupby

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.exceptions import InconsistentVersionWarning
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.contrib.concurrent import thread_map
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from transformers import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from dtokenizer.audio.utility import collate_fn_pad, chunks
from dtokenizer.audio.vocoder.hifigan import load_hifigan


class SpeechDataset(Dataset):
    def __init__(self, paths, input_values, processor, sampling_rate=16000):
        self.paths = paths
        self.input_values = input_values
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        if index < len(self.paths):
            speech, sr = torchaudio.load(self.paths[index])
        else:
            speech = self.input_values[index - len(self.paths)][None, :]
            sr = self.sampling_rate

        speech = speech.mean(0)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            speech = resampler.forward(speech.squeeze(0))
        else:
            speech = speech.squeeze(0)
        input_values = self.processor(speech, return_tensors="pt",
                                      sampling_rate=self.sampling_rate).input_values
        return input_values.squeeze(0)

    def __len__(self):
        return len(self.paths) + len(self.input_values)


class _Code2Speech(object):
    def __init__(self, tts_checkpoint, model_cfg=None, end_tok=None, code_begin_pad=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_rate = 16000
        self.hifigan = load_hifigan(model_path=tts_checkpoint, model_cfg=model_cfg)
        self.end_tok = end_tok
        self.code_begin_pad = code_begin_pad

    def __call__(self, code, strength=0.1, dur_prediction=True):
        with torch.no_grad():
            code = [i + self.code_begin_pad for i in code]
            if self.end_tok is not None and code[-1] != self.end_tok:
                code.append(self.end_tok)
            tts_input = torch.tensor(code, dtype=torch.long)
            x = {
                "code": tts_input.view(1, -1)
            }
            audio_seq = self.hifigan(x, dur_prediction=dur_prediction)

            return audio_seq


def dataloader_collate(batch):
    return torch.cat(batch, dim=0), [b.shape[0] for b in batch]


class _Speech2Code(object):
    def __init__(self, hubert_model, km_path, km_layer,
                 sampling_rate=16000,
                 chunk_sec=10,
                 worker=0,
                 return_diff=False,
                 batch=None):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
        self.model = HubertModel.from_pretrained(hubert_model)
        self.model.eval()
        self.sampling_rate = sampling_rate
        self.chunk_length = sampling_rate * chunk_sec
        self.km_model = joblib.load(km_path)
        self.km_layer = km_layer
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.worker = worker
        self.return_diff = return_diff
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()
            self.model = self.model.cuda()
        self.max_batch = batch if batch else self.get_max_batch()

    def get_max_batch(self):
        batch = 1
        with torch.no_grad():
            try:
                while True:
                    self.model(torch.rand([batch, self.chunk_length]).cuda())
                    batch += 2
                    gc.collect()
                    torch.cuda.empty_cache()
            except:
                pass
        batch = max(int(batch * 0.95), 1)
        return batch

    def _process_feature(self, k, top_k=100, feat_norm=False, beamsearch=False, beamsize=5):
        feature = torch.cat(k, dim=0) if isinstance(k, list) else k
        if feat_norm:
            m = nn.BatchNorm1d(feature.shape[-1], affine=False).to(self.device)
            feature = m(feature)
        dist = torch.sqrt(
            feature.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feature, self.C)
            + self.Cnorm
        )
        min_dist = torch.topk(dist.detach().cpu(), top_k, dim=-1, largest=False)
        pred_ind_array = min_dist.indices.cpu().numpy()
        pred_values_array = min_dist.values.cpu().numpy()
        code_output = min_dist.indices.T.cpu().numpy()[0]
        return_dict = {
            'code': list(code_output),
            'merged_code': [k for k, _ in groupby(code_output)]
        }
        if self.return_diff:
            return_dict.update({
                'distance': list(dist.detach().cpu().numpy()),
                'center_diff': list((feature.cpu() - torch.index_select(torch.tensor(self.C_np.transpose()).cpu(), 0,
                                                                        min_dist.indices[:, 0].cpu())).numpy()),
            })
        if beamsearch:
            sequences = [[[], 1.0]]
            self.var_list = []
            for i_row, v_row in zip(pred_ind_array, pred_values_array):
                all_candidates = list()
                for seq in sequences:
                    tokens, score = seq
                    self.var_list.append(np.var(v_row))
                    for k, v in zip(i_row, v_row):
                        norm_len_rate = (len(code_output) / len([k for k, _ in groupby(tokens + [k])]))
                        norm_dist_rate = np.var(v_row) / v
                        candidate = [tokens + [k], score + norm_len_rate * norm_dist_rate]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                sequences = ordered[:beamsize]
            code_output = sequences[0][0]
            return_dict['beam_code'] = code_output
            return_dict['beam_merged_code'] = [k for k, _ in groupby(code_output)]
        return return_dict

    def __call__(self, filepaths=None, input_values=None, feat_norm=False, beamsearch=False, top_k=5, beamsize=5):
        with torch.no_grad():
            if filepaths is None:
                filepaths = []

            if input_values is None:
                input_values = []

            if len(filepaths) == 0 and len(input_values) == 0:
                raise ValueError("Both 'filepaths' and 'input_values' are empty. Provide at least one of them.")

            is_single_input = (len(filepaths) == 1 and len(input_values) == 0) or (
                    len(filepaths) == 0 and len(input_values) == 1)

            if isinstance(input_values, torch.Tensor):
                input_values = [input_values]

            if len(filepaths) > 0:
                dataset = SpeechDataset(filepaths, input_values, self.processor, self.sampling_rate)
                dataloader = DataLoader(dataset=dataset, batch_size=self.max_batch,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=dataloader_collate)
                process_data_batches = iter(dataloader)
            else:
                def process_input_values():
                    for iv in input_values:
                        yield iv, [iv.shape[0]]

                process_data_batches = process_input_values()

            return_list = []
            for data_batch, size in process_data_batches:
                batch_data = []
                batch_map_audio = []
                for b_id, audio in enumerate(torch.split(data_batch, size)):
                    split_chunks = list(torch.split(audio, self.chunk_length, dim=-1))

                    # Check if the last chunk is smaller than the sampling_rate
                    if split_chunks[-1].shape[-1] < self.sampling_rate:
                        concat_index = -2 if len(split_chunks) >= 2 else 0
                        split_chunks[concat_index] = torch.cat(split_chunks[-2:], dim=-1)
                        split_chunks = split_chunks[:concat_index + 1]

                    # Iterate through chunks and append them to batch_data and batch_map_audio
                    for chunk in split_chunks:
                        batch_data.append(chunk)
                        batch_map_audio.append(b_id)

                code_result = defaultdict(list)
                for bd, bm in zip(chunks(batch_data, self.max_batch), chunks(batch_map_audio, self.max_batch)):
                    batch, lengths, masks = collate_fn_pad(bd, self.device)
                    masks_ratio = lengths / torch.max(lengths)
                    hidden = self.model(batch,
                                        output_hidden_states=True).hidden_states[self.km_layer].detach()
                    mask_len = (hidden.shape[1] * masks_ratio).int()
                    for a, h, ml in zip(bm, hidden, mask_len):
                        code_result[a].append(h[:ml, :])

                for k, v in code_result.items():
                    result = {}
                    for d in thread_map(
                            partial(self._process_feature,
                                    top_k=top_k,
                                    beamsearch=beamsearch,
                                    beamsize=beamsize,
                                    feat_norm=feat_norm), v,
                            leave=False, disable=True):
                        for k2, v2 in d.items():
                            if k2 in result:
                                result[k2].extend(v2)
                            else:
                                result[k2] = v2
                    return_list.append(result)

        if is_single_input:
            return return_list[0]
        else:
            return return_list
