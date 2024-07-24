import math
import os
import nlp2
import torch
from semanticodec.main import AUDIOMAE_PATCH_DURATION, SAMPLE_RATE, SEGMENT_DURATION, MEL_TARGET_LENGTH
from semanticodec.utils import extract_kaldi_fbank_feature

from dtokenizer.interface import BaseTokenizer

try:
    from semanticodec import SemantiCodec
except ImportError:
    raise ImportError(
        "Please install semanticodec: pip install git+https://github.com/haoheliu/SemantiCodec-inference.git")

CONFIG = {
    "semanticodec_25_035": (25, 32768),
    "semanticodec_25_034": (25, 16384),
    "semanticodec_25_033": (25, 8192),
    "semanticodec_25_031": (25, 4096),
}


class SemanticodecTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.sampling_rate = 16000
        if config in CONFIG:
            token_rate, semantic_vocab_size = CONFIG[config]
            self.model = SemantiCodec(token_rate, semantic_vocab_size)
        else:
            raise ValueError(f"config {config} not found in {CONFIG.keys()}")

    def encode(self, speech, sampling_rate):
        if speech.shape[0] > 1:
            speech = speech[0:1]
        original_duration = speech.shape[1] / sampling_rate
        original_duration = original_duration + (
                AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION
        )
        target_token_len = (
                8 * original_duration / AUDIOMAE_PATCH_DURATION / self.stack_factor_K
        )
        segment_sample_length = int(sampling_rate * SEGMENT_DURATION)
        if speech.shape[1] % segment_sample_length < segment_sample_length:
            speech = torch.cat(
                [
                    speech,
                    torch.zeros(
                        1,
                        int(
                            segment_sample_length
                            - speech.shape[1] % segment_sample_length
                        ),
                    ),
                ],
                dim=1,
            )

        mel_target_length = MEL_TARGET_LENGTH * int(
            speech.shape[1] / segment_sample_length
        )
        mel = extract_kaldi_fbank_feature(
            speech, sampling_rate, target_length=mel_target_length
        )["ta_kaldi_fbank"].unsqueeze(0)
        tokens = self.model.encoder(mel.to(self.device))
        tokens = tokens[:, : math.ceil(target_token_len), :]
        return tokens, None

    def encode_file(self, input_file):
        return self.model.encode(input_file), None

    def decode(self, code):
        return self.model.decode(code)
