import os
import nlp2

from dtokenizer.interface import BaseTokenizer
from .modeling_hubert import _Speech2Code, _Code2Speech
import torchaudio


class HubertTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.sampling_rate = 16000
        if config in CONFIG:
            self.sc, self.cs = CONFIG[config]()
        else:
            raise ValueError(f"config {config} not found in {CONFIG.keys()}")

    def encode(self, speech, sampling_rate):
        # if sampling_rate is not 16000
        if sampling_rate != self.sampling_rate:
            speech = torchaudio.functional.resample(speech, sampling_rate, self.sampling_rate)
        return self.sc(input_values=speech), None

    def encode_file(self, input_file):
        return self.sc(filepaths=[input_file])['code'], None

    def decode(self, code):
        if not self.cs:
            raise ValueError("No hubert vocoder is available")
        else:
            return self.cs(code)


def hubert_layer6_code50(sampling_rate=16000,
                         chunk_sec=10,
                         worker=8,
                         return_diff=False,
                         batch=None):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin', './')
    if not os.path.exists('./hubert_base_ls960_L6_km50.bin'):
        os.rename('./km.bin', './hubert_base_ls960_L6_km50.bin')
    sc = _Speech2Code("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km50.bin', 6,
                      sampling_rate=sampling_rate,
                      chunk_sec=chunk_sec,
                      worker=worker,
                      return_diff=return_diff,
                      batch=batch)
    return sc, None


def hubert_layer6_code100(sampling_rate=16000,
                          chunk_sec=10,
                          worker=8,
                          return_diff=False,
                          batch=None):
    # https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_to_speech/docs/direct_s2st_discrete_units.md
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km100.bin')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000',
        './', 'hifigan_hubert_layer6_code100_g_00500000')
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json',
        './', 'hifigan_hubert_layer6_code100_config.json')
    sc = _Speech2Code("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km100.bin', 6,
                      sampling_rate=sampling_rate,
                      chunk_sec=chunk_sec,
                      worker=worker,
                      return_diff=return_diff,
                      batch=batch)
    cs = _Code2Speech(tts_checkpoint='./hifigan_hubert_layer6_code100_g_00500000',
                      model_cfg=nlp2.read_json('./hifigan_hubert_layer6_code100_config.json'))
    return sc, cs


def hubert_layer6_code200(sampling_rate=16000,
                          chunk_sec=10,
                          worker=8,
                          return_diff=False,
                          batch=None):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin', './')
    os.rename('./km.bin', './hubert_base_ls960_L6_km200.bin')
    sc = _Speech2Code("facebook/hubert-base-ls960", './hubert_base_ls960_L6_km200.bin', 6,
                      sampling_rate=sampling_rate,
                      chunk_sec=chunk_sec,
                      worker=worker,
                      return_diff=return_diff,
                      batch=batch)
    return sc, None


def hubert_layer9_code500(sampling_rate=16000,
                          chunk_sec=10,
                          worker=8,
                          return_diff=False,
                          batch=None):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin', './')
    sc = _Speech2Code("facebook/hubert-base-ls960", './hubert_base_ls960_L9_km500.bin', 9,
                      sampling_rate=sampling_rate,
                      chunk_sec=chunk_sec,
                      worker=worker,
                      return_diff=return_diff,
                      batch=batch)
    return sc, None


def zh_hubert_layer20_code2000(sampling_rate=16000,
                               chunk_sec=10,
                               worker=8,
                               return_diff=False,
                               batch=None):
    nlp2.download_file(
        'https://huggingface.co/anthony-wss/extract-ssl-bpe/resolve/main/km_2000.mdl', './')
    sc = _Speech2Code("TencentGameMate/chinese-hubert-large", './km_2000.mdl', 20,
                      sampling_rate=sampling_rate,
                      chunk_sec=chunk_sec,
                      worker=worker,
                      return_diff=return_diff,
                      batch=batch)
    return sc, None


CONFIG = {
    "hubert_layer6_code50": hubert_layer6_code50,
    "hubert_layer6_code100": hubert_layer6_code100,
    "hubert_layer6_code200": hubert_layer6_code200,
    "hubert_layer9_code500": hubert_layer9_code500,
    "zh_hubert_layer20_code2000": zh_hubert_layer20_code2000,
}
