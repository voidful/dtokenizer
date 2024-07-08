import os
import nlp2

from dtokenizer.audio.model.hubert.modeling_hubert import _Code2Speech, _Speech2Code


def hubert_layer6_code50(sampling_rate=16000,
                         chunk_sec=10,
                         worker=8,
                         return_diff=False,
                         batch=None):
    nlp2.download_file(
        'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin', './')
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
                      model_cfg=nlp2.read_json('./hifigan_hubert_layer6_code100_config.json'), vocoder='hifigan')
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
