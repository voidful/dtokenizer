import torch
from dtokenizer.audio.model.encodec_model import EncodecTokenizer
import torchaudio

ht = EncodecTokenizer('encodec_24k_6bps')
code, stuff_for_decode = ht.encode_file('./sample2_22k.wav')
wav_values = ht.decode(stuff_for_decode)
torchaudio.save('output.wav', torch.from_numpy(wav_values), 22050)
