from dtokenizer.audio.model.hubert_model import HubertTokenizer

ht = HubertTokenizer('hubert_layer6_code100')
print(ht.encode_file('./sample2_22k.wav'))
code, decodec_stuff = ht.encode_file('./sample2_22k.wav')
print(code)
wav_values = ht.decode(code)
# write wav file
import soundfile as sf

sf.write('output.wav', wav_values, 16000)
