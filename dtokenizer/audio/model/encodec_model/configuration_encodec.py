import soundfile
import torch

from dtokenizer.interface import BaseTokenizer
from SoundCodec import codec

CONFIG = {
    "encodec_24k_1_5bps": "encodec_24k_1_5bps",
    "encodec_24k_3bps": "encodec_24k_3bps",
    "encodec_24k_6bps": "encodec_24k_6bps",
    "encodec_24k_12bps": "encodec_24k_12bps",
    "encodec_24k_24bps": "encodec_24k_24bps"
}


class EncodecTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.model = codec.load_codec(config)

    def encode(self, input_array, sampling_rate):
        data_item = {'audio': {'array': input_array,
                               'sampling_rate': sampling_rate}}
        unit_item = self.model.extract_unit(data_item)
        return unit_item.unit,unit_item.stuff_for_synth

    def encode_file(self, input_file):
        # read audio file into array and sampling_rate
        input_array, sampling_rate = soundfile.read(input_file)
        data_item = {'audio': {'array': input_array,
                               'sampling_rate': sampling_rate}}
        unit_item = self.model.extract_unit(data_item)
        return unit_item.unit,unit_item.stuff_for_synth

    def decode(self, stuff_for_decode):
        return self.model.decode_unit(stuff_for_decode)
