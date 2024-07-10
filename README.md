# dtokenizer
discretize everything into tokens

## Introduction
`dtokenizer` is a Python library designed to discretize audio files into tokens using various models. It supports models like Hubert and Encodec for tokenization.

## Installation
To use `dtokenizer`, first ensure you have Python and pip installed. Then, install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

### Hubert Tokenizer
The Hubert tokenizer can be used to tokenize audio files into discrete tokens and then decode them back. Here's how you can use it:

```python
from dtokenizer.audio.model.hubert_model import HubertTokenizer
import soundfile as sf

ht = HubertTokenizer('hubert_layer6_code100')
code, decodec_stuff = ht.encode_file('./sample2_22k.wav')
wav_values = ht.decode(code)

# Write the decoded audio to a file
sf.write('output.wav', wav_values, 16000)
```

### Encodec Tokenizer
Similarly, the Encodec tokenizer allows for efficient audio file tokenization. Here's an example of its usage:

```python
import torch
from dtokenizer.audio.model.encodec_model import EncodecTokenizer
import torchaudio

et = EncodecTokenizer('encodec_24k_6bps')
code, stuff_for_decode = et.encode_file('./sample2_22k.wav')
wav_values = et.decode(stuff_for_decode)

# Save the decoded audio to a file
torchaudio.save('output.wav', torch.from_numpy(wav_values), 22050)
```

## Contributing
We welcome contributions to the `dtokenizer` project. Please feel free to submit issues or pull requests.

## License
This project is released under the MIT License. See the LICENSE file for more details.