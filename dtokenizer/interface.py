class BaseTokenizer:
    def __init__(self):
        pass

    def encode(self, input_value):
        return input_value

    def encode_file(self, input_file):
        return input_file

    def decode(self, code):
        return code

    def batch_encode(self, input_values):
        return [self.encode(input_value) for input_value in input_values]

    def batch_file_encode(self, input_values):
        return [self.encode_file(input_value) for input_value in input_values]

    def batch_decode(self, codes):
        return [self.decode(code) for code in codes]
