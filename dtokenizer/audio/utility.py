import torch


def collate_fn_pad(batch, device):
    '''
    Padds batch of variable length
    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    ## padd
    batch = [torch.Tensor(t).to(device) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
