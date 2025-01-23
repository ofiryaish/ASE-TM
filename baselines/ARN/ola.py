import torch
from torch.autograd import Variable


def pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2
    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(
        batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)
    return input, rest


def create_chuncks(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size,
                                                                dim, -1, segment_size)
    segments2 = input[:, :, segment_stride:].contiguous().view(
        batch_size, dim, -1, segment_size)
    segments = torch.cat([segments1, segments2], 3).view(
        batch_size, dim, -1, segment_size).transpose(2, 3)
    return segments.contiguous(), rest


def merge_chuncks(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = input.transpose(2, 3).contiguous().view(
        batch_size, dim, -1, segment_size*2)  # B, N, K, L

    input1 = input[:, :, :, :segment_size].contiguous().view(
        batch_size, dim, -1)[:, :, segment_stride:]
    input2 = input[:, :, :, segment_size:].contiguous().view(
        batch_size, dim, -1)[:, :, :-segment_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]
    return output.contiguous()  # B, N, T
