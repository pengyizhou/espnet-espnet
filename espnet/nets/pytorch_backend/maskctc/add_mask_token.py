#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Token masking module for Masked LM."""

import numpy
import math
import logging

def mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples)

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


# for mask token list
def new_mask_uniform(ys_pad, mask_token, eos, ignore_id, scale):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    # logging.warning("ys_pad: {}".format(ys_pad))

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]

    # logging.warning("ys.len: {}, ys: {}".format(len(ys[0]), ys[0])) # 去除了-1
    # logging.warning("ys_out.len: {}, ys_out: {}".format(len(ys_out[0]), ys_out[0])) # 全是-1， 长度跟真正token长度一样
    # logging.warning("ys_in.len: {}, ys_in: {}".format(len(ys_in[0]), ys_in[0])) # 跟ys一致
    for i in range(len(ys)):
        # logging.warning("## {}".format(int(math.ceil(len(ys[i]) * 0.30) + 1)))
        # num_samples = numpy.random.randint(1, int(math.ceil(len(ys[i]) * float(scale)) + 1))
        num_samples = int(math.ceil(len(ys[i]) * float(scale)) + 1)

        idx = numpy.random.choice(len(ys[i]), num_samples)

        # logging.warning("num_samples: {}".format(num_samples))
        # logging.warning("idx: {}".format(idx))

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]
    # logging.warning("final ys_in.len: {}, ys_in: {}".format(len(ys_in[0]), ys_in[0])) 
    # logging.warning("final ys_in_pad.len: {}, ys_in_pad: {}".format(len(pad_list(ys_in, eos)[0]), pad_list(ys_in, eos)[0])) 
    # logging.warning("final ys_out.len: {}, ys_out: {}".format(len(ys_out[0]), ys_out[0])) 
    # logging.warning("final ys_out_pad.len: {}, ys_out: {}".format(len(pad_list(ys_out, ignore_id)), pad_list(ys_out, ignore_id)[0])) 

    # logging.warning("final ys_in: {}".format(ys_in[0]))
    # logging.warning("final ys_in_pad: {}".format(pad_list(ys_in, eos)[0]))
    # logging.warning("final ys_out: {}".format(ys_out[0]))
    # return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

    return pad_list(ys_in, ignore_id)
# for mask token list
def new_mask_uniform2(ys_pad, mask_token, eos, ignore_id, scale):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list
    # logging.warning("ys_pad: {}".format(ys_pad))

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]

    # logging.warning("ys.len: {}, ys: {}".format(len(ys[0]), ys[0])) # 去除了-1
    # logging.warning("ys_out.len: {}, ys_out: {}".format(len(ys_out[0]), ys_out[0])) # 全是-1， 长度跟真正token长度一样
    # logging.warning("ys_in.len: {}, ys_in: {}".format(len(ys_in[0]), ys_in[0])) # 跟ys一致
    for i in range(len(ys)):
        # logging.warning("## {}".format(int(math.ceil(len(ys[i]) * 0.30) + 1)))
        # num_samples = numpy.random.randint(1, int(math.ceil(len(ys[i]) * float(scale)) + 1))
        num_samples = int(math.ceil(len(ys[i]) * float(scale)) + 1)

        idx = numpy.random.choice(len(ys[i]), num_samples)

        # logging.warning("num_samples: {}".format(num_samples))
        # logging.warning("idx: {}".format(idx))

        ys_in[i][idx] = mask_token

        index = [i for i in range(0, len(ys[i]))]
        ys_out[i][index] = ys[i][index]
        # ys_out[i][idx] = ys[i][idx]
    # logging.warning("final ys_in.len: {}, ys_in: {}".format(len(ys_in[0]), ys_in[0])) 
    # logging.warning("final ys_in_pad.len: {}, ys_in_pad: {}".format(len(pad_list(ys_in, eos)[0]), pad_list(ys_in, eos)[0])) 
    # logging.warning("final ys_out.len: {}, ys_out: {}".format(len(ys_out[0]), ys_out[0])) 
    # logging.warning("final ys_out_pad.len: {}, ys_out: {}".format(len(pad_list(ys_out, ignore_id)), pad_list(ys_out, ignore_id)[0])) 

    # logging.warning("final ys_in: {}".format(ys_in[0]))
    # logging.warning("final ys_in_pad: {}".format(pad_list(ys_in, eos)[0]))
    # logging.warning("final ys_out: {}".format(ys_out[0]))
    # return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def coupling_mask_uniform(ys_pad, mask_token, eos, ignore_id):
    """Replace random tokens with <mask> label and add <eos> label.

    The number of <mask> is chosen from a uniform distribution
    between one and the target sequence's length.
    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int mask_token: index of <mask>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_out = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in = [y.clone() for y in ys]

    ys_out_coup = [y.new(y.size()).fill_(ignore_id) for y in ys]
    ys_in_coup = [y.clone() for y in ys]
    for i in range(len(ys)):
        num_samples = numpy.random.randint(1, len(ys[i]) + 1)
        idx = numpy.random.choice(len(ys[i]), num_samples)
        coup = list()
        for j in range(len(ys[i])):
            if j not in idx:
                coup.append(j)

        ys_in[i][idx] = mask_token
        ys_out[i][idx] = ys[i][idx]

        ys_in_coup[i][numpy.array(coup)] = mask_token
        ys_out_coup[i][numpy.array(coup)] = ys[i][numpy.array(coup)]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id), pad_list(ys_in_coup, eos), pad_list(ys_out_coup, ignore_id)
