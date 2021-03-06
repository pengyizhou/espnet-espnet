# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Mask CTC based non-autoregressive speech recognition model (pytorch).

See https://arxiv.org/abs/2005.08700 for the detail.

"""

from itertools import groupby
import logging
import math
import torch.nn.functional as F
from distutils.util import strtobool
import numpy
import torch
import ipdb
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.maskctc.add_mask_token import mask_uniform
from espnet.nets.pytorch_backend.maskctc.mask import square_mask
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy

from espnet.nets.pytorch_backend.calcute_mwer import Calcute_Minmum_Word_Error_Rate
class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_maskctc_arguments(parser)

        return parser

    @staticmethod
    def add_maskctc_arguments(parser):
        """Add arguments for maskctc model."""
        group = parser.add_argument_group("maskctc specific setting")

        group.add_argument(
            "--maskctc-use-conformer-encoder",
            default=False,
            type=strtobool,
        )
        group = add_arguments_conformer_common(group)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        odim += 1  # for the mask token

        super().__init__(idim, odim, args, ignore_id)
        assert 0.0 <= self.mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"

        self.mask_token = odim - 1
        self.sos = odim - 2
        self.eos = odim - 2
        self.odim = odim
        logging.warning("odim: {}, self.sos: {}".format(self.odim, self.eos))
        if args.maskctc_use_conformer_encoder:
            if args.transformer_attn_dropout_rate is None:
                args.transformer_attn_dropout_rate = args.conformer_dropout_rate
            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
                selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
                activation_type=args.transformer_encoder_activation_type,
                macaron_style=args.macaron_style,
                use_cnn_module=args.use_cnn_module,
                cnn_module_kernel=args.cnn_module_kernel,
            )
        self.calcute_mwer = Calcute_Minmum_Word_Error_Rate(1)
        self.char_list = args.char_list
        self.NBest = 4
        self.maskctc_probability_threshold = 0.9
        # self.maskctc_n_iterations = 10
        self.reset_parameters(args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # logging.warning("Start Encoder forward for batch")
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad
        # logging.warning("End Encoder forward for batch")
        ys_in_pad, ys_out_pad = mask_uniform(
            ys_pad, self.mask_token, self.eos, self.ignore_id
        )
        ys_mask = square_mask(ys_in_pad, self.eos)
        # logging.warning("End Random Mask for labels")
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        # logging.warning("End Decoder Forward for batch")
        # calcute mwer loss
        # mwer_loss = self.calcute_mwer.get_Mwer_loss(new_pred_pad, ys_out_pad)
        # logging.warning("mwer_loss: {}".format(mwer_loss))
        mwer_loss = 0
        for i in range(xs_pad.shape[0]):
            # logging.warning("Start recognize for one utt")
            # ipdb.set_trace()
            bestScores = []
            bestIds = []
            for best in range(self.NBest):
                # ipdb.set_trace()
                # best_scores, best_ids = self.recognize_mwer(xs_pad[i], self.maskctc_probability_threshold, 10)
                best_scores, best_ids = self.recognize_mwer_rand_mask(xs_pad[i], 10)
                bestScores.append(best_scores)
                bestIds.append(best_ids)

            # logging.warning("ys_pad[i]: {}".format(ys_pad[i]))
            # logging.warning("End recognize for one utt")
            # ipdb.set_trace()
            
            mwer_loss += self.calcute_mwer.getMwerLossFromNbestCTCPositive(bestScores, bestIds, ys_pad[i])
        logging.warning("mwer_loss for one batch: {}".format(mwer_loss))
        logging.warning("End train for one batch")
        self.pred_pad = pred_pad
        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # 4. compute ctc loss
        loss_ctc, cer_ctc = None, None
        if self.mtlalpha > 0:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)
        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att + mwer_loss
            loss_att_data = 0.01 * float(loss_att) + float(mwer_loss)
            loss_ctc_data = None
        else:
            # self.loss = 0.01 * (alpha * loss_ctc + (1 - alpha) * loss_att) + mwer_loss
            # self.loss = alpha * loss_ctc + (1 - alpha) * (0.01 * loss_att + mwer_loss)
            # self.loss = 0.01 * loss_att + mwer_loss 
            
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att + mwer_loss - float(mwer_loss)
            loss_att_data = float(loss_att) 
            loss_ctc_data = float(loss_ctc)
            loss_mwer_data = float(mwer_loss)

        loss_data = float(self.loss)
        logging.warning("## Att loss: {}, org loss: {}, final loss: {}".format(float(loss_att), float(alpha * loss_ctc + (1 - alpha) * loss_att), loss_data))
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report_mwer(
                loss_ctc_data, loss_att_data, loss_mwer_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """

        def num2str(char_list, mask_token, mask_char="_"):
            def f(yl):
                cl = [char_list[y] if y != mask_token else mask_char for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(char_list, self.mask_token)

        self.eval()
        h = self.encode(x).unsqueeze(0)

        # greedy ctc outputs
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
        logging.warning("ctc_probs.shape: " + str(ctc_probs.shape) + "\n" + str(ctc_probs)) # [1, 104]
        logging.warning("ctc_ids.shape: " + str(ctc_ids.shape) + "\n" + str(ctc_ids)) # [1, 104]

        # for x in groupby(ctc_ids[0]):
        #     logging.warning("x: {}".format(x))
        # torch.stack():??????????????????????????????????????????????????????
        # groupby: ???ctc_ids???????????????id????????????
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        logging.warning("y_hat: {}, value: {}".format(y_hat.shape, y_hat)) # y_hat: torch.Size([26]), value: tensor([   0, 1917,    0, 1657,    0,  675,    0,  582,    0, 1832,    0, 1380, 54,    0,  486,    0, 1203,    0,  908,    0, 1539,    0,  580,    0, 702,    0])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1) # y_hat???id??????0?????????
        logging.warning("y_idx: {}, value: {}".format(y_idx.shape, y_idx)) # y_idx: torch.Size([13]), value: tensor([ 1,  3,  5,  7,  9, 11, 12, 14, 16, 18, 20, 22, 24])

        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        probs_hat = []
        cnt = 0
        # ??????y_hat???????????????????????????????????????????????????????????????????????????????????????
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item() # item(), ??????????????????????????????????????????????????????????????????????????????
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))
        logging.warning("probs_hat: {}".format(probs_hat))
        # mask ctc outputs based on ctc probabilities
        p_thres = recog_args.maskctc_probability_threshold
        logging.warning("probs_hat[y_idx]: {}".format(probs_hat[y_idx]))
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        logging.warning("p_thres: {}".format(p_thres))
        logging.warning("mask_idx: {}".format(mask_idx))
        logging.warning("confident_idx: {}".format(confident_idx))

        mask_num = len(mask_idx)

        y_in = torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token # y_in value: ????????????????????????1985?????????tensor([[1985, 1985, 1985, 1985, 1985, 1985, 1985, 1985, 1988...
        logging.warning("mask_token: {}; y_in.shape(): {}; y_in value: {}".format(self.mask_token, y_in.shape, y_in))
        # ???confident_idx???????????????????????????mask token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx]
        logging.warning("y_in: {}".format(y_in))
        logging.info("ctc:{}".format(n2s(y_in[0].tolist())))

        # iterative decoding
        if not mask_num == 0:
            K = recog_args.maskctc_n_iterations   # 10
            num_iter = K if mask_num >= K and K > 0 else mask_num
            logging.warning("K: {}; num_iter: {}".format(K, num_iter))
            for t in range(num_iter - 1):
                pred, _ = self.decoder(y_in, None, h, None) # [1, 20, 1986], ???ctc?????????????????????
                logging.warning("pred.shape: {}; pred value: {}".format(pred.shape, pred))
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1) # ????????????pred????????????????????????mask?????????
                logging.warning("pred_score: {}; pred_id: {}".format(pred_score, pred_id))
                # topk(???????????????k?????????)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                logging.warning("cand: {}".format(cand))
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)

                logging.info("msk:{}".format(n2s(y_in[0].tolist())))

            # predict leftover masks (|masks| < mask_num // num_iter)
            pred, pred_mask = self.decoder(y_in, None, h, None)
            y_in[0][mask_idx] = pred[0][mask_idx].argmax(dim=-1)

            logging.info("msk:{}".format(n2s(y_in[0].tolist())))

        ret = y_in.tolist()[0]
        hyp = {"score": 0.0, "yseq": [self.sos] + ret + [self.eos]}

        return [hyp]


    def recognize_mwer(self, x, maskctc_probability_threshold=0.999, maskctc_n_iterations=10):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """

        def num2str(char_list, mask_token, mask_char="_"):
            def f(yl):
                cl = [char_list[y] if y != mask_token else mask_char for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(self.char_list, self.mask_token)

        self.eval()
        h = self.encode(x).unsqueeze(0)

        # greedy ctc outputs
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
        # logging.warning("ctc_probs.shape: " + str(ctc_probs.shape) + "\n" + str(ctc_probs)) # [1, 104]
        # logging.warning("ctc_ids.shape: " + str(ctc_ids.shape) + "\n" + str(ctc_ids)) # [1, 104]

        
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1) # y_hat???id??????0?????????

        probs_hat = []
        cnt = 0
        # ??????y_hat???????????????????????????????????????????????????????????????????????????????????????
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item() # item(), ??????????????????????????????????????????????????????????????????????????????
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))
        p_thres = maskctc_probability_threshold
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)

        mask_num = len(mask_idx)
        logging.warning("## At threshold {}, mask number is {}, mask token index is {}, total length is {}".format(p_thres, mask_num, mask_idx, len(probs_hat)))
        best_scores = [[] for i in range(y_idx.shape[0])]
        best_ids = [[self.mask_token] for i in range(y_idx.shape[0])]
        for i in range(confident_idx.shape[0]):
            best_scores[confident_idx[i]] = [0] 
            best_ids[confident_idx[i]] = [int(y_hat[y_idx][confident_idx[i]])]
        # logging.warning("best_scores: {}".format(best_scores))
        # logging.warning("best_ids: {}".format(best_ids))

        y_in = torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token # y_in value: ????????????????????????1985?????????tensor([[1985, 1985, 1985, 1985, 1985, 1985, 1985, 1985, 1988...
        # ???confident_idx???????????????????????????mask token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx].cpu()
        # logging.info("ctc:{}".format(n2s(y_in[0].tolist())))
        # iterative decoding
        if not mask_num == 0:
            K = maskctc_n_iterations   # 10
            num_iter = K if mask_num >= K and K > 0 else mask_num
            # logging.warning("K: {}; num_iter: {}".format(K, num_iter))
            for t in range(num_iter - 1):
                temp_mask_idx = mask_idx
                y_in = y_in.to(h.device)
                # pred = pred.to(y_in.device)
                pred, _ = self.decoder(y_in, None, h, None) # [1, 20, 1986], ???ctc?????????????????????
                pred = F.log_softmax(pred, -1)
                
                # logging.warning("pred: {}".format(pred))
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1) # ????????????pred????????????????????????mask?????????
                # topk(???????????????k?????????)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)
                
                ## add for mwer
                pred_score_mwer, pred_id_mwer = torch.topk(pred[0][temp_mask_idx], 2, -1)
                
                for i in cand:
                    id = temp_mask_idx[i]
                    best_scores[id] = pred_score_mwer[i]
                    best_ids[id] = pred_id_mwer[i]
            # predict leftover masks (|masks| < mask_num // num_iter)
            y_in = y_in.to(h.device) 
            pred, pred_mask = self.decoder(y_in, None, h, None)
            pred = F.log_softmax(pred, -1)
            ## add for mwer
            pred_score_mwer, pred_id_mwer = torch.topk(pred[0][mask_idx], 2, -1)
            for i in range(mask_idx.shape[0]):
                best_scores[mask_idx[i]] = pred_score_mwer[i]
                best_ids[mask_idx[i]] = pred_id_mwer[i]

        return best_scores, best_ids

    def recognize_mwer_rand_mask(self, x, maskctc_n_iterations=10):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """

        def num2str(char_list, mask_token, mask_char="_"):
            def f(yl):
                cl = [char_list[y] if y != mask_token else mask_char for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(self.char_list, self.mask_token)

        self.eval()
        h = self.encode(x).unsqueeze(0)

        # greedy ctc outputs
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
        # logging.warning("ctc_probs.shape: " + str(ctc_probs.shape) + "\n" + str(ctc_probs)) # [1, 104]
        # logging.warning("ctc_ids.shape: " + str(ctc_ids.shape) + "\n" + str(ctc_ids)) # [1, 104]

        
        y_hat = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx = torch.nonzero(y_hat != 0).squeeze(-1) # y_hat???id??????0?????????

        probs_hat = []
        cnt = 0
        # ??????y_hat???????????????????????????????????????????????????????????????????????????????????????
        for i, y in enumerate(y_hat.tolist()):
            probs_hat.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if probs_hat[i] < ctc_probs[0][cnt]:
                    probs_hat[i] = ctc_probs[0][cnt].item() # item(), ??????????????????????????????????????????????????????????????????????????????
                cnt += 1
        probs_hat = torch.from_numpy(numpy.array(probs_hat))
        if len(y_idx) > 0:
            mask_num = numpy.random.randint(1, len(y_idx) + 1)
            mask_idx = list(set(list(numpy.random.choice(len(y_idx), mask_num))))
            mask_idx.sort()
        else:
            mask_idx = []
        confident_idx = []
        for idx in range(len(y_idx)):
            if idx not in mask_idx:
                confident_idx.append(idx)
        """
        p_thres = maskctc_probability_threshold
        mask_idx = torch.nonzero(probs_hat[y_idx] < p_thres).squeeze(-1)
        confident_idx = torch.nonzero(probs_hat[y_idx] >= p_thres).squeeze(-1)
        """
        confident_idx = torch.tensor(confident_idx).type(torch.long)
        mask_idx = torch.tensor(mask_idx).type(torch.long)
        # ipdb.set_trace()
        mask_num = len(mask_idx)
        logging.warning("## mask number is {}, mask token index is {}, total length is {}".format(mask_num, mask_idx, len(y_idx)))
        best_scores = [[] for i in range(y_idx.shape[0])]
        best_ids = [[self.mask_token] for i in range(y_idx.shape[0])]
        for i in range(confident_idx.shape[0]):
            best_scores[confident_idx[i]] = [0] 
            best_ids[confident_idx[i]] = [int(y_hat[y_idx][confident_idx[i]])]
        # logging.warning("best_scores: {}".format(best_scores))
        # logging.warning("best_ids: {}".format(best_ids))

        y_in = torch.zeros(1, len(y_idx), dtype=torch.long) + self.mask_token # y_in value: ????????????????????????1985?????????tensor([[1985, 1985, 1985, 1985, 1985, 1985, 1985, 1985, 1988...
        # ???confident_idx???????????????????????????mask token
        y_in[0][confident_idx] = y_hat[y_idx][confident_idx].cpu()
        # logging.info("ctc:{}".format(n2s(y_in[0].tolist())))
        # iterative decoding
        
        if not mask_num == 0:
            K = maskctc_n_iterations   # 10
            num_iter = K if mask_num >= K and K > 0 else mask_num
            # logging.warning("K: {}; num_iter: {}".format(K, num_iter))
            for t in range(num_iter - 1):
                temp_mask_idx = mask_idx
                y_in = y_in.to(h.device)
                # pred = pred.to(y_in.device)
                pred, _ = self.decoder(y_in, None, h, None) # [1, 20, 1986], ???ctc?????????????????????
                pred = F.log_softmax(pred, -1)
                
                # logging.warning("pred: {}".format(pred))
                pred_score, pred_id = pred[0][mask_idx].max(dim=-1) # ????????????pred????????????????????????mask?????????
                # topk(???????????????k?????????)
                cand = torch.topk(pred_score, mask_num // num_iter, -1)[1]
                y_in[0][mask_idx[cand]] = pred_id[cand]
                mask_idx = torch.nonzero(y_in[0] == self.mask_token).squeeze(-1)
                
                ## add for mwer
                pred_score_mwer, pred_id_mwer = torch.topk(pred[0][temp_mask_idx], 2, -1)
                
                for i in cand:
                    id = temp_mask_idx[i]
                    best_scores[id] = pred_score_mwer[i]
                    best_ids[id] = pred_id_mwer[i]
            # predict leftover masks (|masks| < mask_num // num_iter)
            y_in = y_in.to(h.device) 
            pred, pred_mask = self.decoder(y_in, None, h, None)
            pred = F.log_softmax(pred, -1)
            ## add for mwer
            pred_score_mwer, pred_id_mwer = torch.topk(pred[0][mask_idx], 2, -1)
            for i in range(mask_idx.shape[0]):
                best_scores[mask_idx[i]] = pred_score_mwer[i]
                best_ids[mask_idx[i]] = pred_id_mwer[i]

        return best_scores, best_ids

