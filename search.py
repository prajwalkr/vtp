# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

# pylint: disable=too-many-statements,too-many-branches
def beam_search(
    decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (3d array of indices) -> batch_size x n_best x t_dec
        - stacked_attention_scores: attention scores (5d array) -> batch_size x n_best x n_dec_layers x t_dec x t_enc
    """
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    # init
    batch_size = src_mask.size(0)

    encoder_output = tile(
        encoder_output.contiguous(), size, dim=0
    )  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0) # batch*k x 1 x src_len

    trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only

    # numbering elements in the batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device
    ) # e.g for bs=16 -> [0,1,2,3....15]

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
    ) # e.g for bs=16 -> [0,3,6,8,...45]

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device,
    ) # bs*k x t_dec

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "attentions" : [[] for _ in range(batch_size)],
        # "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):

        decoder_input = alive_seq  # complete prediction so far

        from dataloader import subsequent_mask
        logits = decoder.decode(encoder_output, 
                            src_mask,
                            decoder_input,
                            subsequent_mask(decoder_input.size(1)).to(encoder_output.device).long(),
                            )
        logits = logits.reshape( (decoder_input.size(0), -1, logits.size(-1)))

        logits = logits[:, -1]

        # batch*k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        output_size = log_probs.size(-1)

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        # reconstruct beam origin and true word ids from flattened order
        try:
            topk_beam_index = topk_ids.div(output_size, rounding_mode='floor')
        except:
            # HACK: to make it work on Gul's old pytorch
            topk_beam_index = topk_ids.div(output_size) # parent index 
        topk_ids = topk_ids.fmod(output_size) # word index 

        # map beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[
            : topk_beam_index.size(0)
        ].unsqueeze(1)
        # for each batch index this determines which partents to get the hypotheses + logprobs
        # w.r.t the flattened batch indices
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )  # batch_size*k x hyp_len

        # select the attention scores for the alive sequences 
        # att_scores = att_scores.index_select(0, select_indices) # batch_size*k x n_dec_layers x t_dec x t_enc

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(True)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            # curr_att = att_scores.view( (-1, size) + att_scores.shape[1:] ) # n_alive x k x n_dec_layers x t_dec x t_enc
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = torch.nonzero(is_finished[i], as_tuple=False).view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    # Check if the prediction has more than one EOS.
                    # If it has more than one EOS, it means that the
                    # prediction should have already been added to
                    # the hypotheses, so you don't have to add them again.
                    if torch.nonzero(predictions[i, j, 1:] == eos_index, as_tuple=False).numel() < 2:
                        # ignore start_token
                        # hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:], curr_att[i,j]))
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))

                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    # for n, (score, pred, att) in enumerate(best_hyp):
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                        # results["attentions"][b].append(att.detach().cpu())
            non_finished = torch.nonzero(end_condition.eq(False), as_tuple=False).view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    final_outputs = [ [ rr.detach().cpu() for rr in br ] for br in results["predictions"] ]
    final_scores = [ [ rr.detach().cpu().item() for rr in br ] for br in results["scores"] ]
    # final_outputs = torch.from_numpy(final_outputs)


    return final_outputs, final_scores
