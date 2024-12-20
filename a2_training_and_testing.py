from tqdm import tqdm
import typing

import torch

import a2_bleu_score
import a2_dataloader
import a2_encoder_decoder


def train_for_epoch(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.wmt16DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device) -> float:
    '''Train an EncoderDecoder for an epoch
    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : wmt16DataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''


    model.train()
    total_loss = 0.0
    total_count = 0
    decoder_padding_id = 20000 # I am using an arbitrary token id for the loss function to use as padding.
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=decoder_padding_id)
    progress_bar = tqdm(dataloader, desc='Training')
    #print('pad id', str(model.decoder.pad_id))
    #print('target eos', str(model.target_eos))

    for F, F_lens, E in progress_bar:
        F, F_lens, E = F.to(device), F_lens.to(device), E.to(device)
        optimizer.zero_grad()

        # Calculating the logits by sending the inputs to model.
        logits = model(F, F_lens, E)

        # Sending the E_target to the get_target_padding_mask and applying the padding mask using
        # my decoder_padding_id I define above.
        E_target = E[1:]
        pad_mask = model.get_target_padding_mask(E_target)
        E_target_masked = torch.masked_fill(E_target, pad_mask, decoder_padding_id)

        # Reshaping the logits and E_target_masked to calculate the loss.
        E_target_masked = E_target_masked.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))

        loss = loss_fn(logits, E_target_masked)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * F.size(0)  # Multiply loss by batch size for total loss
        total_count += F.size(0)

        progress_bar.set_description(f"Training (loss {loss.item():.4f})")
        del F, F_lens, E, logits, loss
    avg_loss = total_loss / total_count
    return avg_loss
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.

def compute_batch_total_bleu(
        E_ref: torch.LongTensor,
        E_cand: torch.LongTensor,
        target_sos: int,
        target_eos: int) -> float:
    '''Compute the total BLEU score over elements in a batch
    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # Calculating the total bleu for the batch by passing the
    # candidate and reference sequences to  a2_bleu_score.BLEU_score() function.
    total_bleu = 0.0
    ref_lists = E_ref.t().tolist()
    cand_lists = E_cand.t().tolist()
    for ref, cand in zip(ref_lists, cand_lists):
        filtered_ref = [x for x in ref if x != target_sos and x != target_eos] # Filtering out eos and sos tokens.
        filtered_cand = [x for x in cand if x != target_sos and x != target_eos] # Filtering out eos and sos tokens.
        if len(filtered_cand) > 0:
            score = a2_bleu_score.BLEU_score(filtered_ref, filtered_cand, 4)
            total_bleu += score
    return total_bleu
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers

def compute_average_bleu_over_dataset(
        model: a2_encoder_decoder.EncoderDecoder,
        dataloader: a2_dataloader.wmt16DataLoader,
        target_sos: int,
        target_eos: int,
        device: torch.device) -> float:
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : wmt16DataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''

    # Calculating the average BLEU using compute_batch_total_bleu
    model.eval()
    total_bleu_score = 0.0
    total_sequences = 0
    for F, F_lens, E_ref in tqdm(dataloader, desc="Evaluating BLEU"):

        F = F.to(device)
        F_lens = F_lens.to(device)

        b_1 = model(F, F_lens)
        E_cand = b_1[..., 0]

        batch_bleu_score = compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos)
        total_bleu_score += batch_bleu_score
        total_sequences += E_ref.size(1)

    avg_bleu = total_bleu_score / total_sequences if total_sequences > 0 else 0

    return avg_bleu


