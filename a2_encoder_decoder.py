import torch
from typing import Optional, Union, Tuple, Type, Set
from torch import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}

        # Initializing the embedding layer.
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.source_vocab_size,
            embedding_dim=self.word_embedding_size,
            padding_idx=self.pad_id,
        )

        # Selecting the appropriate RNN type based on cell_type
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional=True,
            )
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional=True,
            )
        else:
            self.rnn = torch.nn.RNN(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size,
                num_layers=self.num_hidden_layers,
                dropout=self.dropout,
                bidirectional=True,
            )

    def forward_pass(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Calculating the encoder hidden states by first getting the inputs,
        # then passing inputs to the get_all_hidden_states function.
        x = self.get_all_rnn_inputs(F)

        h = self.get_all_hidden_states(x, F_lens, h_pad)

        return h
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states


    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:

        # Getting all rnn inputs by passing the F tensor to the embedding layer.
        x = self.embedding(F)  # (S, M, I)

        return x
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)

    def get_all_hidden_states(
            self,
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:

        F_lens_cpu = F_lens.to('cpu')

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens_cpu, batch_first=False, enforce_sorted=False)

        packed_h, _ = self.rnn(packed_x)

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=False, padding_value=h_pad)

        return h

        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):

        # Initializing the embedding and cells.
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.target_vocab_size,
            embedding_dim=self.word_embedding_size,
            padding_idx=self.pad_id,
        )

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size
            )
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size
            )
        else:
            self.cell = torch.nn.RNNCell(
                input_size=self.word_embedding_size,
                hidden_size=self.hidden_state_size
            )
        # Initializing the linear layer.
        self.ff = torch.nn.Linear(
            in_features=self.hidden_state_size,
            out_features=self.target_vocab_size,
        )

        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}


    def forward_pass(
        self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> Tuple[
                torch.FloatTensor, Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Getting the current input, then getting the current hidden state using current input.
        # Then, using the current hidden state, getting the logits for tokens.
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)

        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        if isinstance(htilde_t, tuple):
            htilde_for_logits = htilde_t[0]
        else:
            htilde_for_logits = htilde_t

        logits_t = self.get_current_logits(htilde_for_logits)

        return (logits_t, htilde_t)

        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # Getting the first hidden state of the decoder based on the encoder hidden states.
        M = h.shape[1]

        forward_states = h[F_lens - 1, torch.arange(M), :self.hidden_state_size // 2] # (..,0:511)

        backward_states = h[0, :, self.hidden_state_size // 2:] # (..,512:1023)

        htilde_0 = torch.cat((forward_states, backward_states), dim=1) # (M x 1024)

        return htilde_0

        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # Getting the current decoder input.
        embedded = self.embedding(E_tm1) # (M x word_embedding_size)

        xtilde_t = embedded # (M x word_embedding_size)

        return xtilde_t
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:

        # Getting the current hidden state of the decoder by passing the current input and current previous to the cell.
        if self.cell_type == 'lstm':
            htilde_t, ctilde_t = self.cell(xtilde_t, htilde_tm1)
            return (htilde_t, ctilde_t)
        else:
            htilde_t = self.cell(xtilde_t, htilde_tm1)
            return htilde_t

        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1


    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:

        # Getting the logits by passing the hidden state to the linear layer.
        logits_t = self.ff(htilde_t)

        return logits_t

        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)



class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):

        # Initializing the layers.
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.target_vocab_size,
            embedding_dim=self.word_embedding_size,
            padding_idx=self.pad_id,
        )
        # The input is composed of word_embedding_size + attention_layer_size
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(
                input_size=self.word_embedding_size + self.hidden_state_size,
                hidden_size=self.hidden_state_size
            )
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(
                input_size=self.word_embedding_size + self.hidden_state_size,
                hidden_size=self.hidden_state_size
            )
        else:
            self.cell = torch.nn.RNNCell(
                input_size=self.word_embedding_size + self.hidden_state_size,
                hidden_size=self.hidden_state_size
            )

        self.ff = torch.nn.Linear(
            in_features=self.hidden_state_size,
            out_features=self.target_vocab_size,
        )
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        M = h.shape[1]
        htilde_0 = torch.zeros(M, self.hidden_state_size, device=h.device)
        return htilde_0
        # Hint: For this time, the hidden states should be initialized to zeros.


    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # Getting the current rnn input by concatenating the word embedding and context vector from attention.
        embedded = self.embedding(E_tm1)  # (M, word_embedding_size)
        c_t = self.attend(htilde_tm1, h, F_lens)  # (M, hidden_state_size)
        xtilde_t = torch.cat((embedded, c_t), dim=1)  # (M, word_embedding_size + hidden_state_size)
        return xtilde_t
        # Hint: Use attend() for c_t


    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # Getting the attention context vector using by taking the weighted sum of encoder hidden layers.
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)  # (S, M)
        c_t = torch.sum(h * alpha_t[:, :, None], dim=0)
        return c_t


    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens.to(h.device)  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:

        # Getting the attention scores of encoder states using
        # cosine similarity between current decoder hidden state and all encoder hidden states.

        if isinstance(htilde_t, tuple):
            htilde_t = htilde_t[0]


        htilde_t = htilde_t.unsqueeze(0)  # Now (1, M, 2*H)
        htilde_t = htilde_t.expand(h.shape[0], -1, -1)  # Now (S, M, 2*H)
        e_t = torch.nn.functional.cosine_similarity(h, htilde_t, dim=2)
        return e_t

        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line
        self.W = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.hidden_state_size,bias=False)
        self.Wtilde = torch.nn.Linear(in_features=self.hidden_state_size,out_features=self.hidden_state_size,bias=False)
        self.Q = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.hidden_state_size,bias=False)

        # Hints:
            # 1. The above line should ensure self.ff, self.embedding, self.cell are
            #    initialized
            # 2. You need to initialize the following submodules:
            #       self.W, self.Wtilde, self.Q
            # 3. You will need the following object attributes:
            #       self.hidden_state_size
            # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
            #    should not be lists!
            # 5. You do *NOT* need self.heads at this point
            # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)


    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # Calculating attention for each head by separately calling super.attend() for each head
        # and concatenating the attention vectors.
        if isinstance(htilde_t, tuple):
            htilde_t = htilde_t[0]


        head_dim = self.hidden_state_size // self.heads
        batch_size = h.size(1)

        # Transforming hidden states for multiple heads
        Q_t = self.Q(htilde_t).view(batch_size, self.heads, head_dim)
        W_h = self.W(h).view(-1, batch_size, self.heads, head_dim).transpose(0, 2)

        combined_context = []
        for i in range(self.heads):
            head_W_h = W_h[i].transpose(0, 1)
            head_Q_t = Q_t[:, i, :]

            head_context = super().attend(head_Q_t, head_W_h, F_lens)
            combined_context.append(head_context.unsqueeze(1))

        context_vector = torch.cat(combined_context, dim=1).view(batch_size, -1)

        return context_vector

        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        assert False, 'Fill me!'

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):

        # Initializing the encoder and decoder.
        self.encoder = encoder_class(
            source_vocab_size=self.source_vocab_size,
            pad_id=self.source_pad_id,
            word_embedding_size=self.word_embedding_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            hidden_state_size=self.encoder_hidden_size,
            dropout=self.encoder_dropout,
            cell_type=self.cell_type,
        )

        self.decoder = decoder_class(
            target_vocab_size=self.target_vocab_size,
            pad_id=self.target_eos,
            word_embedding_size=self.word_embedding_size,
            hidden_state_size=2*self.encoder_hidden_size,
            cell_type=self.cell_type,
            heads=self.heads,
        )
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it


    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor,
            E: torch.LongTensor) -> torch.FloatTensor:

        # Calculating the logits using the for each decoder pass and stacking the logits.
        logits_list = []

        htilde_t = self.decoder.get_first_hidden_state(h, F_lens)

        if self.cell_type == 'lstm':
            cell_state = torch.zeros_like(htilde_t)
            htilde_t = (htilde_t, cell_state)

        for t in range(E.size(0) - 1):
            E_tm1 = E[t]
            logits_t, htilde_t = self.decoder(E_tm1, htilde_t, h, F_lens)
            logits_list.append(logits_t)

        logits = torch.stack(logits_list, dim=0)

        return logits
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        assert False, 'Fill me'

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:

        # Updating the beam by calculating all probabilities
        V = logpy_t.shape[2]  # logpy_t is of shape [M, K, V]
        M = logpy_t.shape[0]
        K = self.beam_width
        H = self.encoder.hidden_state_size

        # Computing all possible log probabilities for next tokens
        all_probs = (logpb_tm1.unsqueeze(2) + logpy_t).view(M, -1)

        # Retrieving the top K combined log probabilities and their indices
        logpb_t, indices = torch.topk(all_probs, K, dim=1)

        beam_indices = indices // V
        token_indices = indices % V

        # Updating the beam sequences
        # b_tm1_1 is [t, M, K]; we need to append new tokens to increase sequence length
        token_indices = token_indices.unsqueeze(0)  # [1, M, K]
        b_t_1 = torch.cat([b_tm1_1, token_indices], dim=0)  # [t + 1, M, K]

        # Updating the hidden states
        if isinstance(htilde_t, tuple):
            hidden_state, cell_state = htilde_t
            beam_indices = beam_indices.unsqueeze(-1).expand(-1, -1, 2 * H)
            b_t_0_hidden = hidden_state.gather(1, beam_indices)
            b_t_0_cell = cell_state.gather(1, beam_indices)
            b_t_0 = (b_t_0_hidden, b_t_0_cell)
        else:
            beam_indices = beam_indices.unsqueeze(-1).expand(-1, -1, 2 * H)
            b_t_0 = htilde_t.gather(1, beam_indices)

        return b_t_0, b_t_1, logpb_t

        '''
        if(isinstance(htilde_t, tuple)):
            htilde_t = htilde_t[0]

        V = logpy_t.shape[2] 
        M = logpy_t.shape[0] 
        K = self.beam_width
        H = htilde_t.shape[-1] 

        # Compute all possible log probabilities for next tokens across the vocabulary
        all_probs = (logpb_tm1.unsqueeze(2) + logpy_t).view(M, -1)  # [M, K*V]

        # Retrieve the top K combined log probabilities and their indices
        logpb_t, indices = torch.topk(all_probs, K, dim=1)

        # Compute new beam indices and token indices
        beam_indices = indices // V 
        token_indices = indices % V  

        # Adjust the dimensions of b_tm1_1 if necessary
        if b_tm1_1.dim() < 3:
            b_tm1_1 = b_tm1_1.expand(-1, M, K)
        # Make sure beam_indices is correctly shaped for gathering
        beam_indices = beam_indices.unsqueeze(0).expand(b_tm1_1.size(0), -1, -1)
        b_tm1_1 = torch.gather(b_tm1_1, 2, beam_indices)  # Gather along the beam dimension

        new_tokens = token_indices.unsqueeze(0)  # Add new sequence length dimension
        b_t_1 = torch.cat([b_tm1_1, new_tokens], dim=0)

        expanded_indices = beam_indices[0, :, :]  # Use only the first time step indices for hidden state update
        b_t_0 = htilde_t.gather(1, expanded_indices.unsqueeze(-1).expand(-1, -1, H))

        return b_t_0, b_t_1, logpb_t
        '''
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]


# cd "/content/drive/My Drive/comp542-assignment2"

# Model without attention train and test commands
# python3 a2_run.py train 'wmt16_en_tr/train.json' 'wmt16_en_tr/valid.json' vocab.e.gz vocab.f.gz model_wo_att.pt.gz --device cuda
# python3 a2_run.py test 'wmt16_en_tr/test.json' vocab.e.gz vocab.f.gz rnn_model_wo_att.pt.gz --device cuda --cell-type rnn

# Model with attention

# python3 a2_run.py train 'wmt16_en_tr/train.json' 'wmt16_en_tr/valid.json' vocab.e.gz vocab.f.gz lstm_dot_model_w_att.pt.gz --with-attention --device cuda --batch-size 64
# python3 a2_run.py test 'wmt16_en_tr/test.json' vocab.e.gz vocab.f.gz lstm_model_w_att.pt.gz --with-attention --device cuda

# Model with multihead attention
# python3 a2_run.py train 'wmt16_en_tr/train.json' 'wmt16_en_tr/valid.json' vocab.e.gz vocab.f.gz rnn_model_w_mhatt.pt.gz --with-multihead-attention --device cuda --batch-size 32 --cell-type rnn
# python3 a2_run.py test 'wmt16_en_tr/test.json' vocab.e.gz vocab.f.gz lstm_model_w_mhatt.pt.gz --with-multihead-attention --device cuda
