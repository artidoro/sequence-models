import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequence_model import SequenceModel
from models.mem_transformer import MemTransformerLM
from models.utils.data_parallel import BalancedDataParallel


class TransformerXL(SequenceModel):

    # Default parameter values
    attn_type = 0
    gpu0_bsz = -1
    clamp_len = -1
    clip = 0.25
    div_val = 1
    d_embed = -1
    d_head = 50
    dropatt = 0.0
    dropout = 0.0
    eval_tgt_len = 50
    ext_len = 0
    fp16 = False
    mem_len = 0
    multi_gpu = False
    n_head = 10
    param_init = 'normal'
    param_init_range = 0.1
    param_init_std = 0.02
    pre_lnorm = False
    proj_init_std = 0.01
    restart = False
    same_length = False
    tgt_len = 70
    tied = True


    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)


    def init_weight(self, weight):
        if self.param_init == 'uniform':
            nn.init.uniform_(weight, -self.param_init_range, self.param_init_range)
        elif self.param_init == 'normal':
            nn.init.normal_(weight, 0.0, self.param_init_std)


    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('BalancedDataParallelEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                self.init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                self.init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.param_init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)


    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = self.dropout


    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = self.dropatt


    def init_model(self):
        n_layer = int((self.depth - 1) / 2) # depth = n_layer * (multi-head + ffn) + linear-softmax
        d_model = self.width
        d_inner = self.width * 2
        vocab_size = self.vocab

        if self.d_embed < 0:
            self.d_embed = d_model

        # Mixed-floating point precision (if fp16 is enabled, storage will be with half-precision)
        if self.fp16 and 'cuda' not in self.device:
            print('WARNING: fp16 requires cuda, ignoring fp16 option')
            self.fp16 = False
        elif self.fp16:
            try:
                from apex.fp16_utils import FP16_Optimizer
                self.optimizer = FP16_Optimizer(self.optimizer,
                    static_loss_scale=args.static_loss_scale,
                    dynamic_loss_scale=args.dynamic_loss_scale,
                    dynamic_loss_args={'init_scale': 2 ** 16})
            except:
                print('WARNING: apex not installed, ignoring fp16 option')
                self.fp16 = False


        if self.restart:
            with open(os.path.join(restart_dir, 'model.pt'), 'rb') as f:
                model = torch.load(f)
            if not fp16:
                model = model.float()
            model.apply(self.update_dropout)
            model.apply(self.update_dropatt)
        else:
            model = MemTransformerLM(vocab_size, n_layer, self.n_head, d_model,
                self.d_head, d_inner, self.dropout, self.dropatt,
                tie_weight=self.tied, d_embed=self.d_embed, div_val=self.div_val, 
                tie_projs=[False], pre_lnorm=self.pre_lnorm, tgt_len=self.tgt_len,
                ext_len=self.ext_len, mem_len=self.mem_len, cutoffs=[],
                same_length=self.same_length, attn_type=self.attn_type,
                clamp_len=self.clamp_len, sample_softmax=-1)
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

        self.model = model
        self.n_all_param = sum([p.nelement() for p in model.parameters()])
        self.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

        if self.multi_gpu:
            self.model = self.model.to(self.device)
            if self.gpu0_bsz >= 0:
                self.para_model = BalancedDataParallel(self.gpu0_bsz, self.model, dim=1).to(self.device)
            else:
                self.para_model = nn.DataParallel(self.model, dim=1).to(self.device)
        else:
            self.para_model = self.model.to(self.device)

        return model


    def predict(self, inputs):
        """
        Gets predictions for the next token of a batch of sequences (as a distribution over vocab tokens).
        
        Arguments:
            inputs : a Tensor of shape (batch_size, input_seq_length)

        Returns:
            probs : a Tensor of shape (batch_size, vocab_size)
        """

        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        if self.mem_len == 0:
            self.model.reset_length(self.eval_tgt_len, 
                self.ext_len + self.tgt_len - self.eval_tgt_len, self.mem_len)
        else:
            self.model.reset_length(self.eval_tgt_len,
                self.ext_len, self.mem_len + self.tgt_len - self.eval_tgt_len)

        # Evaluation
        with torch.no_grad():
            # Transpose data, since MemTransformerLM expects batches in each column
            inputs = inputs.t()

            # Get logits
            mems = tuple()
            ret = self.model.forward_generate(inputs, *mems)
            logits, mems = ret[0], ret[1:]
            logits = logits[-1] # Only keep logits from the last step

            probs = F.softmax(logits, dim=-1)

        # Switch back to the training mode
        self.model.reset_length(self.tgt_len, self.ext_len, self.mem_len)
        self.model.train()

        return probs        


    def train_step(self, inputs, targets, mems=tuple(), train_step=0):
        """
        Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """

        # Zero out model gradients
        self.model.zero_grad()

        # Transpose data, since MemTransformerLM expects batches in each column
        inputs, targets = inputs.t(), targets.t()

        # Calculate loss
        ret = self.para_model(inputs, targets, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        # Gradient clipping
        if self.clip is not None:
            if self.fp16:
                self.optimizer.clip_master_grads(self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        # Update scheduler
        self.update_scheduler(train_step)

        return loss


