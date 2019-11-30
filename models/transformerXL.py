import torch
import torch.nn as nn

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
    mem_len = 0
    multi_gpu = True
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
        n_layer = (self.depth - 1) / 2 # depth = n_layer * (multi-head + ffn) + linear-softmax
        d_model = self.width
        d_inner = self.width * 2

        if self.d_embed < 0:
            self.d_embed = d_model

        if self.restart:
            with open(os.path.join(restart_dir, 'model.pt'), 'rb') as f:
                model = torch.load(f)
            if not fp16:
                model = model.float()
            model.apply(self.update_dropout)
            model.apply(self.update_dropatt)
        else:
            model = MemTransformerLM(self.vocab_size, n_layer, self.n_head, d_model,
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
            if args.gpu0_bsz >= 0:
                self.para_model = BalancedDataParallel(self.gpu0_bsz, self.model, dim=1).to(self.device)
            else:
                self.para_model = nn.DataParallel(self.model, dim=1).to(self.device)
        else:
            self.para_model = self.model.to(self.device)

        return model


    def predict(self, batch, padding=True):
        """
        Gets all one-step predictions for a batch of sentences.
        For each context window output the next word.
        seq[0:k] -> pred[k+1]
        and so we output seq_len - k predictions
        without padding
        and seq_len predictions
        with padding
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
        total_len, total_loss = 0, 0.
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                    break
                ret = self.model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        self.model.reset_length(self.tgt_len, self.ext_len, self.mem_len)
        self.model.train()

        return total_loss / total_len
        


    def train_step(self, inputs, targets, mems=0, train_step=0):
        """
        Performs an unsupervised train step for a given batch.
        Returns loss on batch.
        """

        # Zero out model gradients
        self.model.zero_grad()

        # Calculate loss
        ret = self.para_model(inputs, targets, *mems)
        loss, mems = ret[0], ret[1:]
        print(inputs.shape, loss.shape)
        loss = loss.float().mean().type_as(loss)
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()
        train_loss += loss.float().item()

        # Gradient clipping
        if self.clip is not None:
            if self.fp16:
                self.optimizer.clip_master_grads(self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)

        self.optimizer.step()

        # Step-wise learning rate annealing
        if self.scheduler_type in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < self.warmup_step:
                curr_lr = self.lr * train_step / self.warmup_step
                self.optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if self.scheduler_type == 'cosine':
                    self.scheduler.step(train_step)
        elif self.scheduler_type == 'inv_sqrt':
            self.scheduler.step(train_step)





