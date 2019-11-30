import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel



### Model Experiment Class 

class SequenceModelExperiment:

    DEFAULT_PARAMS = {
        'optimizer_type': 'adam',
        'scheduler_type': 'constant',
        'lr': 0.0002,
        'lr_min': 0,
    }

    def __init__(self, hyperparams):
        # Set default hyperparameter values
        for k, v in self.DEFAULT_PARAMS.items():
            setattr(self, k, v)

        # Set given hyperparameter values
        for k, v in hyperparams.items():
            setattr(self, k, v)

        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.init_scheduler()
        self.init_optimizer()
        self.model = self.init_model()

        if self.fp16:
            self.model = self.model.half()

        if self.multi_gpu:
            self.model = self.model.to(device)
            if self.gpu0_bsz >= 0:
                self.para_model = BalancedDataParallel(self.gpu0_bsz, self.model, dim=1).to(device)
            else:
                self.para_model = nn.DataParallel(self.model, dim=1).to(device)
        else:
            self.para_model = self.model.to(device)


    def init_optimizer():
        if self.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.mom)
        elif self.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == 'adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr=self.lr)

        return self.optimizer()


    def init_scheduler(self):
        if self.scheduler_type == 'cosine':
            eta_min = self.eta_min if self.eta_min is not None else 0
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_step, eta_min=eta_min)
        elif self.scheduler_type == 'inv_sqrt':
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and self.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > self.warmup_step \
                           else step / (self.warmup_step ** 1.5)
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.scheduler_type == 'dev_perf':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_rate, patience=self.patience, min_lr=self.lr_min)
        elif self.scheduler_type == 'constant':
            pass

    def init_model(depth, width):
        ### Each subclass should implement this
        pass


    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        if self.mem_len == 0:
            model.reset_length(self.eval_tgt_len, self.ext_len+self.tgt_len-self.eval_tgt_len, self.mem_len)
        else:
            model.reset_length(self.eval_tgt_len, self.ext_len, self.mem_len+self.tgt_len-self.eval_tgt_len)

        # Evaluation
        total_len, total_loss = 0, 0.
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if self.max_eval_steps > 0 and i >= self.max_eval_steps:
                    break
                ret = model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        model.reset_length(self.tgt_len, self.ext_len, self.mem_len)
        model.train()

        return total_loss / total_len


    def train_batch_loop(self, batch, (data, target, *mems)):
        self.model.zero_grad()

        ret = self.para_model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        if self.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()
        self.train_loss += loss.float().item()

        # Gradient clipping
        if self.clip is not None:
            if self.fp16:
                self.optimizer.clip_master_grads(self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        # Step-wise learning rate annealing
        train_step += 1
        if self.scheduler_type in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < self.warmup_step:
                curr_lr = self.lr * self.train_step / self.warmup_step
                self.optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if scheduler_type.scheduler == 'cosine':
                    self.scheduler.step(train_step)
        elif self.scheduler_type == 'inv_sqrt':
            self.scheduler.step(train_step)

        # Logging
        if self.train_step % self.log_interval == 0:
            cur_loss = self.train_loss / self.log_interval
            elapsed = time.time() - self.log_start_time

            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                    epoch, train_step, batch+1, self.optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / self.log_interval, cur_loss)
            perplexity = math.exp(cur_loss) if self.log_ppl else cur_loss
            log_str += ' | perplexity {:9.3f}'.format(perplexity)
            print(log_str)
                
            self.train_loss = 0
            self.log_start_time = time.time()


        # Evaluation
        if self.train_step % self.eval_interval == 0:
            val_loss = evaluate(self.va_iter)
            print('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| val loss {:5.2f}'.format(
                self.train_step // self.eval_interval, self.train_step,
                (time.time() - self.eval_start_time), val_loss)
            perplexity = math.exp(val_loss) if self.log_ppl else val_loss
            log_str += ' | val perplexity {:9.3f}'.format(val_loss)
            print(log_str)
            print('-' * 100)

            # Save the model if the validation loss is the best we've seen so far.
            if not self.best_val_loss or val_loss < self.best_val_loss:
                if not self.debug:
                    with open(os.path.join(self.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(self.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                self.best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if self.scheduler_type == 'dev_perf':
                self.scheduler.step(val_loss)

            self.eval_start_time = time.time()

            if self.train_step == self.max_step:
                break


    def train_epoch(self):
        # Turn on training mode which enables dropout.
        global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
        model.train()
        mems = tuple()
        train_iter = self.tr_iter.get_varlen_iter() if self.varlen else self.tr_iter

        for batch, (data, target, seq_len) in enumerate(train_iter):
            self.train_batch_loop(batch, (data, target, *mems))


    def train(self):
        # Loop over epochs.
        self.train_step = 0
        self.train_loss = 0
        self.best_val_loss = None

        self.log_start_time = time.time()
        self.eval_start_time = time.time()

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in itertools.count(start=1):
                train()
                if self.train_step == self.max_step:
                    print('-' * 100)
                    print('End of training')
        except KeyboardInterrupt:
            print('-' * 100)
            print('Exiting from training early')


        # Load the best saved model.
        with open(os.path.join(self.work_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        para_model = self.model.to(device)

        # Run on test data.
        test_loss = evaluate(te_iter)
        print('=' * 100)
        print('| End of training | test loss {:5.2f} | test perplexity {:9.3f}'.format(test_loss, math.exp(test_loss)))
        print('=' * 100)






class TransformerXL(SequenceModelExperiment):

    self.max_step = 100000
    self.clip = 0.25

    self.n_layer = 12
    self.n_head = 10
    self.d_head = 50
    self.d_embed = -1
    self.d_model = 500
    self.d_inner = 1000
    self.


    def __init__(self, hyperparams):
        super().__init__(hyperparams)


    def init_weight(weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)


    def init_bias(bias):
        nn.init.constant_(bias, 0.0)


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)


    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = self.dropout


    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = self.dropatt


    if self.restart:
        with open(os.path.join(self.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        if not self.fp16:
            model = model.float()
        model.apply(self.update_dropout)
        model.apply(self.update_dropatt)
    else:
        model = MemTransformerLM(ntokens, self.n_layer, self.n_head, self.d_model,
            self.d_head, self.d_inner, self.dropout, self.dropatt,
            tie_weight=self.tied, d_embed=self.d_embed, div_val=self.div_val,
            tie_projs=tie_projs, pre_lnorm=self.pre_lnorm, tgt_len=self.tgt_len,
            ext_len=self.ext_len, mem_len=self.mem_len, cutoffs=cutoffs,
            same_length=self.same_length, attn_type=self.attn_type,
            clamp_len=self.clamp_len)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

    self.model = model
    self.n_all_param = sum([p.nelement() for p in model.parameters()])
    self.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])


