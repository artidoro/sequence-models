import torch
import torch.nn as nn



def TransformerXL(SequenceModel):

    def __init__(depth, width):
        hyperparams = {
            
        }
        super().__init__(depth, width, hyperparams)


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


    def init_model(self, depth, width):
        ### TODO calculate model layers/head/dimensions from depth/width:
        self.n_layer = depth
        self.n_head = 10
        self.d_model = 500
        self.d_inner = 1000
        self.d_head = 50
        self.d_embed = -1
        self.dropout = 0.0
        self.dropatt = 0.0
        self.div_val = 1
        self.tied = True
        self.pre_lnorm = False

        self.tgt_len = 70
        self.eval_tgt_len = 50
        self.ext_len = 0
        self.mem_len = 0
        self.same_length = False
        self.attn_type = 0
        self.clamp_len = -1

        self.ntokens = int(1000) # TODO: FIX THIS

        if self.restart:
            with open(os.path.join(self.restart_dir, 'model.pt'), 'rb') as f:
                model = torch.load(f)
            if not self.fp16:
                model = model.float()
            model.apply(self.update_dropout)
            model.apply(self.update_dropatt)
        else:
            model = MemTransformerLM(self.ntokens, self.n_layer, self.n_head, self.d_model,
                self.d_head, self.d_inner, self.dropout, self.dropatt,
                tie_weight=self.tied, d_embed=self.d_embed, div_val=self.div_val, 
                pre_lnorm=self.pre_lnorm, tgt_len=self.tgt_len,
                ext_len=self.ext_len, mem_len=self.mem_len,
                same_length=self.same_length, attn_type=self.attn_type,
                clamp_len=self.clamp_len)
            model.apply(weights_init)
            model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

        self.model = model
        self.n_all_param = sum([p.nelement() for p in model.parameters()])
        self.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
        return model
