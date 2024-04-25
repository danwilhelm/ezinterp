'''
ezinterp.py
by Dan Wilhelm [dan@danwil.com]

Minimalistic NumPy-based small/medium transformer interpretability library for interactive exploration.

Easily:
- Access all intermediate model values.
- Load, observe, and modify weights.
- Selectively disable or re-route nearly every aspect of the model.
- Compute special values useful for interpretability.

EZModel stores the model weights, and EZRun stores the intermediate values of a model run.

To view available options, introspect one of the objects.
'''

import copy
import numpy as np

class EZModel:
    '''Store/load model weights.'''

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True))
    
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def gelu_fast(x):
        return 0.5 * x * (1.0 + np.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

    @staticmethod
    def zscores(x, eps=1e-5):
        centered = x - np.mean(x, axis=-1, keepdims=True)
        return centered / np.sqrt(np.mean(np.square(centered), axis=-1, keepdims=True) + eps)

    def __init__(self, state_dict=None, attn_normalize=True, mlp_normalize=True, final_normalize=True):
        self.run_defaults = {
            'use_pos_embeds': True,             # Add pos_embeds to embeds
            'use_attn': True,                   # Use attn block
            'use_mlp': True,                    # Use mlp block
            'use_attn_bias': True,              # Add final output bias (bo)
            'use_mlp_actfn': True,              # Apply activation function to mlp intermediate neuron outputs
            'use_attn_skip': True,              # Add attn_in/block_in to attn_out
            'use_mlp_skip': True,               # Add mlp_in to mlp_out
            'attn_normalize': attn_normalize,   # Normalize (zscores) attn input
            'attn_norm': attn_normalize,        # Project attn input
            'mlp_normalize': mlp_normalize,     # Normalize (zscores) mlp input
            'mlp_norm': mlp_normalize,          # Project mlp input
            'final_normalize': final_normalize, # Normalize (zscores) output
            'final_norm': final_normalize,      # Project output
        }
        
        if state_dict is not None:
            self.load_pt(state_dict)


    def __repr__(self):
        return f'<EZModel n_blocks={self.n_blocks} n_heads={self.n_heads} k={self.k} k_head={self.k_head} n_mlp={self.n_mlp} | n_vocab={self.n_vocab} seq_len={self.seq_len} n_classes={self.n_classes}>\n' +\
               f'- EMBEDS: embeds, pos_embeds, wu/bu (cols)\n' +\
               f'- NORMS: wn1/bn1 (pre-attn), wn2/bn2 (pre-mlp), wnf/bnf (final)\n' +\
               f'- ATTENTION: wq/bq, wk/bk, wv/bv, wo/bo, attn_mask, attn_ignore\n' +\
               f'- MLP: wm/bm (in projection), wp/bp (out projection)\n' +\
               f'- INTERP: wvo (wv @ wo), wqk (wq @ wk.T), wmp (wm @ wp)'
    
    
    def run(self, toks_in=None, **kwargs):
        'For options, see ``'
        return EZRun(self, toks_in, **kwargs)


    def decomp_attn(self, normed, block_num, head_num):
        '''
        Returns the total attention decomposition (main, column biases, row biases, const bias),
          with shapes (n_toks,n_toks), (1,n_toks), (n_toks,1), and (1,1),
          applied to an (n_toks,k) input (typically normed).
        - Note the row/const biases do not affect the attention softmax.
        - The sum of the decomposition is equivalent to attention: (normed @ wq + bq) @ (normed @ wk + bk).T
        '''
        main = normed @ (self.wqk[block_num,head_num]) @ normed.T                           # (n_toks,kh) @ (k,kh) @ (kh,k) @ (k,n_toks) = (n_toks,n_toks), where (kh = k_head)
        colsb = (self.bq[block_num,head_num] @ self.wk[block_num,head_num].T) @ normed.T    # (1,kh) @ (kh,k) @ (k,n_toks) = (1,n_toks)
        rowsb = normed @ self.bqk[block_num,head_num]                                       # (n_toks,k) @ (k,kh) @ (kh,1) = (n_toks,1)   <- DOESN'T AFFECT SOFTMAX
        constb = self.bq[block_num,head_num] @ self.bk[block_num,head_num].T                # (1,k) @ (k,1) = (1,1)                       <- DOESN'T AFFECT SOFTMAX
        
        return main, colsb, rowsb, constb
    

    def set_params_from_weights(self):
        self.n_blocks = len(self.wq)
        self.n_vocab, self.k = self.embeds.shape
        
        self.seq_len = len(self.pos_embeds)
        self.n_classes = len(self.bu)
        self.n_mlp = self.wm.shape[-1]               # Number of intermediate neurons
        
        self.n_heads = len(self.wv[0]) if self.n_blocks > 0 else 0
        self.k_head = self.k // self.n_heads if self.n_blocks > 0 else 0    
        
        # Useful for interpretability
        self.wvo = self.wv @ self.wo                       # (n_blocks, n_heads, k, k)
        self.wqk = self.wq @ self.wk.transpose([0,1,3,2])  # (n_blocks, n_heads, k, k)
        self.wmp = self.wm @ self.wp                       # (n_blocks, k, k)  MLP w/o activation
        
        self.bqk = self.wq @ self.bk[...,np.newaxis]       # (n_blocks,n_heads, k,k_head) @ (n_blocks,n_heads, k_head,1) = (n_blocks,n_heads, k,1)


    def to_tflens(self, pt_model=None, device='cpu'):
        '''
        Returns a TransformerLens model.
        - pt_model: Either a PyTorch filename -or- PyTorch object.
        '''
        import torch as t
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            n_layers=self.n_blocks,
            d_model=self.k,
            d_head=self.k // self.n_heads,
            n_heads=self.n_heads,
            d_mlp=self.n_mlp,
            d_vocab=self.n_vocab,
            n_ctx=self.seq_len,
            act_fn="gelu",
            normalization_type="LNPre",
        )
        
        tl_model = HookedTransformer(cfg)
        if pt_model is str:
            tl_model.load_state_dict(t.load(pt_model))
        else:
            tl_model.load_state_dict(pt_model)
        
        tl_model.to(device)

        return tl_model


class EZRun:
    '''Run a model and retain intermediate results.'''

    SAVED_FIELDS = [
        'block_in', 'attn_in_normed', 'attn_out', 'mlp_in', 'mlp_in_normed', 'mlp_out',
        'block_out', 'attn_q', 'attn_k', 'attn_v', 'qk_out', 'masked_out', 'softmax_out', 
        'heads_out', 'mlp_interm', 'mlp_actfn', 'logits', 'probs'
    ]
    
    def __init__(self, model, toks_in=None, **kwargs):
        self.model = model
        self.n_toks = 0
        self.prompt = ''
        
        if toks_in is not None:
            self.run(toks_in, **kwargs)

    
    def __repr__(self):
        return f'<EZRun n_toks={self.n_toks} prompt={self.prompt[:50]}{"..." if len(self.prompt) > 50 else ""}>\n' +\
               f'- IN: prompt, in_labels, in_toks, in_embeds, in_pos_embeds\n' +\
               f'- OUT: out_labels, out_toks, out (normed), logits, probs\n' +\
               f'- RESIDUAL STREAM: block_in, attn_in_normed, attn_out, mlp_in, mlp_in_normed, mlp_out, block_out\n' +\
               f'- ATTN HEADS: attn_q, attn_k, attn_v, qk_out, masked_out, softmax_out, heads_out\n' +\
               f'- MLP: mlp_interm, mlp_actfn\n'
               
    
    def alloc_outputs(self, m, n_blocks=None):
        '''Allocate memory for intermediate results.'''

        n_blocks = m.n_blocks if n_blocks is None else n_blocks

        ### RESIDUAL STREAM VALUES
        self.block_in = np.zeros((n_blocks, self.n_toks, m.k))
        self.attn_in_normed = np.zeros_like(self.block_in)  # attn_in identical to block_in
        self.attn_out = np.zeros_like(self.block_in)
        
        self.mlp_in = np.zeros_like(self.block_in)
        self.mlp_in_normed = np.zeros_like(self.block_in)
        self.mlp_out = np.zeros_like(self.block_in)
        
        self.block_out = np.zeros_like(self.block_in)
        
        ### ATTENTION QUERY/KEY/VALUE TRANSFORMS
        self.attn_q = np.zeros((n_blocks, m.n_heads, self.n_toks, m.k_head))  # each head's initial affine transform
        self.attn_k = np.zeros_like(self.attn_q)
        self.attn_v = np.zeros_like(self.attn_q)
        
        # ATTENTION QK HEADS, ATTN GRID
        self.qk_out = np.zeros((n_blocks, m.n_heads, self.n_toks, self.n_toks))  # each head's attn_q @ attn_k.T / sqrt(k_head)
        self.masked_out = np.zeros_like(self.qk_out)
        self.softmax_out = np.zeros_like(self.qk_out)        
        self.heads_out = np.zeros((n_blocks, m.n_heads, self.n_toks, m.k))    # each head following the output linear transform

        # MLP
        self.mlp_interm = np.zeros((n_blocks, self.n_toks, m.n_mlp))
        self.mlp_actfn = np.zeros_like(self.mlp_interm)
    

    def dealloc_outputs(self, keep=None):
        '''
        Deallocate some intermediate results to save memory.
        - keep: If `None`, all results retained. Otherwise, a list of fields to retain from `self.SAVED_FIELDS`.
        '''
        if keep is None:
            return

        keep = set(keep)
        for field in self.SAVED_FIELDS:
            if field in self.__dict__ and field not in keep:
                del self.__dict__[field]
    
    
    def run(self, in_toks, in_embeds=None, blocks=None, keep=None, **kwargs):
        '''
        Run the model on a prompt (string) or list of input token indexes.
        - in_toks: Input string -or- iterable of token indexes (which directly index into the embeds)
        - in_embeds: (n_toks, k)-dimensional ndarray that directly specifies the input residual stream (ignores `in_toks`)
        - blocks: Iterable of block indexes to run in order (default: `range(n_blocks)`)
                  This can be used with `in_embeds` to incrementally run a model.
        - keep: To save memory, indicates which fields to retain from `self.SAVED_FIELDS` (all others will be deallocated)

        REPLACEMENT WEIGHT ARGS:
        - Specify any Model attribute to use it instead of the original (e.g. `run(..., bo=np.zeros(model.k))`)

        BOOLEAN ARGS:
        - use_pos_embeds: Add `pos_embeds` to token embeddings
        - use_attn: Run attention block
        - use_mlp: Run mlp block
        - use_attn_bias: Add final attention output bias `bo`
        - use_mlp_actfn: Apply activation function to the intermediate neuron outputs
        - use_attn_skip: `mlp_in` = `attn_out` + `attn_in` (skip connection)
        - use_mlp_skip: `block_out` = `mlp_out` + `mlp_in` (skip connection)
        - attn_normalize/mlp_normalize/final_normalize: Normalize (zscore) the input
        - attn_norm/mlp_norm/final_norm: Project the input plus bias
        '''

        # Replace some model attributes for this run only, if specified
        m = copy.copy(self.model)
        m.__dict__.update({k:v for k,v in kwargs.items() if k in m.__dict__})

        # Replace some Boolean for this run only, if specified
        p = m.run_defaults.copy()
        p.update({k:v for k,v in kwargs.items() if k in p})
        
        blocks = range(m.n_blocks) if blocks is None else blocks
        
        if in_embeds is None:
            self.in_labels = np.array(list(in_toks)) if type(in_toks) is str else m.decoder(in_toks)
            self.prompt = ''.join(self.in_labels)
            self.in_toks = m.encoder(in_toks) if type(in_toks) is str else np.array(in_toks)
            self.n_toks = len(self.in_toks)
            self.in_embeds = m.embeds[self.in_toks]
            self.in_pos_embeds = m.pos_embeds[:self.n_toks] if p['use_pos_embeds'] else np.zeros_like(self.embeds_in)
            self.stream_in = self.in_embeds + self.in_pos_embeds
        else:
            self.in_labels = np.array([])
            self.n_toks = len(in_embeds)
            self.in_embeds = in_embeds
            self.stream_in = in_embeds   # Directly specify the (n_toks,k)-dimensional input
        
        # Pre-allocate space for results
        self.alloc_outputs(m, n_blocks=len(blocks))
        
        # For each attn/MLP block ...
        block_out = self.stream_in
        for i,b in enumerate(blocks):
            self.block_in[i] = block_out
            attn_in = block_out
            
            ### ATTENTION           
            if p['use_attn']:
                self.attn_in_normed[i] = m.zscores(attn_in) if p['attn_normalize'] else attn_in
                self.attn_in_normed[i] = (self.attn_in_normed[i] @ m.wn1[b] + m.bn1[b]) if p['attn_norm'] else self.attn_in_normed[i]

                self.attn_q[i] = self.attn_in_normed[i] @ m.wq[b] + m.bq[b][:,np.newaxis]    # (n_toks,k) @ (n_heads,k,k_head) + (n_heads,1,k_head)
                self.attn_k[i] = self.attn_in_normed[i] @ m.wk[b] + m.bk[b][:,np.newaxis]    #   => (n_heads,n_toks,k_head)
                self.attn_v[i] = self.attn_in_normed[i] @ m.wv[b] + m.bv[b][:,np.newaxis]    # 
                
                self.qk_out[i] = self.attn_q[i] @ self.attn_k[i].transpose([0,2,1]) / np.sqrt(m.k_head)  # (n_heads, n_toks, n_toks)
                self.masked_out[i] = np.where(m.attn_mask[b, :self.n_toks, :self.n_toks],
                                              self.qk_out[i], m.attn_ignore[b])
                self.softmax_out[i] = m.softmax(self.masked_out[i])
                
                # (n_heads, n_toks, n_toks) @ (n_heads, n_toks, k_head)  @ (n_heads, k_head, k) => (n_heads, n_toks, k)
                self.heads_out[i] = self.softmax_out[i] @ self.attn_v[i] @ m.wo[b]
                
                self.attn_out[i] = np.sum(self.heads_out[i], axis=0) + (m.bo[b] if p['use_attn_bias'] else 0.0)     # (n_toks, k)
            
            ### MLP
            mlp_in = self.attn_out[i] + (attn_in if p['use_attn_skip'] else 0.0)   # Skip connection
            self.mlp_in[i] = mlp_in.copy()

            if p['use_mlp']:
                self.mlp_in_normed[i] = m.zscores(mlp_in) if p['mlp_normalize'] else mlp_in
                self.mlp_in_normed[i] = (self.mlp_in_normed[i] @ m.wn2[b] + m.bn2[b]) if p['mlp_norm'] else self.mlp_in_normed[i]

                self.mlp_interm[i] = self.mlp_in_normed[i] @ m.wm[b] + m.bm[b]
                self.mlp_actfn[i] = m.gelu(self.mlp_interm[i]) if p['use_mlp_actfn'] else self.mlp_interm[i]
                self.mlp_out[i] = self.mlp_actfn[i] @ m.wp[b] + m.bp[b]
            
            block_out = self.mlp_out[i] + (mlp_in if p['use_mlp_skip'] else 0.0)    # Skip connection
            self.block_out[i] = block_out
        
        block_out = m.zscores(block_out) if p['final_normalize'] else block_out
        block_out = (block_out @ m.wnf + m.bnf) if p['final_norm'] else block_out
        
        # UNEMBED
        self.out = block_out
        self.logits = block_out @ m.wu + m.bu      # (n_toks,k) @ (k,n_classes) + (n_classes,1) => (n_toks,n_classes)
        self.probs = m.softmax(self.logits)
        self.out_labels = np.argmax(self.probs, axis=1)
        self.out_toks = m.decoder(self.out_labels)
        
        self.attn_in = self.block_in
        self.dealloc_outputs(keep)
        
        return self.probs


EZModel.run.__doc__ = EZRun.run.__doc__
