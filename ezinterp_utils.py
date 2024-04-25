'''
ezinterp_utils.py

Utility functions for ezinterp.
'''

import torch as t
import numpy as np

from ezinterp import EZModel, EZRun


def pt_to_ez(pt_model, token_to_int=None):
    '''
    Return an EZModel, given a PyTorch model.
    - pt_model: Either a PyTorch filename -or- PyTorch object.
    - token_to_int: A dict with tokens as keys and embedding indexes as values.
                    Strongly recommended (required to use a string prompt).
    '''

    block_ezlabel_cond_ptlabel = {
        # ezlabel: (conditional, PyTorch label)
        'wn1': ('attn_norm', 'ln1.w'),   # (n_blocks, k)
        'bn1': ('attn_norm', 'bn1.w'),
        'wn2': ('mlp_norm', 'ln2.w'),
        'bn2': ('mlp_norm', 'bn2.w'),

        'wq': ('use_attn', 'attn.W_Q'),  # Q,K,V = (n_blocks, n_heads, k, k_head)
        'bq': ('use_attn', 'attn.b_Q'),  # bq,bk,bv = (n_blocks, n_heads, k_head)
        'wk': ('use_attn', 'attn.W_K'),
        'bk': ('use_attn', 'attn.b_K'),
        'wv': ('use_attn', 'attn.W_V'),
        'bv': ('use_attn', 'attn.b_V'),

        'wo': ('use_attn', 'attn.W_O'),  # O = (n_blocks, n_heads, k_head, k)
        'bo': ('use_attn', 'attn.b_O'),  # bo = (n_blocks, k)
        
        # [[True, False, ...], [True, True, False, ...], ..., [True, ...]]
        'attn_mask': ('use_attn', 'attn.mask'),      # (n_blocks, seq_len, seq_len)
        'attn_ignore': ('use_attn', 'attn.IGNORE'),  # (n_blocks,) = -inf

        'wm': ('use_mlp', 'mlp.W_in'),   # mlp in projection = (n_blocks, k, n_mlp)
        'bm': ('use_mlp', 'mlp.b_in'),   # bm = (n_blocks, n_mlp)
        'wp': ('use_mlp', 'mlp.W_out'),  # mlp out projection = (n_blocks, n_mlp, k)
        'bp': ('use_mlp', 'mlp.b_out'),  # bp = (n_blocks, k)
    }
    
    if type(pt_model) is str:
        pt_model = t.load(pt_model)
    
    m = EZModel()

    if token_to_int is not None:
        if type(token_to_int) is list or type(token_to_int) is np.array:
            token_to_int = {tok: i for i,tok in enumerate(token_to_int)}
        
        m.token_to_int = token_to_int
        m.int_to_token = {i: tok for i,tok in enumerate(token_to_int)}
        
        m.encode = lambda c: m.token_to_int[c]   # single token -> int
        m.encoder = lambda chars: np.array([m.token_to_int[c] for c in chars])  # iterable of tokens -> ints
        m.decode = lambda tok: m.int_to_token[tok]
        m.decoder = lambda toks: np.array([m.int_to_token[tok] for tok in toks])
        m.vocab = np.array([tok for tok,i in sorted(m.token_to_int.items(), key=lambda x: x[1])])
    
    m.embeds = pt_model['embed.W_E'].cpu().numpy()
    m.pos_embeds = pt_model['embed.W_E'].cpu().numpy()
    m.wu = pt_model['unembed.W_U'].cpu().numpy() if 'unembed.W_U' in pt_model else m.embeds
    m.bu = pt_model['unembed.b_U'].cpu().numpy() if 'unembed.b_U' in pt_model else np.zeros(m.wu.shape[-1])
    
    # Assume if one present, all are present
    p = m.run_defaults
    p['attn_norm'] = any('ln1' in k for k in pt_model.keys())
    p['mlp_norm'] = any('ln2' in k for k in pt_model.keys())
    p['final_norm'] = any('ln_final' in k for k in pt_model.keys())
    p['use_attn'] = any('attn' in k for k in pt_model.keys())
    p['use_mlp'] = any('mlp' in k for k in pt_model.keys())
    
    m.embeds = pt_model['embed.W_E'].cpu().numpy()            # (n_vocab, k)
    m.pos_embeds = pt_model['pos_embed.W_pos'].cpu().numpy()  # (seq_len, k)
    
    # Compute number of blocks based on PyTorch keys
    n_blocks = max([int(key.split('.')[1]) + 1 if key.split('.')[0] == 'blocks' else 0 for key in pt_model.keys()])
    
    # Stack each block's matrix together, store inside ezmodel
    if n_blocks > 0:
        for ezlabel, (cond,ptlabel) in block_ezlabel_cond_ptlabel.items():
            if p[cond] is True:
                m.__dict__[ezlabel] = np.stack([pt_model[f'blocks.{b}.{ptlabel}'].cpu().numpy() for b in range(n_blocks)])
    
    if p['final_norm']:
        m.wnf = pt_model['ln_final.w'].cpu().numpy()   # (k,)
        m.bnf = pt_model['ln_final.b'].cpu().numpy()   # (k,)
    
    m.wu = pt_model['unembed.W_U'].cpu().numpy()     # (k, n_classes)
    m.bu = pt_model['unembed.b_U'].cpu().numpy()     # (n_classes,)
    
    m.set_params_from_weights()
    
    return m


def compare_tflens(inp, ezmodel, pt_model, device='cpu'):
    '''
    Compare how similar the ezmodel is to TransformerLens.
    - pt_model: Either a PyTorch filename -or- PyTorch object.
    '''

    tl_model = ezmodel.to_tflens(pt_model)
    
    toks = ezmodel.encoder(inp, device=device)
    ezrun = EZRun(ezmodel, toks)
    
    block_tl_ez = {
        'hook_resid_pre': ezrun.block_in,
        'attn.hook_q': ezrun.attn_q.transpose([0,2,1,3]),
        'ln1.hook_normalized': ezrun.attn_in_normed,
        'hook_resid_mid': ezrun.mlp_in,
        'mlp.hook_pre': ezrun.mlp_interm,
        'mlp.hook_post': ezrun.mlp_actfn,
        'hook_mlp_out': ezrun.mlp_out,
        'hook_resid_post': ezrun.block_out,
    }
    
    with t.inference_mode():
        logits, cache = tl_model.run_with_cache(t.from_numpy(toks).to('cpu'))
        logits = logits.cpu().numpy()[0]
        probs = ezmodel.softmax(logits)
    
    print(f'Largest/mean absolute difference per hook (across all blocks):')
    
    ex = 0  # example number
    for tl_label, ez_values in block_tl_ez.items():
        # Take max/mean across all blocks
        block_diffs = np.array([np.abs(cache[f'blocks.{b}.{tl_label}'].cpu().numpy()[ex] - ez_values[b]) for b in range(ezmodel.n_blocks)])
        print(f'{np.max(block_diffs):.8f} {np.mean(block_diffs):.8f}\t{tl_label}')
    
    final_norm_diff = np.abs(cache[f'ln_final.hook_normalized'].cpu().numpy()[ex] - ezrun.out)
    print(f'{np.max(final_norm_diff):.8f} {np.mean(final_norm_diff):.8f}\tln_final.hook_normalized')
    print(f'{np.max(np.abs(logits - ezrun.logits)):.8f} {np.mean(np.abs(logits - ezrun.logits)):.8f}\tfinal logits')
    print(f'{np.max(np.abs(probs - ezrun.probs)):.8f} {np.mean(np.abs(probs - ezrun.probs)):.8f}\tfinal probs')
