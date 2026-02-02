hyper_params = {
    'device' : 'cpu', # 'cuda'
    # Paper: "learning rate of 0.001 using the Adam optimizer over 250 epochs"
    'ae_lrn_rt' : 1e-3,
    'n_epochs' : 250,
    'btch_sz' : 32,
    # Paper: "[8, 16, 32] filters per layer" (we use same)
    'cnvl_filters' : [8, 16, 32],
    # Paper uses equal weights: "L = LMSE + Ltask + Lsession"
    # No weighting mentioned in paper - all losses contribute equally
}