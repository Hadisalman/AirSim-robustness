## Adversarial Attack Config

# Linf Whitebox
attack_config = {
    'random_start' : True, 
    'step_size' : 1./255,
    'epsilon' : 2./255, 
    'num_steps' : 2, 
    'norm' : 'linf',
    }

# Linf Blackbox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 4./255,
#     'epsilon' : 16./255, 
#     'num_steps' : 8, 
#     'norm' : 'linf',
#     'est_grad': (5, 200)
#     }

# L2 Whitebox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 150./255,
#     'epsilon' : 255./255, 
#     'num_steps' : 2, 
#     'norm' : 'l2',
#     }

# L2 Blackbox
# attack_config = {
#     'random_start' : True, 
#     'step_size' : 1000./255,
#     'epsilon' : 6000./255, 
#     'num_steps' : 8, 
#     'norm' : 'l2',
#     'est_grad': (5, 200)
#     }
