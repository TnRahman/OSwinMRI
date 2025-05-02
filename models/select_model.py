
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt, mode='train'):
    model = opt['model']

    if model == 'swinv2im_ks4_cc':
        from models.model_swinv2im_ks4_cc import MRI_SwinV2IM_KS4_CC as M

    elif model == 'swinv2im_ks4_m4raw':
        from models.model_swinv2im_ks4_m4raw import MRI_SwinV2IM_KS4_M4RAW as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    if mode =='train':
        m = M(opt)

        print('Training model [{:s}] is created.'.format(m.__class__.__name__))

    elif mode == 'eval':
        m = M(opt)
        param_key_g = 'params'

        pretrained_model = torch.load(opt['model_path'])
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    else:
        pass
    return m
