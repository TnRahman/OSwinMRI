


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    
    if dataset_type in ['ccoks']:
        from data.dataset_CC359ks import DatasetCCks as D
    
    elif dataset_type in ['m4ks']:
        from data.dataset_M4Rawks import DatasetM4Rawks as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
