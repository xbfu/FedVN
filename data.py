import torch
from GOOD.data.good_datasets import good_motif, good_cmnist, good_zinc, good_sst2


def load_data(dataset_name, data_path, domain, shift='covariate', small_scale=True):
    assert dataset_name in ['sst2', 'zinc', 'motif', 'cmnist']
    if dataset_name == 'sst2':
        assert domain == 'length'
        dataset, meta_info = good_sst2.GOODSST2.load(data_path, domain=domain, shift=shift, generate=True)
    elif dataset_name == 'zinc':
        assert domain in ['scaffold']
        dataset, meta_info = good_zinc.GOODZINC.load(data_path, domain=domain, shift=shift, generate=True)
    elif dataset_name == 'cmnist':
        assert domain in ['color']
        dataset, meta_info = good_cmnist.GOODCMNIST.load(data_path, domain=domain, shift=shift, generate=True)
    else:
        assert domain in ['basis', 'size']
        dataset, meta_info = good_motif.GOODMotif.load(data_path, domain=domain, shift=shift, generate=True)

    num_envs = meta_info['num_envs']

    train, val, test = dataset['train'], dataset['val'], dataset['test']

    dataset['train'].dataset_type = meta_info['dataset_type']

    if dataset_name == 'motif':
        partitions = []
        for env_id in range(num_envs):
            indices = torch.where(train.env_id == env_id)[0]
            partition = train[indices]
            partition.dataset_type = meta_info['dataset_type']
            partitions.append(partition)

        val.dataset_type = meta_info['dataset_type']
        partitions.append(val)
        test.dataset_type = meta_info['dataset_type']
        partitions.append(test)

    else:
        partitions = []
        for env_id in range(num_envs):
            indices = torch.where(train.env_id == env_id)[0]
            partition = train[indices]
            partition.dataset_type = meta_info['dataset_type']
            partitions.append(partition)

    return dataset, partitions


if __name__ == '__main__':
    shift = 'covariate'
    data_path = './data/'
    # dataset, partitions = load_data(dataset_name='cmnist', data_path=data_path, domain='color')
    dataset, partitions = load_data(dataset_name='motif', data_path=data_path, domain='size')
    # dataset, partitions = load_data(dataset_name='zinc', data_path=data_path, domain='scaffold')
    # dataset, partitions = load_data(dataset_name='sst2', data_path=data_path, domain='length')
