import sys
import omniglot
import mini_imagenet
import shapenet

"""
General function that selects and initializes the particular dataset to use for
few-shot classification. Additional dataset support should be added here.
"""


def get_data(dataset, mode='train'):
    if dataset == 'Omniglot':
        return omniglot.OmniglotData(path='../data/omniglot.npy',
                                     train_size=1100,
                                     validation_size=100,
                                     augment_data=True,
                                     seed=111)
    elif dataset == 'miniImageNet':
        return mini_imagenet.MiniImageNetData(path='../data', seed=42)
    elif dataset == 'shapenet':
        return shapenet.ShapeNetData(path='../data',
                                     train_fraction=0.7,
                                     val_fraction=0.1,
                                     num_instances_per_item=36,
                                     seed=42,
                                     mode=mode)
    else:
        sys.exit("Unsupported dataset type (%s)." % dataset)
