import torch as t
import warnings

class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'model name'
    train_data_root = 'the path of training set'
    val_data_root = 'the path of validation set'
    test_data_root = 'the path of testing set'
    load_model_path = 'the path of model'
    batch_size_0 = 256
    batch_size_1 = 256
    use_gpu = True
    num_workers = 8
    print_freq = 20
    debug_file = ''
    result_file = 'file to save the output of the model'
    max_epoch = 10
    lr = 0.0001
    lr_decay = 0.5
    weight_decay = 1e-5

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        opt.device = t.device('cuda:0') if opt.use_gpu else t.device('cpu')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
# new_config = {'batch_size_0': 256}
# opt._parse(new_config)