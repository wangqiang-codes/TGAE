# -*- coding: utf-8 -*-
'''
@Time    : 2021/9/2 16:17
@Author  : Wang Qiang
@FileName: train.py
'''
import yaml
from torch.utils.data.dataloader import DataLoader
from loader.LPDatasetLoader import LPDataset
from utils.eval_utils import *
from utils.train_utils import setup_seed
from models.TGAE import TGAE

def train_test_model(config, gpu_num=0):

    batch_size = 1
    dataset = config['dataset']
    model_name = config['model_name']
    window_size = config['window_size']
    test_prop = config['test_prop']

    if 'save' not in config:
        config['save'] = False
    if 'save_record' not in config:
        config['save_record'] = False

    base_path = os.path.realpath(os.path.join(r"../data", dataset))
    train_test_save_path = os.path.join(base_path, 'train_test_{}.pkl'.format(test_prop))

    train_test_data = LPDataset(train_test_save_path, window_size)

    config['node_num'] = train_test_data.nb_nodes
    config['max_thres'] = train_test_data.max_thres

    train_test_loader = DataLoader(
        dataset=train_test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    print('\n{}:'.format(dataset))
    print(model_name)
    print('initialize...')
    edr = EvalDataRecorder()

    model = model_initialize(model_name, config, gpu_num=gpu_num)

    running_time_list = []

    for i, data in enumerate(train_test_loader):
        if model:
            model.batch = i
        print("\n\nBatch {}:".format(i))
        train_in_shots, train_out_shot, test_in_shots, test_out_shot = data
        print('==========train=========')
        running_time = model_train(model, model_name, train_in_shots, train_out_shot)

        print('==========test==========')
        result = model_test(model, model_name, test_in_shots)
        if model_name not in ('GCN-GAN', 'TGI', 'TGAE'):
            running_time = result['running_time']
        test_out_shot = test_out_shot.cpu()
        test_label = (test_out_shot > 0.).float()
        # link prediction evaluation
        predicted_shot = result['out_shot_link'].cpu()
        edr.update_link_eval(predicted_shot, test_label)

        # link prediction evaluation (weight)
        predicted_shot = result['out_shot_weight'].cpu()
        predicted_shot = renormalize(predicted_shot, train_test_data.max_thres, epsilon=0.01)
        edr.update_weight_eval(predicted_shot, (test_out_shot * train_test_data.max_thres))

        # report_intermediate_result
        running_time_list.append(running_time)
        edr.update_running_time(running_time)

    # overall indicators
    print('=======ave result=======')
    edr.add_ave_eval(batch_size=batch_size)
    if config['save_record']:
        edr.save_record(model_name, dataset, config)

    # report_final_result
    print("RMSE:", edr.record['ave_RMSE'][-1], "JS:", edr.record['ave_JS'][-1], "mismatch:",
          edr.record['ave_mismatch'][-1], "AUROC:", edr.record['ave_AUROC'][-1], "RunningTime:", np.mean(running_time_list))

def model_initialize(model_name, config, gpu_num=-1):
    if int(gpu_num) < 0 or torch.cuda.device_count() == 0:
        device = 'cpu'
    elif torch.cuda.device_count() >= int(gpu_num) + 1:
        device = torch.device("cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu")
    else:
        raise Exception("Torch can not detect the GPU number {}.".format(gpu_num))
    if model_name == 'TGAE':
        model = TGAE(device, config)
    else:
        raise Exception('Unsupported model {}'.format(model_name))
    return model

def model_train(model, model_name, train_in_shots, train_out_shot):
    if model_name == 'TGAE':
        running_time = model.train(train_in_shots, train_out_shot)
    else:
        raise Exception('Unsupported model {}'.format(model_name))
    return running_time

def model_test(model, model_name, test_in_shots):
    if model_name == 'TGAE':
        result = model.test(test_in_shots)
    else:
        raise Exception('Unsupported model {}'.format(model_name))
    return result

def main(params):
    model_name = 'TGAE'
    dataset = 'New York Green Taxi'
    window_size = 15
    test_prop = 1
    setup_seed(0)

    if model_name == 'TGAE':
        config = yaml.load(open('./configs/TGAE_config.yml'))
    else:
        raise Exception('Unsupported model {}'.format(model_name))

    # update parameters
    config.update(params)
    config.update({'dataset': dataset, 'model_name': model_name, 'window_size': window_size, 'test_prop': test_prop})

    train_test_model(config, gpu_num=0)


if __name__ == '__main__':
    params = {
        'lr': 0.0001,
        'lstm_dim': 128,
        'weight_decay': 0.0,
        'weight_coef': 0.3,
        'n_heads': 1,
        'window_size': 15
    }

    main(params)

