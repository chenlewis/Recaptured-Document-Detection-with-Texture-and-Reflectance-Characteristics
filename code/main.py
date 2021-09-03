from config import opt
import os
import torch as t
from torch.utils.data import DataLoader
import models
from data.dataset import Copy_Detection
from torchnet import meter
from utils.visualizer import Visualizer
import torch.nn as nn
from torch.nn import functional as T
from tqdm import tqdm
import numpy as np

def train(**kwargs):
    opt._parse(kwargs)
    # vis = Visualizer(opt.env, port=opt.vis_port)
    model = getattr(models, opt.model)()
    net = nn.DataParallel(model, device_ids=[0])
    net.to(opt.device)


    train_data = Copy_Detection(opt.train_data_root, train=True)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size_0, shuffle=True, num_workers=opt.num_workers)
    val_data = Copy_Detection(opt.val_data_root, train=False)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size_1, shuffle=True, num_workers=opt.num_workers)

    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    confusion_matrix = meter.ConfusionMeter(2)
    best_acc = 0.5
    for epoch in range(opt.max_epoch):
        print(epoch)
        loss_meter.reset()
        confusion_matrix.reset()
        for ii, (data, label) in tqdm(enumerate(train_loader)):
            input = data.to(opt.device)
            target = label.to(opt.device)
            score = net(input)
            optimizer.zero_grad()
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())
            #
            # if (ii + 1) % opt.print_freq == 0:   #print_freq=20
            #     vis.plot('loss', loss_meter.value()[0])
            #     if os.path.exists(opt.debug_file):
            #         import ipdb;
            #         ipdb.set_trace()
        cm_accuracy = confusion_matrix.value()
        train_accuracy = 100 * (cm_accuracy[0][0] + cm_accuracy[1][1]) / (cm_accuracy.sum())
        # vis.plot('train_accuracy', train_accuracy)

        val_cm, val_accuracy = val(model, val_loader)
        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
        #     lr=lr))
        if best_acc < val_accuracy:
            best_acc = val_accuracy
            t.save(model.state_dict(), 'The name of the saved model'.format(epoch))
        print(loss_meter.value()[0])
        print(train_accuracy)
        print(val_accuracy)
        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]




def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in enumerate(dataloader):
        val_input = val_input.to(opt.device)
        label = label.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def test(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)().eval()

    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path, map_location='cuda:0'))
        # model.load(opt.load_model_path)
    net = nn.DataParallel(model, device_ids=[0])
    net.to(opt.device)
    test_data = Copy_Detection(opt.test_data_root, test=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size_1, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_loader):
        input = data.to(opt.device)
        score = net(input)

        probability = T.softmax(score, dim=1)[:, 1].detach().tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)
    return

def write_csv(results, file_name):
    import csv
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    import fire
    fire.Fire()