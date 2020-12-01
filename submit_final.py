import sys
import warnings
from PIL.Image import DecompressionBombWarning
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DecompressionBombWarning)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataload import Dataset,Testset
import time
from models.mymodel import BaseModel,BaseModel1
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
import os


def test_model():
    model1.eval()
    model2.eval()
    # model3.eval()
    num_steps = len(eval_loader)
    print(f'total batches: {num_steps}')

    preds_5512 = []
    preds_5513 = []
    preds_5523 = []
    preds_433 = []
    preds_343 = []
    preds_442 = []
    preds_333 = []
    image_names = []
    confident=[]
    with torch.no_grad():
        for i, (XI, image_name) in enumerate(tqdm(eval_loader)):
            # if i % 50 == 0:
            #     print(i, i/len(eval_loader))
            bs,nc,c,h,w = XI.size()
            x = XI.cuda(device_id)


            output1 = model1(x.view(-1,c,h,w))
            output2 = model2(x.view(-1, c, h, w))
            # output3 = model3(x.view(-1, c, h, w))


            # output=model(x)
            output1 = output1.view(bs,nc,-1).mean(1)
            output2 = output2.view(bs, nc, -1).mean(1)
            # output3 = output3.view(bs, nc, -1).mean(1)
            output1 = nn.Softmax(dim=1)(output1)
            output2 = nn.Softmax(dim=1)(output2)
            # output3 = nn.Softmax(dim=1)(output3)




            output_442=0.4*output1+0.6*output2
            output_333 = 0.5 * output1 + 0.5 * output2
            output_5512 = 0.6 * output1 + 0.4 * output2
            output_5513 = 0.3 * output1 + 0.7 * output2
            output_5523 = 0.7 * output1 + 0.3 * output2
            # output_433 = 0.4 * output1 + 0.3 * output2 + 0.3 * output3
            # output_343 = 0.3 * output1 + 0.4 * output2 + 0.3 * output3

            confs, predicts_442 = torch.max(output_442.detach(), dim=1)
            confs, predicts_333 = torch.max(output_333.detach(), dim=1)
            confs, predicts_5512 = torch.max(output_5512.detach(), dim=1)
            confs, predicts_5513 = torch.max(output_5513.detach(), dim=1)
            confs, predicts_5523 = torch.max(output_5523.detach(), dim=1)
            # confs, predicts_433 = torch.max(output_433.detach(), dim=1)
            # confs, predicts_343 = torch.max(output_343.detach(), dim=1)


            # confident+=list(confs.cpu().numpy() > 0.7)
            preds_442 += list(predicts_442.cpu().numpy())
            preds_333 += list(predicts_333.cpu().numpy())
            preds_5512 += list(predicts_5512.cpu().numpy())
            preds_5513 += list(predicts_5513.cpu().numpy())
            preds_5523 += list(predicts_5523.cpu().numpy())
            # preds_433 += list(predicts_433.cpu().numpy())
            # preds_343 += list(predicts_343.cpu().numpy())

            image_names += list(image_name)


    print(len(preds_442), len(image_names))



    if not os.path.exists(csv_path_442):
        with open(csv_path_442, 'w'):
            pass
    with open(csv_path_442, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds_442))):
            f.write('{0},{1}\n'.format(image_names[i], preds_442[i]))

    if not os.path.exists(csv_path_333):
        with open(csv_path_333, 'w'):
            pass
    with open(csv_path_333, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds_333))):
            f.write('{0},{1}\n'.format(image_names[i], preds_333[i]))

    if not os.path.exists(csv_path_5512):
        with open(csv_path_5512, 'w'):
            pass
    with open(csv_path_5512, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds_5512))):
            f.write('{0},{1}\n'.format(image_names[i], preds_5512[i]))

    if not os.path.exists(csv_path_5513):
        with open(csv_path_5513, 'w'):
            pass
    with open(csv_path_5513, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds_5513))):
            f.write('{0},{1}\n'.format(image_names[i], preds_5513[i]))

    if not os.path.exists(csv_path_5523):
        with open(csv_path_5523, 'w'):
            pass
    with open(csv_path_5523, 'w') as f:
        f.write('{0},{1}\n'.format('image_name', 'class'))
        for i in tqdm(range(len(preds_5523))):
            f.write('{0},{1}\n'.format(image_names[i], preds_5523[i]))

    # if not os.path.exists(csv_path_433):
    #     with open(csv_path_433, 'w'):
    #         pass
    # with open(csv_path_433, 'w') as f:
    #     f.write('{0},{1}\n'.format('image_name', 'class'))
    #     for i in tqdm(range(len(preds_433))):
    #         f.write('{0},{1}\n'.format(image_names[i], preds_433[i]))
    #
    # if not os.path.exists(csv_path_343):
    #     with open(csv_path_343, 'w'):
    #         pass
    # with open(csv_path_343, 'w') as f:
    #     f.write('{0},{1}\n'.format('image_name', 'class'))
    #     for i in tqdm(range(len(preds_343))):
    #         f.write('{0},{1}\n'.format(image_names[i], preds_343[i]))


            # if confident[i]:
            #     f.write('{0},{1}\n'.format(image_names[i], preds[i]))


if __name__ == '__main__':
    csv_path_442 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_442.csv'
    csv_path_333 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_333.csv'
    csv_path_5512 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_5512.csv'
    csv_path_5513 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_5513.csv'
    csv_path_5523 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_5523.csv'
    csv_path_433 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_433.csv'
    csv_path_343 = './output/se_resnext50_32x4d/se_resnext50_2_acc_0.6070_343.csv'
    test_batch_size =8
    num_class = 5000
    device_id = 0
    model_name1 = 'efficientnet-b4'
    # model_name2 = 'efficientnet-b5'

    model_path1= './best_test52_acc_0.5635.pth'
    model_path2='./best_3_acc_0.6892.pth'
    # model_path3 = './B5_acc_0.6681.pth'
    # model = get_efficientnet(model_name=model_name)
    # model, *_ = model_selection(modelname=model_name, num_out_classes=5000, dropout=None)
    model1 = BaseModel(model_name=model_name1, num_classes=num_class, pretrained=1,
                       pool_type='cat', down=1)
    model2 = BaseModel(model_name=model_name1, num_classes=num_class, pretrained=1,
                       pool_type='cat', down=1)
    # model3 = BaseModel(model_name=model_name2, num_classes=num_class, pretrained=1,
    #                    pool_type='cat', down=1)
    if model_path1 is not None:
        my_model1 = torch.load(model_path1, map_location='cpu')
        model1.load_state_dict(my_model1['net'])
        print('Model found in {}'.format(model_path1))
    else:
        print('No model found, initializing random model.')
    if model_path2 is not None:
        my_model2 = torch.load(model_path2, map_location='cpu')
        model2.load_state_dict(my_model2['net'])
        print('Model found in {}'.format(model_path2))
    else:
        print('No model found, initializing random model.')
    # if model_path3 is not None:
    #     my_model3 = torch.load(model_path3, map_location='cpu')
    #     model3.load_state_dict(my_model3['net'])
    #     print('Model found in {}'.format(model_path3))
    # else:
    #     print('No model found, initializing random model.')

    model1 = model1.cuda(device_id)
    model2 = model2.cuda(device_id)
    # model3 = model3.cuda(device_id)

    start = time.time()
    epoch_start = 1
    num_epochs = 1

    testset = Testset(root='./data/test')
    # testset1 = Dataset('./data/test_448')
    # eval_loader = DataLoader(xdl_test, batch_size=test_batch_size, shuffle=False, num_workers=4)
    eval_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    test_dataset_len = len(testset)
    print('test_dataset_len:', test_dataset_len)
    test_model()
    print('Total time:', time.time() - start)







