import os
import glob
import torch
import argparse

parser = argparse.ArgumentParser(description='model information')
parser.add_argument('--num', help='Number of model', default=0, type=str)
args = parser.parse_args()

folder_path = './models_save'
file_list = os.listdir(folder_path)
file_list.sort()
date = args.num
i = 0
for file_name in file_list:
    if os.path.isdir(os.path.join(folder_path, file_name)):
        model_save_path = os.path.join(folder_path, file_name)
        pt_files = glob.glob(os.path.join(model_save_path, date + '*.pth'))
        ft_files = glob.glob(os.path.join(model_save_path, date + '*ft*.pth'))
        pt_files = list(set(pt_files).difference(set(ft_files)))
        if len(pt_files) > 0:
            i += 1
            pt_files.sort(key=os.path.getctime)
            ckpt = pt_files[-1]
            state = torch.load(ckpt)
            acc_record = state['acc_record']
            acc = sum(acc_record[-10:])/len(acc_record[-10:])
            print('=== ' + str(i) + ' ===')
            print(model_save_path + ', acc: %.3f' % acc)
        if len(ft_files) > 0:
            ft_files.sort(key=os.path.getctime)
            ckpt = ft_files[-1]
            state = torch.load(ckpt)
            acc_record = state['acc_record']
            acc = sum(acc_record[-10:]) / len(acc_record[-10:])
            print(model_save_path + '_ft, acc: %.3f' % acc)
