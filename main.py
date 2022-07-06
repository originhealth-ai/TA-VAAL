
# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from load_medical_dataset import load_medical_dataset
from selection_methods import query_samples
from config import *
from classifier_model.original_model import OriginalClassificationModel
import neptune.new as neptune

parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar10im",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="TA-VAAL",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")

args = parser.parse_args()

##
# Main
if __name__ == '__main__':

    run = neptune.init(
    project="origin-health/Active-learning",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiOWYxMDQyZS1iY2E4LTQ1NjEtYmI1Yy04N2QzNmFmYWVlYWEifQ==",
    )  

    run["sys/tags"].add(["Learning loss","resnext101","Proper", "Large dataset"])

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL']
    datasets = ['cifar10','cifar10im', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    '''
    results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles

    params = {
    "batch_size": BATCH,
    "addendum": ADDENDUM,
    "epoch": args.no_of_epochs,
    "cycles" : CYCLES,
    "optimizer": "Adam",
    "method": method
    }
    run["parameters"] = params
    for trial in range(TRIALS):

        # Load training and testing dataset
        #data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_medical_dataset()
        #print(f"\n\ndata_train = {data_train}\n\n, data_unlabeled = {data_unlabeled}\n\n, data_test = {data_test}\n\n, adden = {adden}\n\n, NO_CLASSES = {NO_CLASSES}\n\n, no_train = {no_train}\n\n")
        
        print('The entire datasize is {}'.format(len(data_train)))       
        ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set= indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetRandomSampler(labeled_set), 
                                    pin_memory=True, drop_last=True)
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        #print(f"\n\ntrain_loader = {train_loader}\n\n, test_loader = {test_loader}\n\n")

        for cycle in range(CYCLES):
            
            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                # if args.dataset == "fashionmnist":
                #     resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
                # else:
                #     #resnet18    = vgg11().cuda() 
                #     resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

                #resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda() #Editted part of the code
                resnext101 = OriginalClassificationModel(backbone = "resnext101", classifiers = {"n_class": NO_CLASSES}).cuda()
                
                if method == 'lloss' or 'TA-VAAL':
                    #loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
                    #loss_module = LossNet(feature_sizes=[56,28,14,7], num_channels=[256,512,1024,2048]).cuda()
                    loss_module = LossNet(feature_sizes=[( 56, 72),(28, 36),(14, 18),(7, 9)], num_channels=[256,512,1024,2048]).cuda()
                    #loss_module = LossNet().cuda()

            models      = {'backbone': resnext101}
            if method =='lloss' or 'TA-VAAL':
                models = {'backbone': resnext101, 'module': loss_module}
            torch.backends.cudnn.benchmark = True
            
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                momentum=MOMENTUM, weight_decay=WDECAY)
 
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss' or 'TA-VAAL':
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}
            
            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL, run)
            acc, f1score, sensitivity, specificity= test(models, EPOCH, method, dataloaders, mode='test')
            
            run["test/accuracy"].log(acc)
            run["test/f1score"].log(f1score)
            run["test/sensitivity"].log(sensitivity)
            run["test/specificity"].log(specificity)
            # run["test/TC_sensitivity"].log(classwise_sensitivity[0])
            # run["test/TC_specificity"].log(classwise_specificity[0])
            # run["test/TV_sensitivity"].log(classwise_sensitivity[1])
            # run["test/TV_specificity"].log(classwise_specificity[1])
            # run["test/other_sensitivity"].log(classwise_sensitivity[2])
            # run["test/other_specificity"].log(classwise_specificity[2])
            print(f"Trial {trial+1}/{TRIALS} || Cycle {cycle+1}/{CYCLES} || Label set size {len(labeled_set)}: Test acc {acc} || Test f1 score {f1score} || Test sensitivity {sensitivity} || Test specificity {specificity}")
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args, run)

            #print(f"\n\nsubset = {subset}, {len(subset)} \n\narg = {arg}, {len(arg)}")
            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True)
    run.stop()
    results.close()