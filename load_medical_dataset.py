#%%
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from PIL import Image
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from transforms.transforms_composer import transforms_composer
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
#%%
class BaseClassifierDataset(Dataset):
    """Classifier dataset."""

    def __init__(self,
                file_paths: str,
                cls_type: str,
                 mode : str,
                 transforms: dict = None,
                 image_root: str = 'data/',
                 test: bool = False) -> None:
        """
        Init the Dataset
        """

        self.mode=mode
        self.cls_type=cls_type 
        self.test = test

        with open(file_paths, 'r') as openfile:
            json_object = json.load(openfile)
        
        self.image_root = image_root + json_object['name'] + '/'
        
        if self.mode=='train':
            json_object['data']=json_object['data'][:int(len(json_object['data'])*json_object['data_split']['train_split'])]
        elif self.mode=='valid':
            json_object['data']=json_object['data'][int(len(json_object['data'])*json_object['data_split']['train_split']):]
        
        self.df=[]
        for row in json_object['data']:
            self.df.append([ row['image_path'], row['label']])

        columns =["image_path","label"]

        self.df=pd.DataFrame(self.df,columns =columns)

        if transforms:
            self.transforms = transforms_composer(transforms)


    def __len__(self) -> int:
        """
        Returns the total length of dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Gets an item from dataset
        """
        row = self.df.loc[idx,:]
        # print(row)
        image_name = row['image_path']
        labels = row['label']
        image = Image.open(os.path.join(self.image_root, 
                                        image_name)).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)
        
        if self.cls_type=='MC':
            cls_label = labels
        else:
            cls_label = torch.Tensor(labels).type(torch.float)
        # print(type(label))
        if self.test:
            return image,cls_label
        else:
            return image, cls_label, idx
#%%
def load_medical_dataset():
    size = [224, 288]
    #size= [32, 32]
    data_train_json = { "train": {
        "path": "TC_TV_other_classification_train_al.json",
        "transform": {
            "RGBToGrayscale": { "p": 1 },
            "Resize": { "new_size": size, "p": 1 },
            "ImageToTensor": { "p": 1 }
        }
        }
    }

    # data_unlabeled_json = { "unlabeled": {
    #     "path": "trail_cp_cyst_classifier_data_unlabeled_al.json",
    #     "transform": {
    #         "RGBToGrayscale": { "p": 1 },
    #         "Resize": { "new_size": size, "p": 1 },
    #         "ImageToTensor": { "p": 1 }
    #     }
    #     }
    # }

    data_test_json = { "test": {
        "path": "TC_TV_other_classification_test_al.json",
        "transform": {
            "RGBToGrayscale": { "p": 1 },
            "Resize": { "new_size": size, "p": 1 },
            "ImageToTensor": { "p": 1 }
        }
        }
    }

    data_train = BaseClassifierDataset(
                    file_paths = data_train_json['train']['path'],
                    cls_type="MC",
                    mode="full",
                    transforms= data_train_json['train']['transform'],
                    image_root="/home/ubuntu/active_learning/",
                    test = False
                )
    
    # data_unlabeled = BaseClassifierDataset(
    #                 file_paths = data_unlabeled_json['unlabeled']['path'],
    #                 cls_type="MC",
    #                 mode="full",
    #                 transforms= data_unlabeled_json['unlabeled']['transform'],
    #                 image_root="/home/ubuntu/active_learning/",
    #                 test = False
    #             )

    data_test = BaseClassifierDataset(
                    file_paths = data_test_json['test']['path'],
                    cls_type="MC",
                    mode="full",
                    transforms= data_test_json['test']['transform'],
                    image_root="/home/ubuntu/active_learning/",
                    test = True
                )
    NO_CLASSES = 3
    ADDENDUM  = 1000
    # no_train = ds.__len__()
    # NUM_TRAIN = no_train
    # indices = list(range(NUM_TRAIN))
    # np.random.shuffle(indices)
    # labeled_set = indices[:ADDENDUM]
    # unlabeled_set = [x for x in indices if x not in labeled_set]
    #train_loader = DataLoader(ds, batch_size=4, sampler=SubsetRandomSampler(labeled_set), pin_memory=True, drop_last=True)
    #print(train_loader[0])
    it = iter(data_train)
    data = next(it)
    print(f"Train data shape{data[0].shape}")
    it = iter(data_test)
    data = next(it)
    print(f"Test data shape{data[0].shape}")
    plt.figure()
    plt.imshow(data[0][0,:,:], cmap='gray')

    return data_train, data_train, data_test, ADDENDUM, NO_CLASSES, data_train.__len__() # batch size * no. of batches
# %%
# data_train, data_unlabeled, data_test, ADDENDUM, NO_CLASSES, x = load_medical_dataset()
