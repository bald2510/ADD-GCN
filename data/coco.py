import os
import json
import subprocess
from PIL import Image
# import numpy as np
import torch
from torch.utils.data import Dataset
import random

urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
        'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def download_coco2014(root, phase):
    '''
    Download COCO 2014 dataset and extract it.
    Args:
        root (str): Root directory of dataset.
        phase (str): train or val.
    '''
    
    # check if the root directory exists
    work_dir = os.getcwd()
    tmpdir = os.path.join(root, 'tmp/')
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
        
    # check if the dataset is already downloaded
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    
    # check if the dataset is already extracted, extract file if needed
    img_data = os.path.join(root, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file,root)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    # check if the annotations are already downloaded
    # if not, download and extract them
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(root, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    print('[annotation] Done!')

    ''' 
    Check if the annotations are already generated. if not, generate them
    
    Generation of annotations: 
        1. load the annotations file
        2. get the category id and category name
        3. get the image id and image name
        4. get the annotations for each image
        5. save the annotations in a json file
        6. save the category id and category name in a json file
        7. save the image id and image name in a json file
    '''
    annotations_data = os.path.join(root, 'annotations')
    anno = os.path.join(root, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(root, 'category.json')):
            json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
    os.chdir(work_dir)

def categoty_to_idx(category):
    '''
    Convert category to index.
    
    Args:
        category (list): List of category names.
    Returns:
        dict: Dictionary of category names and their corresponding index.
    '''
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(Dataset):
    '''
    COCO2014 dataset for classification.
    Args:
        root (str): Root directory of dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        phase (str): train or val.
    '''
    
    def __init__(self, root, transform=None, phase='train'):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.transform = transform
        download_coco2014(self.root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx) 
        #self.img_list = random.sample(self.img_list, int(len(self.img_list) * 0.005))
        print('[dataset] COCO2014 classification phase={} number of classes={}  number of images={}'.format(phase, self.num_classes, len(self.img_list)))

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # target = np.zeros(self.num_classes, np.float32) - 1
        target = torch.zeros(self.num_classes,  dtype=torch.float32) - 1
        target[labels] = 1
        data = {'image':img, 'name': filename, 'target': target}
        return data
