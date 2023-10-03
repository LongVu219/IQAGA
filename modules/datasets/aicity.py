from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import os
import random

from .data import BaseImageDataset

class AIcityT1(BaseImageDataset):
    def __init__(self, root="./datasets", verbose=True,**kwargs):
        super(AIcityT1, self).__init__()
        self.dataset_name = 'market'
        self.dataset_dir = osp.join(root)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')

        # self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.val_dir, relabel=False)
        # gallery = self._process_dir(self.gallery_dir, relabel=False)
        gallery = []
        for  _ in range(int(len(query) * 0.5)): #take 10% image for query
            item = random.choice(query)
            gallery += [item]
            query.pop(query.index(item))

        if verbose:
            print("=> AIcity loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        

    def _random_selection(self, dir_path):
        min_img = 100
        max_img = 500
        list_imgs = []
        for file_name in os.listdir(dir_path):
            sub_dir = os.path.join(dir_path, file_name)
            if (os.path.isdir(sub_dir)):
                num_per = random.randint(min_img, max_img)
                selected_sample = random.sample(os.listdir(sub_dir), num_per)
                selected_sample = [os.path.join(sub_dir, x) for x in selected_sample]
                list_imgs += selected_sample
        return list_imgs

    def _process_dir(self, dir_path, relabel=False):
        # img_paths = glob.glob(osp.join(dir_path, '/**/*.jpg'), recursive=True)
        img_paths = self._random_selection(dir_path) 
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            # if relabel is False:
            #     print(img_path)
            #     print(pid)
            #     print(camid)
            #     exit()

        return dataset