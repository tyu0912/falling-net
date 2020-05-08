# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

import os
import random

dataset_path = '/data/w251fall/jpg/'
label_path = '/data/w251fall/labels/'
sets_path = '/data/w251fall/file_list/'

train_file = 'w251fall_rgb_train_split_1.txt'
val_file = 'w251fall_rgb_val_split_1.txt'

train_split_ratio = 0.8

if __name__ == '__main__':
    directories = []

    with open(os.path.join(label_path, 'categories.txt')) as f:
        categories_raw = f.readlines()
        #categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
        categories=[]
        for cat in categories_raw:
            categories.append(cat.rstrip().split(' '))

    #KH assert len(categories) == 9
    dict_categories = {}
    for category in categories:
        dict_categories[category[0]] = category[1]

    # Go through each this_catergory

    for category_id, category in dict_categories.items():

        # Check for dir for category
        if os.path.isdir(os.path.join(dataset_path, category)):
            video_dirs = [ name for name in os.listdir(os.path.join(dataset_path, category)) if os.path.isdir(os.path.join(dataset_path, category, name)) ]
            #print(video_dirs)
            for vid_dir in video_dirs:
                file_count = len([name for name in os.listdir(os.path.join(dataset_path, category, vid_dir)) if os.path.isfile(os.path.join(dataset_path, category, vid_dir, name))])
                directories.append([os.path.join(dataset_path, category, vid_dir), str(file_count), category_id])

    print(directories)

    # Split to train and val
    train = []
    val = []

    # for dir in directories:
    #     if random.random() <= train_split_ratio:
    #         train.append(dir)
    #     else:
    #         val.append(dir)

    for dir in directories:
        #SM if '.train.avi' in dir:
        #SM print(dir[0])
        if '.train' in dir[0]:
            #SM print(dir)
            train.append(dir)
        #SM elif '.val.avi':
        elif '.val' in dir[0]:
            val.append(dir)
        else:
            print ("Directory not train or validation: {}".format(dir))
    
    random.shuffle(train)

    # Write files.
    with open(os.path.join(sets_path, train_file), 'w') as f:
        for dir in train:
            f.write(" ".join(dir))
            f.write("\n")

    # Write files.
    with open(os.path.join(sets_path, val_file), 'w') as f:
        for dir in val:
            f.write(" ".join(dir))
            f.write("\n")
