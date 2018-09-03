import numpy as np
import pickle
import joblib

def unpickle(file, selection):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if (selection == None):
        return dict
    else:
        return dict[selection]

def format_RGB(imgdata):
    temp = np.zeros((10000,32,32,3))
    for image in range(10000):
        for row in range(32):
            for col in range(32):
                for color in range(3):
                    val = 32*row + col
                    temp[image,row,col,color] = imgdata[image,val+color*1024]/256
    return temp

def makeOneHot(data):
    oneHot = np.zeros([10000,10])
    for i in range(10000):
        label = np.zeros([10])
        label[data[i]] = 1
        oneHot[i] = label
    return oneHot

def format_helper(imgdata, fname):
    joblib.dump(imgdata,open(fname,'wb'))

def concat(data1,data2,labels):
    return np.concatenate((data1,data2),axis=0), np.concatenate((labels,labels[:10000]),axis=0)

def format():
    print("Unpacking batches...")
    import os, random
    from imgaug import augmenters as iaa
    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Fliplr(0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)
    batch1_d = unpickle('unprocessed_batches/data_batch_1',b'data')
    batch2_d = unpickle('unprocessed_batches/data_batch_2',b'data')
    batch3_d = unpickle('unprocessed_batches/data_batch_3',b'data')
    batch4_d = unpickle('unprocessed_batches/data_batch_4',b'data')
    batch5_d = unpickle('unprocessed_batches/data_batch_5',b'data')
    test_d = unpickle('unprocessed_batches/test_batch',b'data')
    batch1_l = unpickle('unprocessed_batches/data_batch_1',b'labels')
    batch2_l = unpickle('unprocessed_batches/data_batch_2',b'labels')
    batch3_l = unpickle('unprocessed_batches/data_batch_3',b'labels')
    batch4_l = unpickle('unprocessed_batches/data_batch_4',b'labels')
    batch5_l = unpickle('unprocessed_batches/data_batch_5',b'labels')
    test_l = unpickle('unprocessed_batches/test_batch',b'labels')
    batch1_RGB = format_RGB(batch1_d)
    batch2_RGB = format_RGB(batch2_d)
    batch3_RGB = format_RGB(batch3_d)
    batch4_RGB = format_RGB(batch4_d)
    batch5_RGB = format_RGB(batch5_d)
    test_RGB = format_RGB(test_d)
    batch1_hot = makeOneHot(batch1_l)
    batch2_hot = makeOneHot(batch2_l)
    batch3_hot = makeOneHot(batch3_l)
    batch4_hot = makeOneHot(batch4_l)
    batch5_hot = makeOneHot(batch5_l)
    test_hot = makeOneHot(test_l)
    print("Batches unpacked, now augmenting data...")
    numAugments = int(input("Number of augmentations:"))
    for augNumber in range(numAugments):
        print("Augmentation:",augNumber+1)
        batch1_aug = seq.augment_images(batch1_RGB[:10000])
        batch2_aug = seq.augment_images(batch2_RGB[:10000])
        batch3_aug = seq.augment_images(batch3_RGB[:10000])
        batch4_aug = seq.augment_images(batch4_RGB[:10000])
        batch5_aug = seq.augment_images(batch5_RGB[:10000])
        batch1_RGB, batch1_hot = concat(batch1_RGB,batch1_aug,batch1_hot)
        batch2_RGB, batch2_hot = concat(batch2_RGB,batch2_aug,batch2_hot)
        batch3_RGB, batch3_hot = concat(batch3_RGB,batch3_aug,batch3_hot)
        batch4_RGB, batch4_hot = concat(batch4_RGB,batch4_aug,batch4_hot)
        batch5_RGB, batch5_hot = concat(batch5_RGB,batch5_aug,batch5_hot)
        print("Batch sizes:",len(batch1_hot))
    format_helper(batch1_RGB, './processed_batches/batch1_data')
    format_helper(batch1_hot, './processed_batches/batch1_labels')
    print("Batch 1 Saved")
    format_helper(batch2_RGB, './processed_batches/batch2_data')
    format_helper(batch2_hot, './processed_batches/batch2_labels')
    print("Batch 2 Saved")
    format_helper(batch3_RGB, './processed_batches/batch3_data')
    format_helper(batch3_hot, './processed_batches/batch3_labels')
    print("Batch 3 Saved")
    format_helper(batch4_RGB, './processed_batches/batch4_data')
    format_helper(batch4_hot, './processed_batches/batch4_labels')
    print("Batch 4 Saved")
    format_helper(batch5_RGB, './processed_batches/batch5_data')
    format_helper(batch5_hot, './processed_batches/batch5_labels')
    print("Batch 5 Saved")
    format_helper(test_RGB, './processed_batches/test_data')
    format_helper(test_hot, './processed_batches/test_labels')
    print("Test Batch Saved")
    print("Formatting Successful")
