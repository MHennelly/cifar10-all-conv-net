import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model
import joblib

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

def load_batch(file):
    with open(file,'rb') as fo:
        batch = joblib.load(fo)
    return batch

def nextBatch(data,labels,options):
    start = options[0] * options[1]
    end = start + options[0]
    options[1] += 1
    if end > len(data):
        end = len(data) - 1
    b_data = data[start:end]
    b_labels = labels[start:end]
    return b_data, b_labels

def showImg(image):
    plt.imshow(image)
    plt.show()

def getImgClass(label):
    classes = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    return classes[label]

def getLoss(data,labels,bSize):
    totalLoss = 0
    options = [bSize,0]
    while options[1]*bSize <= len(data) - options[0]:
        b_data, b_labels = nextBatch(data,labels,options)
        totalLoss += sess.run(model.cross_entropy,feed_dict={
            model.IL:b_data,
            model.OL_:b_labels,
            model.keep_prob:1
        })
    return totalLoss/len(data)

def getAcc(data,labels,bSize):
    totalCorrect = 0
    options = [bSize,0]
    while options[1]*bSize <= len(data) - options[0]:
        b_data, b_labels = nextBatch(data,labels,options)
        totalCorrect += sess.run(model.accuracy,feed_dict={
            model.IL:b_data,
            model.OL_:b_labels,
            model.keep_prob:1
        })*len(b_data)
    return totalCorrect*100 / len(data)

def test():
    import random
    choice = input("Select a batch to test 1/2/3/4/5/6 (6 for test batch):")
    if int(choice) == 1:
        data = load_batch('./processed_batches/batch1_data')
        labels = load_batch('./processed_batches/batch1_labels')
    elif int(choice) == 2:
        data = load_batch('./processed_batches/batch2_data')
        labels = load_batch('./processed_batches/batch2_labels')
    elif int(choice) == 3:
        data = load_batch('./processed_batches/batch3_data')
        labels = load_batch('./processed_batches/batch3_labels')
    elif int(choice) == 4:
        data = load_batch('./processed_batches/batch4_data')
        labels = load_batch('./processed_batches/batch4_labels')
    elif int(choice) == 5:
        data = load_batch('./processed_batches/batch5_data')
        labels = load_batch('./processed_batches/batch5_labels')
    elif int(choice) == 6:
        data = load_batch('./processed_batches/test_data')
        labels = load_batch('./processed_batches/test_labels')
    else:
        return
    print("Batch loaded")
    saver.restore(sess, "./model_data/model.ckpt")
    print("Model restored")
    acc = getAcc(data,labels,1)
    print("Overall Model Test Accuracy: {0:4.2f}%".format(acc))
    print("Testing Network...")
    loop = True
    try:
        while loop is True:
            num = random.randrange(len(data))
            prediction = sess.run(tf.argmax(model.OL,1),feed_dict={
                model.IL:data[num].reshape(1,32,32,3),
                model.OL_:labels[num].reshape(1,10),
                model.keep_prob:1
            })[0]
            print('Network prediction:', getImgClass(prediction))
            showImg(data[num])
            label = np.argmax(labels[num])
            print('Correct answer:',getImgClass(label),'\n')
    except KeyboardInterrupt:
        loop = False
        plt.close()
    print("Testing Completed")

def train_batch(data,labels,size):
    order = np.arange(len(data))
    np.random.shuffle(order)
    batch_data = np.zeros([size,32,32,3])
    batch_labels = np.zeros([size,10])
    nextIndex = 0
    for selection in order:
        if nextIndex == size:
            sess.run(model.optimizer, feed_dict={
                model.IL: batch_data,
                model.OL_: batch_labels,
                model.keep_prob:0.9
            })
            batch_data = np.zeros([size,32,32,3])
            batch_labels = np.zeros([size,10])
            nextIndex = 0
        else:
            batch_data[nextIndex] = data[selection]
            batch_labels[nextIndex] = labels[selection]
            nextIndex += 1

def train():
    print("Loading batches...")
    batch1_data = load_batch('./processed_batches/batch1_data')
    batch1_labels = load_batch('./processed_batches/batch1_labels')
    batch2_data = load_batch('./processed_batches/batch2_data')
    batch2_labels = load_batch('./processed_batches/batch2_labels')
    batch3_data = load_batch('./processed_batches/batch3_data')
    batch3_labels = load_batch('./processed_batches/batch3_labels')
    batch4_data = load_batch('./processed_batches/batch4_data')
    batch4_labels = load_batch('./processed_batches/batch4_labels')
    batch5_data = load_batch('./processed_batches/batch5_data')
    batch5_labels = load_batch('./processed_batches/batch5_labels')
    test_data = load_batch('./processed_batches/test_data')
    test_labels = load_batch('./processed_batches/test_labels')
    print("Batches loaded")
    epochs = 350
    bSize = 10
    load = input("Load previous model? (Y/N): ")
    if load.lower() == "y":
        saver.restore(sess, "./model_data/model.ckpt")
        print("Model restored")
        temp = getAcc(test_data,test_labels,1)
    print("Training started...")
    try:
        for epoch in range(epochs):
            print("Epoch:",epoch+1)
            if epoch == 100 or epoch == 200 or epoch == 300:
                model.learning_rate *= 0.1
            if epoch == 300:
                bSize = 1
            train_batch(batch1_data,batch1_labels,bSize)
            train_batch(batch2_data,batch2_labels,bSize)
            train_batch(batch3_data,batch3_labels,bSize)
            train_batch(batch4_data,batch4_labels,bSize)
            train_batch(batch5_data,batch5_labels,bSize)
            if (epoch + 1) % 5 == 0:
                temp = getLoss(test_data,test_labels,1)
                print("Loss: {0:4.2f}".format(temp))
                temp = getAcc(test_data,test_labels,1)
                print("Test Accuracy: {0:4.2f}%".format(temp))
            if (epoch + 1) % 20 == 0:
                temp = getAcc(batch1_data,batch1_labels,bSize)
                print("Batch 1 Accuracy: {0:4.2f}%".format(temp))
                temp = getAcc(batch2_data,batch2_labels,bSize)
                print("Batch 2 Accuracy: {0:4.2f}%".format(temp))
                temp = getAcc(batch3_data,batch3_labels,bSize)
                print("Batch 3 Accuracy: {0:4.2f}%".format(temp))
                temp = getAcc(batch4_data,batch4_labels,bSize)
                print("Batch 4 Accuracy: {0:4.2f}%".format(temp))
                temp = getAcc(batch5_data,batch5_labels,bSize)
                print("Batch 5 Accuracy: {0:4.2f}%".format(temp))
            print('')
    except KeyboardInterrupt:
        pass
    print("Training Completed")
    save = input("Save the current model (Y/N):")
    if save.lower() == "y":
        save_path = saver.save(sess, "./model_data/model.ckpt")
        print("Model saved at",save_path)
