import matplotlib
matplotlib.use('Agg')
import glob
import os
import numpy as np
import tensorflow as tf
from keras import Input
import cv2
import matplotlib.pyplot as plt
import keras.backend as K
from keras.applications import VGG19
import time
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array, load_img
#from scipy.misc import imsave, imresize
# from matplotlib.pyplot import imread,IMREAD_COLOR
import matplotlib.image as mpimg
from cv2 import imread, COLOR_BGR2RGB, IMREAD_COLOR
from PIL import Image
import logging

config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def sample_images_mask(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)
    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)
    #print('images_batch', len(images_batch))
    low_resolution_images = []
    high_resolution_images = []
    for img in images_batch:
        # Get an ndarray of the current image
        # img1 =cv2.imread(img)
        img1 = plt.imread(img)
        img1 = (img1[:, :, 0:3]*255).astype(np.uint8)
        
        
        high = (img.strip('_low.png')) + '.png'
        mask=np.load((img.strip('_low.png')) + '_mask.npy')
        low = cv2.inpaint(img1, mask, 3, cv2.INPAINT_NS)
        # img1_high_resolution = cv2.imread(high)
        img1_high_resolution = plt.imread(high)
        img1_high_resolution = img1_high_resolution[:, :, 0:3]
        #print("#######HIGH RESOLUTION###########")
        #print(np.min(img1_high_resolution))
        #print(np.max(img1_high_resolution))
        # img1_high_resolution = cv2.cvtColor(img1_high_resolution, cv2.COLOR_BGR2RGB)
        # img1_low_resolution = imresize(img1, low_resolution_shape)
        # img1_high_resolution = img1
        img1_low_resolution = low/255
        # Do a random flip
        '''
                if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)
        '''

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)
    return (np.array(high_resolution_images), np.array(low_resolution_images))
# TF_FORCE_GPU_ALLOW_GROWTH=True
# session=tf.Session(config=config)
def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)
    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)
    #print('images_batch', len(images_batch))
    low_resolution_images = []
    high_resolution_images = []
    for img in images_batch:
        # Get an ndarray of the current image
        # img1 =cv2.imread(img)
        img1 = plt.imread(img)
        img1 = img1[:, :, 0:3]
        # print("##########LOW RESOLUTION##########")
        # print(np.min(img1))
        # print(np.max(img1))
        # img1=Image.open(img)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = img1.astype(np.float32)
        # Resize the image

        high = (img.strip('_low.png')) + '.png'

        # img1_high_resolution = cv2.imread(high)
        img1_high_resolution = plt.imread(high)
        img1_high_resolution = img1_high_resolution[:, :, 0:3]
        #print("#######HIGH RESOLUTION###########")
        #print(np.min(img1_high_resolution))
        #print(np.max(img1_high_resolution))
        # img1_high_resolution = cv2.cvtColor(img1_high_resolution, cv2.COLOR_BGR2RGB)
        # img1_low_resolution = imresize(img1, low_resolution_shape)
        # img1_high_resolution = img1
        img1_low_resolution = img1
        # Do a random flip
        '''
                if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)
        '''

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)
    return (np.array(high_resolution_images), np.array(low_resolution_images))


def build_generator():
    """    Create a generator network using the hyperparameter values defined below    :return:    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (128, 128, 3)
    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)
    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(input_layer)
    gen1 = PReLU(shared_axes=[1, 2])(gen1)
    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)
    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    # Take the sum of the output from the pre-residual block(gen1) and
    #   the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    # Add an upsampling block
    gen4 = UpSampling2D(size=1)(gen3)  ## 1 time upsampling
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 =PReLU(shared_axes=[1,2])(gen4)
    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1,
                  padding='same')(gen5)
    gen5 =PReLU(shared_axes=[1,2])(gen5)
    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)
    print("OUTPUT SHAPE", gen6.shape)
    # Keras model
    model = Model(inputs=[input_layer], outputs=[output],
                  name='generator')
    print("GENERATOR SUMMARY")
    model.summary()
    return model


def residual_block(x):
    """    Residual block    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"
    res = Conv2D(filters=filters[0], kernel_size=kernel_size,
                 strides=strides, padding=padding,activation='relu')(x)

    res = BatchNormalization(momentum=momentum)(res)
    res = PReLU(shared_axes=[1, 2])(res)
    res = Conv2D(filters=filters[1], kernel_size=kernel_size,
                 strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)
    # Add res and x
    res = Add()([res, x])
    return res


def build_discriminator():
    """    Create a discriminator network using the hyperparameter values defined below    :return:    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)
    input_layer = Input(shape=input_shape)
    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)
    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)
    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)
    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)
    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)
    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)
    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)
    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)
    # Last dense layer - for classification
    output = Dense(units=1, activation='sigmoid')(dis9)
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model


def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    '''
    low_resolution_image=(low_resolution_image+1)*127.5
    original_image = (original_image + 1) * 127.5
    generated_image = (generated_image + 1) * 127.5
    '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)

    ax.axis("off")
    ax.set_title("Original")
    #print("BEFORE SAVING")

    ax = fig.add_subplot(1, 3, 3)
    #print(np.max(generated_image))
    #print(np.min(generated_image))
    # generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
    # generated_image = (generated_image +1.)
    #print('NOW **********************')
    #print(np.max(generated_image))
    #print(np.min(generated_image))
    ax.imshow(generated_image)

    ax.axis("off")
    ax.set_title("Generated")
    plt.savefig(path)
    plt.close()
    '''
        images = []
    images.append(low_resolution_image)
    images.append(original_image)
    images.append(generated_image)
        
    titles = ['low', 'Original Image', 'generated']

    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.savefig('temp' + path)
    plt.close()
    '''



def write_log(callback, name, value, batch_no):
    """
    Write scalars to Tensorboard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def preprocess(x):
    # scaling from [-1,1] to [0, 255]
    x += 1.
    x *= 127.5

    # RGB to BGR

    x = x[..., ::-1]


    # applying Imagenet preprocessing : BGR mean
    mean = [103.939, 116.778, 123.68]
    IMAGENET_MEAN = K.constant(-np.array(mean))
    x = K.bias_add(x, K.cast(IMAGENET_MEAN, K.dtype(x)))

    return x

def vgg_loss(gen_img,high_img):
    """    Build the VGG network to extract image features    """
    input_shape = (256, 256, 3)
    # Load a pre-trained VGG19 model
    vgg19 = VGG19(include_top=False,input_shape=input_shape,weights="imagenet")
    vgg19.trainable=False
    for layer in vgg19.layers:
        layer.trainable=False


    # Create model
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)
    features_pred =model(preprocess(gen_img))
    features_true = model(preprocess(high_img))
    return 0.006*K.mean(K.square(features_pred - features_true), axis=-1)


def build_adversarial_model(generator, discriminator, vgg):
    input_low_resolution = Input(shape=(128, 128, 3))
    fake_hr_images = generator(input_low_resolution)
    fake_features = vgg(fake_hr_images)
    discriminator.trainable = False
    output = discriminator(fake_hr_images)
    model = Model(inputs=[input_low_resolution],
                  outputs=[output, fake_features])
    for layer in model.layers:
        print(layer.name, layer.trainable)
    print(model.summary())
    return model


# Define hyperparameters
data_dir = "training_imageData_Albedo2/**_low.png"
pretrained_weights='initialModel_VGGloss_albedo.h5'
dataset_name="Albedo"
epochs = 500
batch_size = 64
# Shape of low-resolution and high-resolution images
low_resolution_shape = (128, 128, 3)
high_resolution_shape = (256, 256, 3)
# Common optimizer for all networks
#common_optimizer = Adam(0.0002, 0.5)
common_optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#load generator weights
generator = build_generator()

generator.load_weights(pretrained_weights)

#generator.compile(loss='mse',optimizer=common_optimizer)
#build discriminator
discriminator = build_discriminator()
discriminator.summary()
discriminator.name = 'discriminator'
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.9))
#build adversarial
discriminator.trainable = False

input_generator_gan = Input(shape=low_resolution_shape ,name='input_generator_gan')

output_generator_gan = generator(input_generator_gan)
output_discriminator_gan = discriminator(output_generator_gan)

adversarial_model = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])

adversarial_model.compile(loss=[vgg_loss, "binary_crossentropy"],
                      loss_weights=[1., 1e-3],
                      optimizer=Adam(lr=1e-4, beta_1=0.9))

adversarial_model.summary()

tensorboard = TensorBoard(log_dir="./testLogs/".format(time.time()))
tensorboard.set_model(generator)

dloss = []
dlossFake=[]
dlossReal=[]
log_dir=dataset_name+"_logs"
os.mkdir(log_dir)

gloss = []
for epoch in range(epochs):
    print("TRAINING EPOCH NUMBER IS:{}".format(epoch))
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                  low_resolution_shape=low_resolution_shape,
                                                                  high_resolution_shape=high_resolution_shape)

    high_resolution_images = high_resolution_images / 0.5 - 1.
    # low_resolution_images = low_resolution_images / 127.5 - 1.
    ####train discriminator#####
    discriminator.trainable = True
    generated_high_resolution_images = generator.predict(low_resolution_images)

    real_labels = np.ones((batch_size,16,16,1))
    fake_labels = np.zeros((batch_size,16,16,1))

    d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    dloss.append(d_loss)
    dlossReal.append(d_loss_real)
    dlossFake.append(d_loss_fake)
    ######train generator
    discriminator.trainable=False
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir,
                                                                  batch_size=batch_size,
                                                                  low_resolution_shape=low_resolution_shape,
                                                                  high_resolution_shape=high_resolution_shape)
    print(np.min(high_resolution_images))
    print(np.max(high_resolution_images))

    # Normalize images
    high_resolution_images = high_resolution_images / 0.5 - 1.

    print("after reshaping")
    print(np.min(high_resolution_images))
    print(np.max(high_resolution_images))

    g_loss = adversarial_model.train_on_batch([low_resolution_images],
                                              [high_resolution_images,real_labels])
    print('epoch: %d, \n [Discriminator :: d_loss: %f],[Discriminator real:: d_loss_real: %f], [Discriminator fake :: d_loss_fake: %f], \n [ Generator mse :: loss: %f],[ Generator ctext :: loss: %f]' % (epoch, d_loss,d_loss_real,d_loss_fake, g_loss[0],g_loss[1]))
    write_log(tensorboard, 'g_loss', g_loss[0], epoch)
    write_log(tensorboard, 'd_loss', d_loss, epoch)
    gloss.append(g_loss[0])

    if epoch % 100 == 0:

        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir,
                                                                      batch_size=batch_size,
                                                                      low_resolution_shape=low_resolution_shape,
                                                                      high_resolution_shape=high_resolution_shape)
        # Normalize images
        # high_resolution_images = high_resolution_images / 127.5 - 1.
        # Generate fake high-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)
        # Save
        print("generated")
        print(np.min(generated_images))
        print(np.max(generated_images))
        generated_images=(generated_images+1)*0.5

        for index, img in enumerate(generated_images):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path=log_dir+"/img_{}_{}".format(epoch, index))
        generator.save(log_dir+"/genmodel"+"_epoch3300_"+str(epoch)+dataset_name+".h5")

# Specify the path for the generator model
fig = plt.figure()
plt.plot(gloss, 'b--')
plt.savefig('ganModel'+dataset_name+'loss.png')
plt.close()
fig = plt.figure()
plt.plot(dloss, 'b--')
plt.plot(dlossFake, 'r--')
plt.plot(dlossReal, 'g--')
plt.savefig('discriminatorModel'+dataset_name+'loss.png')
plt.close()

generator.save("genModel"+dataset_name+".h5")
discriminator.save("disModel"+dataset_name+".h5")



