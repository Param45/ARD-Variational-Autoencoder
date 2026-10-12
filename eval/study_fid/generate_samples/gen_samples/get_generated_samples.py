import tensorflow as tf
import os
import numpy as np
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Utility functions
def show_images(images, batch_size):
    row_cnt, col_cnt = int(np.sqrt(batch_size)), int(np.sqrt(batch_size))
    images = images[:(row_cnt*col_cnt)]
    fig = plt.figure(figsize=(40, 40))
    gs = gridspec.GridSpec(row_cnt, col_cnt)
    gs.update(wspace=0.1, hspace=0.1)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, vmin=0, vmax=255)

    return fig

def get_generated_data(dataset_name, run_id, basedir, latent_dim, use_decoder_batch_norm, num_filter, latent_vectors, batch_size):

    ###################
    # Autoencoder model
    ###################
    if dataset_name=='MNIST':
        encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [32, 32, 3]
    elif dataset_name=='CelebA':
        encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [64, 64, 3]
    elif dataset_name=='CIFAR10':
        encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter)
        gen_image_shape = [32, 32, 3]
    ####################
    # Checkpoint details
    ####################
    # build the decoder first so its variables are restored (and checked) now; encoder/optimizer are not needed here
    decoder(np.zeros((1, latent_dim), dtype=np.float32), use_batch_norm=use_decoder_batch_norm, training=False)
    model_checkpoint = tf.train.Checkpoint(decoder=decoder)
    checkpoint_dir = os.path.join(basedir, 'Run_' + str(run_id), 'Models')
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    assert checkpoint_path is not None, "No trained checkpoint found in " + checkpoint_dir
    status = model_checkpoint.restore(checkpoint_path).expect_partial()
    status.assert_existing_objects_matched()
    print("Loaded saved model parameters!!")

    fid_samples = latent_vectors.shape[0]
    gen_images_array = np.zeros((fid_samples, gen_image_shape[0], gen_image_shape[1], gen_image_shape[2]), dtype=np.uint8)
    max_iter = fid_samples//batch_size
    print("Number of Iterations are : " + str(max_iter))
    for iter in range(max_iter):
        start_index = iter * batch_size
        end_index = start_index + batch_size
        sampled_z = latent_vectors[start_index:end_index]
        gen_images = decoder(sampled_z, use_batch_norm=use_decoder_batch_norm, training=False)
        if dataset_name == 'MNIST':
            gen_images = tf.image.grayscale_to_rgb(gen_images, name=None)
        if 'DSprites' in dataset_name:
            gen_images = tf.math.sigmoid(gen_images)
            gen_images = tf.image.grayscale_to_rgb(gen_images, name=None)
        if dataset_name == 'CelebA':
            gen_images = (gen_images + 1) / 2
        gen_images = gen_images.numpy()
        gen_images = (gen_images*255).astype(np.uint8)
        gen_images_array[start_index:end_index] = gen_images

    return gen_images_array