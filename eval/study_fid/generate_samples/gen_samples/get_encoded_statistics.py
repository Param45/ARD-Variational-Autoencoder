import tensorflow as tf
import os
import numpy as np
from gen_samples import get_data
from models import ae_model_CelebA, ae_model_CIFAR10, ae_model_MNIST
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy


@tf.function
def compute_jacobian(minibatch, encoder, decoder, encoder_use_batch_norm, decoder_use_batch_norm):
    batch_size = minibatch.shape[0]
    encoder_output = encoder(minibatch, use_batch_norm=encoder_use_batch_norm, training=False)
    mean, log_var = tf.split(encoder_output, num_or_size_splits=2, axis=1)
    sample_z = encoder.sampler.get_sample([mean, log_var])
    latent_dim = mean.shape[1]
    # d(decoder)/d(mean) == d(decoder)/d(z) since z = mean + sigma*eps. Forward mode needs one JVP per
    # latent axis instead of one backward pass per output pixel (3072 for CIFAR-10)
    columns = tf.TensorArray(tf.float32, size=latent_dim)
    for axis in tf.range(latent_dim):
        tangent = tf.one_hot(tf.fill([batch_size], axis), latent_dim)
        with tf.autodiff.ForwardAccumulator(sample_z, tangent) as acc:
            decoder_output = decoder(sample_z, use_batch_norm=decoder_use_batch_norm, training=False)
        columns = columns.write(axis, tf.reshape(acc.jvp(decoder_output), [batch_size, -1]))
    # [latent, batch, pixels] -> [batch, pixels, latent], the batch_jacobian layout
    return tf.transpose(columns.stack(), [1, 2, 0])


def get_encoded_data(dataset_name, run_id, basedir, latent_dim, use_encoder_batch_norm, use_decoder_batch_norm,
                     num_filter, fid_samples, batch_size, generated_image_dir, mode, perc_explained_var):

    relevance_stat_dir = os.path.join(generated_image_dir, 'relevance_stat')
    os.makedirs(relevance_stat_dir, exist_ok=True)

    ###################
    # Autoencoder model
    ###################
    if dataset_name=='MNIST':
        encoder = ae_model_MNIST.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_MNIST.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    elif dataset_name=='CelebA':
        encoder = ae_model_CelebA.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CelebA.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    elif dataset_name=='CIFAR10':
        encoder = ae_model_CIFAR10.Encoder(latent_dim=latent_dim, num_filter=num_filter)
        decoder = ae_model_CIFAR10.Decoder(latent_dim=latent_dim, num_filter=num_filter)
    sample_data = get_data.get_data_for_encoded_statistics(dataset_name, fid_samples, mode)

    ####################
    # Checkpoint details
    ####################
    # build the models first so every variable is restored (and checked) now; optimizer state is not needed here
    encoder(sample_data[:1], use_batch_norm=use_encoder_batch_norm, training=False)
    decoder(np.zeros((1, latent_dim), dtype=np.float32), use_batch_norm=use_decoder_batch_norm, training=False)
    model_checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint_dir = os.path.join(basedir, 'Run_' + str(run_id), 'Models')
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    assert checkpoint_path is not None, "No trained checkpoint found in " + checkpoint_dir
    status = model_checkpoint.restore(checkpoint_path).expect_partial()
    status.assert_existing_objects_matched()
    print("Loaded saved model parameters!!")

    mean_vectors = np.zeros((fid_samples, latent_dim))
    sigma_vectors = np.zeros((fid_samples, latent_dim))
    samples_t_stat = np.zeros((fid_samples, latent_dim))
    gaussian_mean = np.zeros(latent_dim)
    gaussian_cov = np.eye(latent_dim)
    max_iter = fid_samples//batch_size
    print("Number of Iterations are : " + str(max_iter))
    for iter in range(max_iter):
        img_start = iter*batch_size
        img_end = img_start+batch_size
        minibatch = sample_data[img_start:img_end]
        encoder_output = encoder(minibatch, use_batch_norm=use_encoder_batch_norm, training=False)
        mean_encodings, logvar_encodings = tf.split(encoder_output, num_or_size_splits=2, axis=1)
        mean_vectors[img_start:img_end] = mean_encodings.numpy()
        sigma_vectors[img_start:img_end] = tf.exp(0.5*logvar_encodings).numpy()
        noise_samples = np.random.multivariate_normal(mean=gaussian_mean, cov=gaussian_cov, size=batch_size)
        samples_t_stat[img_start:img_end] = mean_vectors[img_start:img_end] + noise_samples*sigma_vectors[img_start:img_end]


    #######################
    # Get the relevant axes
    #######################
    rel_samples = 100
    alpha0 = 0
    beta0 = np.zeros(latent_dim, dtype=np.float32)

    cur_beta_update = np.sum(samples_t_stat ** 2, axis=0) / 2
    samples_count = samples_t_stat.shape[0]

    # Estimated variance
    beta = beta0 + cur_beta_update
    alpha = alpha0 + samples_count / 2
    est_var = beta / alpha
    print("Estimated the variance along axes....")

    rel_data = sample_data[:rel_samples]
    grads = compute_jacobian(rel_data, encoder, decoder, use_encoder_batch_norm, use_decoder_batch_norm).numpy()

    abs_grads = np.abs(grads)
    # average jacobian acorss samples
    mean_abs_grads = np.mean(abs_grads, axis=0)
    # squared norm of the gradient
    mean_abs_grads_norm = np.sum(mean_abs_grads ** 2, axis=0)
    scaled_est_var_grads_norm = mean_abs_grads_norm * est_var

    # Compute the cumulative explained variance
    sort_scaled_est_var_grads_norm = np.sort(scaled_est_var_grads_norm)[::-1]
    argsort_scaled_est_var_grads_norm = np.argsort(scaled_est_var_grads_norm)[::-1]
    cumulative_variances = np.cumsum(sort_scaled_est_var_grads_norm)
    cumulative_variances = (cumulative_variances/cumulative_variances[-1])*100

    line = plt.plot(range(1, latent_dim+1), cumulative_variances, marker='o', linestyle='-', color='r',
                    label='Cumulative Explained Variance')
    cum_explained_var_image_path = os.path.join(relevance_stat_dir, "Cum_explained_var.png")
    plt.xlabel('Latent axes')
    plt.ylabel('Explained variance')
    plt.title('Explained variance by different latent axes')
    plt.legend(loc='lower right')
    plt.savefig(cum_explained_var_image_path)
    plt.close(plt.gcf())

    # save the numpy array
    np_scaled_est_var_grads_norm_savepath = os.path.join(relevance_stat_dir, 'scaled_est_var_grads_norm.npy')
    np.save(np_scaled_est_var_grads_norm_savepath, scaled_est_var_grads_norm)

    # determine the relevant axes and its count
    num_rel_axes = np.where(cumulative_variances > perc_explained_var)[0][0]
    sel_axes = argsort_scaled_est_var_grads_norm[:num_rel_axes+1]
    rel_axes = np.zeros(latent_dim, dtype=np.uint8)
    rel_axes[sel_axes] = 1
    print("Number of relevant axes : " + str(num_rel_axes+1))

    if mode=='reconstruction':
        noise_samples = np.random.multivariate_normal(mean=gaussian_mean, cov=gaussian_cov, size=fid_samples)
        rel_var = sigma_vectors*rel_axes
        latent_vectors = mean_vectors + noise_samples * rel_var
    else:
        # latent vectors used to produce generated samples
        mean_representation = np.mean(mean_vectors, axis=0)
        rel_est_var = est_var*rel_axes
        noise_samples = np.random.multivariate_normal(mean=gaussian_mean, cov=gaussian_cov, size=fid_samples)
        latent_vectors = mean_representation + noise_samples * np.sqrt(rel_est_var)

    del grads, mean_vectors, sigma_vectors, samples_t_stat

    return latent_vectors, num_rel_axes+1