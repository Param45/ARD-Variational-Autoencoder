import tensorflow as tf
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from gen_samples import get_encoded_statistics, get_generated_samples
import argparse
from config.local_config import configurations
import imageio
try:
    import nvidia_smi
    HAS_NVIDIA_SMI = True
except ImportError:
    HAS_NVIDIA_SMI = False


def select_GPU(gpu_id=None, min_gpu_mem_frac=0.7):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    use_gpu = None
    free_mem = 0

    if HAS_NVIDIA_SMI and gpu_id is None:
        try:
            nvidia_smi.nvmlInit()
            device_count = nvidia_smi.nvmlDeviceGetCount()
            for device_index in range(device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                if info.free > min_gpu_mem_frac * info.total:
                    use_gpu = device_index
                    free_mem = info.free
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
                    break
            nvidia_smi.nvmlShutdown()
        except Exception:
            pass

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if use_gpu is None:
                use_gpu = 0
    except Exception:
        pass

    return use_gpu, free_mem


###################
# generated samples
###################
def get_std_normnal_samples(noise_dim, sample_count, variance=1):
    target_distribution_mean = np.zeros(noise_dim)
    target_distribution_cov = np.eye(noise_dim)*variance
    std_normal_data = np.random.multivariate_normal(target_distribution_mean, target_distribution_cov, sample_count).astype(np.float32)

    return std_normal_data


def set_seed(seed=0):
    np.random.seed(seed)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--gen_type", type=str, required=True, help="Can be one of the following options: generation, reconstruction")
    parser.add_argument("--eval_ids", type=int, nargs='+', default=[1], help="Run IDs to evaluate, e.g. --eval_ids 1 2 3 4 5")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index to use")
    args = parser.parse_args()

    latent_dim = args.latent_dim
    set_seed()
    mode = args.gen_type
    if args.config_id in configurations:
        config = configurations[args.config_id]
    elif 0 in configurations and args.config_id in configurations[0]:
        config = configurations[0][args.config_id]
    else:
        config = configurations[0]

    # model configurations
    model_name                  = config['model_name']
    dataset_name                = config['dataset_name']
    batch_size                  = config['batch_size']
    num_filter                  = config['num_filter']
    fid_samples                 = config['fid_samples']
    encoder_use_batch_norm      = config['encoder_use_batch_norm']
    decoder_use_batch_norm      = config['decoder_use_batch_norm']
    perc_explained_var          = 99.00

    basedir                     = os.path.join('..',  '..', '..', 'logs', dataset_name, 'Dim_'+str(latent_dim))
    eval_ids                    = args.eval_ids

    use_gpu, mem_free = select_GPU(gpu_id=args.gpu)
    if use_gpu is not None:
        print(f"Selected GPU {use_gpu} for evaluation")
    else:
        print("Evaluating on CPU")

    # Log number of relevant axes
    log_dir = os.path.join('logs', dataset_name, mode, 'Dim_'+str(latent_dim))
    os.makedirs(log_dir, exist_ok=True)
    rel_axis_stat = np.zeros(len(eval_ids))
    log_filepath = os.path.join(log_dir, 'rel_axis_stat.txt')
    log_fileptr = open(log_filepath, 'w')

    run_index = 0
    for run_id in eval_ids:

        tf.keras.backend.clear_session()

        generated_image_dir = os.path.join(log_dir, 'run_id_' + str(run_id))
        os.makedirs(generated_image_dir, exist_ok=True)

        # Produces encoded data and its associated statistics
        if mode=='reconstruction':
            latent_vectors, num_rel_axes = \
                get_encoded_statistics.get_encoded_data(dataset_name, run_id, basedir, latent_dim, encoder_use_batch_norm,
                                                        decoder_use_batch_norm, num_filter, fid_samples, batch_size, generated_image_dir, mode, perc_explained_var)
        elif mode == 'generation':
            latent_vectors, num_rel_axes = \
                get_encoded_statistics.get_encoded_data(dataset_name, run_id, basedir, latent_dim, encoder_use_batch_norm,
                                                        decoder_use_batch_norm, num_filter, fid_samples, batch_size, generated_image_dir, mode, perc_explained_var)

        rel_axis_stat[run_index] = num_rel_axes
        run_index += 1
        log_fileptr.write('Relevant axes for run ID ' + str(run_id) + ' using estimated variance : ' +
                          str(num_rel_axes) + ' explaining ' + str(perc_explained_var) + ' variability' + '\n')
        log_fileptr.flush()

        ###############################################
        # Generate samples using encoded representation
        ###############################################
        gen_images_array = get_generated_samples.get_generated_data(dataset_name, run_id, basedir, latent_dim, decoder_use_batch_norm, num_filter, latent_vectors, batch_size)
        image_dir = os.path.join(generated_image_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        if dataset_name=='MNIST':
            for image_index in range(gen_images_array.shape[0]):
                save_image = gen_images_array[image_index, :, :, :]
                gen_image_path = os.path.join(image_dir, "generated_image_" + str(image_index) + ".jpg")
                imageio.imwrite(gen_image_path, save_image)
        else:
            save_nparray_path = os.path.join(image_dir, 'generated_images.npy')
            np.save(save_nparray_path, gen_images_array)


    # save the rel axes statistics
    np_save_path = os.path.join(log_dir, 'rel_axis_stat.npy')
    np.save(np_save_path, rel_axis_stat)

    avg_rel_axes = np.mean(rel_axis_stat)
    stddev_fid_axes = np.std(rel_axis_stat)
    log_fileptr.write('Average number of relative axes over ' + str(len(eval_ids)) + ' models....' + '\n')
    log_fileptr.write(str(np.around(avg_rel_axes, 2)) + ' \u00B1 ' + str(np.around(stddev_fid_axes, 2)))
    log_fileptr.flush()
    log_fileptr.close()