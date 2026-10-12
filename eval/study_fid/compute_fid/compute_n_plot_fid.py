import os
import numpy as np
import argparse
from config.local_config import configurations
import get_fid
try:
    import nvidia_smi
    HAS_NVIDIA_SMI = True
except ImportError:
    HAS_NVIDIA_SMI = False
import tensorflow as tf


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


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--config_id", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--gen_type", type=str, required=True, help="Can be one of the following options: generation, reconstruction")
    parser.add_argument("--eval_ids", type=int, nargs='+', default=[1], help="Run IDs to evaluate, e.g. --eval_ids 1 2 3 4 5")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index to use")
    args = parser.parse_args()

    latent_dim = args.latent_dim
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
    fid_samples                 = config['fid_samples']
    eval_ids                    = args.eval_ids
    fidstat_basedir             = 'fid_stats'
    if dataset_name=='MNIST':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_mnist.npz')
        gen_samples_np = False
    if dataset_name=='CelebA':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_celeba.npz')
        gen_samples_np = True
    if dataset_name=='CIFAR10':
        fid_stat_path = os.path.join(fidstat_basedir, 'fid_stats_cifar10_train.npz')
        gen_samples_np = True

    # no-op when valid reference statistics (with activations) already exist
    if dataset_name in ('MNIST', 'CIFAR10'):
        import prepare_fid_stats
        if dataset_name == 'MNIST':
            prepare_fid_stats.prepare_mnist_stats(fidstat_basedir, sample_count=fid_samples)
        elif dataset_name == 'CIFAR10':
            prepare_fid_stats.prepare_cifar10_stats(fidstat_basedir, sample_count=fid_samples)

    use_gpu, mem_free = select_GPU(gpu_id=args.gpu)
    if use_gpu is not None:
        print(f"Selected GPU {use_gpu} for FID computation")
    else:
        print("Computing FID on CPU")

    fid_scores_stat = np.zeros(len(eval_ids))
    precision_stat = np.zeros(len(eval_ids))
    recall_stat = np.zeros(len(eval_ids))
    log_dir = os.path.join('logs', dataset_name, mode, 'Dim_'+str(latent_dim))
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'fid_stat.txt')
    log_fileptr = open(log_filepath, 'w')


    run_index = 0
    for run_id in eval_ids:

        generated_image_array_dir = os.path.join('..', 'generate_samples', 'logs', dataset_name, mode, 'Dim_'+str(latent_dim), 'run_id_' + str(run_id), 'images')

        if gen_samples_np:
            generated_image_array_path = os.path.join(generated_image_array_dir, "generated_images.npy")
            generated_image = np.load(generated_image_array_path).astype(np.float32)
            assert generated_image.shape[0]==fid_samples, "Number of generated samples are not sufficient...."
        else:
            generated_image = generated_image_array_dir

        # compute the FID score for each annulus
        model_fid_score, precision, recall = get_fid.calculate_fid_given_paths(generated_image, fid_stat_path, None, gen_samples_np=gen_samples_np)

        # save the fid score
        fid_scores_stat[run_index] = model_fid_score
        precision_stat[run_index] = precision
        recall_stat[run_index] = recall
        run_index += 1
        result_line = 'FID score for run ID ' + str(run_id) + ' is ' + str(np.around(model_fid_score, 2)) + \
                      ' | precision ' + str(np.around(precision, 3)) + ' | recall ' + str(np.around(recall, 3))
        print(result_line, flush=True)  # stdout is a pipe under `!python`, so it is block-buffered
        log_fileptr.write(result_line + '\n')
        log_fileptr.flush()

    # fid score statistics
    avg_fid_score = np.mean(fid_scores_stat)
    stddev_fid_score = np.std(fid_scores_stat)

    # save the fid score statistics
    log_fileptr.write('Average of the FID stat over ' + str(len(eval_ids)) + ' models....' + '\n')
    log_fileptr.write(str(np.around(avg_fid_score, 2)) + ' \u00B1 ' + str(np.around(stddev_fid_score, 2)))
    log_fileptr.flush()
    log_fileptr.close()

    np_save_path = os.path.join(log_dir, 'fid_scores_stat.npy')
    np.save(np_save_path, fid_scores_stat)
    np.save(os.path.join(log_dir, 'precision_stat.npy'), precision_stat)
    np.save(os.path.join(log_dir, 'recall_stat.npy'), recall_stat)