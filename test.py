#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.cow import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    labels = get_label_names(_config['dataset'])

    class_dice = {}
    class_iou = {}

    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], _config['test_label'])):
            continue

        _log.info(f'Test Class: {label_name}')

        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val

        with torch.no_grad():
            model.eval()

            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):

                # Unpack query data.
                query_image = [sample['image'][i].float().cuda() for i in
                               range(sample['image'].shape[0])]
                query_label = sample['label'].long()
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # Compute output.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):  # n_part = 3
                    support_image_s = [support_image[sub_chunck]]
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]
                    query_pred_s = []
                    for i in range(query_image_s.shape[0]):
                        _pred_s, _, _, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
                                           train=False)
                        query_pred_s.append(_pred_s)
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()

                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                scores.record(query_pred, query_label)

                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')

                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')

    def dict_Avg(Dict):
        A = sum(Dict.values()) / len(Dict)
        return A
    res = dict_Avg(class_dice)

    _log.info(f'Total Mean Dice: {res}')
    _log.info(f'End of validation.')
    return 1
