import glob

import h5py
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from resources.dataset_modules import Whole_Slide_Bag_Simple
from resources.models import get_encoder, MAD_MIL_reg
from resources.models import MIL_fc_reg, MIL_fc_reg_att, MIL_fc_reg_top_k_att
from resources.models.model_clam import CLAM_SB
from resources.wsi_core.WholeSlideImage_mod import WholeSlideImageOpenslide
from resources.wsi_core import segmentTissue, process_contours
# from multiprocessing import cpu_count
# other imports
import os
import time
from tqdm import tqdm
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def segment(wsi_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        raise NotImplementedError
    # Segment
    else:
        segmentTissue(wsi_object, seg_params=seg_params, filter_params=filter_params,
                      ref_patch_size=512, exclude_ids=[], keep_ids=[])

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return seg_time_elapsed


def patching(wsi_object, contour_params):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    patch_tuple = process_contours(wsi_object, **contour_params)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return patch_tuple, patch_time_elapsed


def feature_extract(wsi_object, patch_tuple, model, img_transforms, extractor_batch_size, verbose=0,
                    device='cuda', num_workers=2, pin_memory=False, hdf5_path='/output/features.h5'):
    ### Start extraction timer
    start_time = time.time()

    # Create a dataset object that handles the whole-slide image (WSI) data
    dataset = Whole_Slide_Bag_Simple(patch_coordinate_tuple=patch_tuple, wsi=wsi_object,
                                     img_transforms=img_transforms)

    # Set up DataLoader parameters based on whether GPU (cuda) is used
    # print(f'num_worker is {cpu_count()}')
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if device == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=extractor_batch_size, **loader_kwargs)

    # Print the number of batches to be processed if verbosity is enabled
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')

    # Open an HDF5 file to store the extracted features
    with h5py.File(hdf5_path, 'w') as hdf:
        # Initialize the HDF5 dataset
        feature_dset = None

        # Iterate over the batches of data
        for count, data in enumerate(tqdm(loader)):
            # Disable gradient calculation for inference
            # with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type=device):
                with torch.inference_mode():
                    batch = data['img']
                    # Move the batch to the appropriate device
                    batch = batch.to(device, non_blocking=True)

                    # Extract features and move them to CPU. Convert to features to numpy array
                    features = model(batch).cpu().numpy()

                    # Create the HDF5 dataset if it doesn't exist
                    if feature_dset is None:
                        # Determine the shape of the dataset
                        feature_shape = (len(loader.dataset), features.shape[1])
                        feature_dset = hdf.create_dataset('features', shape=feature_shape, dtype=np.float32)

                    # Calculate the indices for the current batch and store the features in the HDF5 file
                    start_idx = count * extractor_batch_size
                    end_idx = start_idx + features.shape[0]
                    feature_dset[start_idx:end_idx] = features

                    # Free up GPU memory after processing the batch
                    del batch

    # Free up GPU memory used by the model
    del model

    # Load the features from the HDF5 file into a PyTorch tensor and move it to the appropriate device
    with h5py.File(hdf5_path, 'r') as hdf:
        bag_feature = torch.tensor(hdf['features'][:]).to(device)

    os.remove(hdf5_path)
    # Calculate the total extraction time
    extraction_time_elapsed = time.time() - start_time

    return bag_feature, extraction_time_elapsed


def load_model(model_params=None, device='cuda', ckpt_path=None):
    #
    if model_params is None:
        model_params = {'dropout': 0.25,
                        'n_classes': 1,
                        'embed_dim': 1024,
                        'model_size': 'small',
                        'model_type': 'mil_reg',
                        'top_k': 5,
                        'gate': False,
                        'cpkt_path': None
                        }
    print('Init Model')

    model = None  # pred_define regression model
    model_dict = {'dropout': model_params['dropout'],
                  'n_classes': model_params['n_classes'],
                  'embed_dim': model_params['embed_dim']}

    if model_params['model_size'] is not None and model_params['model_type'] in ['clam_sb', 'clam_mb']:
        model_dict.update({'size_arg': model_params['model_size']})

    if model_params['model_type'] in ['mil_reg', 'mil_reg_att',
                                      'mil_reg_topk_att', 'clam_sb', 'mad_mil_reg']:  # args.model_type == 'mil_reg'
        if model_params['model_type'] == 'mil_reg':
            model_dict.update({'top_k': model_params['top_k']})
            model = MIL_fc_reg(**model_dict)
        elif model_params['model_type'] == 'mil_reg_att':
            model_dict.update({'gate': model_params['gate']})
            model = MIL_fc_reg_att(**model_dict)
        elif model_params['model_type'] == 'mil_reg_topk_att':
            model_dict.update({'gate': model_params['gate']})
            model_dict.update({'top_k': model_params['top_k']})
            model = MIL_fc_reg_top_k_att(**model_dict)
        elif model_params['model_type'] == 'clam_sb':
            model_dict.update({'gate': model_params['gate']})
            if model_params['inst_loss'] == 'svm':
                from topk.svm import SmoothTop1SVM
                instance_loss_fn = SmoothTop1SVM(n_classes=2)
                if device == 'cuda':
                    instance_loss_fn = instance_loss_fn.cuda()
            else:
                weights = [0.25, 0.75]
                class_weights = torch.FloatTensor(weights).cuda()
                instance_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            model_dict.update({'instance_loss_fn': instance_loss_fn})
            model_dict.update({'k_sample': model_params['k_sample']})
            model_dict.update({'subtyping': model_params['subtyping']})
            model = CLAM_SB(**model_dict)
        elif model_params['model_type'] == 'mad_mil_reg':
            model_dict.update({'size_arg': model_params['model_size']})
            model_dict.update({'gate': model_params['gate']})
            model_dict.update({'n_heads': model_params['n_heads']})
            model = MAD_MIL_reg(**model_dict)
        else:
            raise NotImplementedError

        # print_network(model)
    if ckpt_path == None:
        ckpt = torch.load(model_params['cpkt_path'])
    else:
        ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt, strict=True)

    model.to(device)
    model.eval()

    return model


def contour_filering_from_low_level_selection(low_level_patch_tuple, high_level_patch_tuple, index_list):
    low_level_patch_coords = low_level_patch_tuple[0]['coords']
    high_level_patch_coords = high_level_patch_tuple[0]['coords']
    selected_coords_of_the_slide = low_level_patch_coords[index_list]
    coords_selected_list = []
    for coordinate_conditioned in selected_coords_of_the_slide[:]:
        # print(coordinate_conditioned)
        coords_dif = np.array(high_level_patch_coords) - coordinate_conditioned
        coords_dif_filtered_ind = np.sum(coords_dif >= 0, axis=1) > 1
        coords_dif_2 = np.array(high_level_patch_coords) - coordinate_conditioned - 224 * 4 ** 3
        coords_dif_2_filtered_ind = np.sum(coords_dif_2 < 0, axis=1) > 1
        coords_selected_ind = coords_dif_filtered_ind * coords_dif_2_filtered_ind
        coords_selected = high_level_patch_coords[coords_selected_ind]
        # print(coords_selected.shape[0])
        coords_selected_list.append(coords_selected)
    asset_dict = {'coords': np.concatenate(coords_selected_list)}
    attr_dict = high_level_patch_tuple[1]
    return asset_dict, attr_dict


def save_dict_as_json(dict_var, save_path):
    # create json object from dictionary
    json_content = json.dumps(dict_var)

    # open file for writing, "w"
    f = open(save_path, "w")

    # write json object to file
    f.write(json_content)

    # close file
    f.close()


def interence_pipeline(config, WSI_path, extractor=None, img_transforms=None, predictor_low_level=None, regressor=None,
                       save_params=False):
    result_dict = {}
    total_time_elasp = 0
    # 1. load WSI

    whole_slide_image = WholeSlideImageOpenslide(path=WSI_path)
    print(f'Processing Slide {whole_slide_image.name}')
    slide_file_size_gb = os.path.getsize(WSI_path) / 10**9
    print(f'The Slide file size is {slide_file_size_gb:.2f} GB')

    # 2. set up low resolution segmentation params

    seg_params = config['seg_params']

    seg_level_in_setting = seg_params.pop('seg_level')

    if seg_level_in_setting == -1:  # update seg_level when using built-in downsample function
        seg_level = whole_slide_image.get_best_level_for_downsample(64)
    else:
        seg_level = seg_level_in_setting

    seg_params.update({'seg_level': seg_level})

    # 3. set up contour filtering params (for holes in the patch)

    filter_params = config['filter_params']

    # 4. perform low resolution segmentation

    seg_time_elapsed = segment(whole_slide_image, seg_params, filter_params)

    result_dict.update({'seg_time_elapsed': seg_time_elapsed})
    total_time_elasp += seg_time_elapsed
    print(f"Segmentation took {seg_time_elapsed} seconds")

    # 5. set up contour processing params two parames one for low level one for high level

    contour_params_low_level = config['contour_params_low_level']
    contour_params = config['contour_params']

    # 6. perform contour processing aka patching process, low level and high level
    patch_tuple_low_level, patch_time_elasped_low_level = patching(whole_slide_image, contour_params_low_level)
    patch_tuple, patch_time_elasped = patching(whole_slide_image, contour_params)
    # patch_tuple = (coordinate_dict, attribution_dict)
    result_dict.update({'patch_time_elasped_low_level': patch_time_elasped_low_level})
    result_dict.update({'patch_time_elasped': patch_time_elasped})
    total_time_elasp += patch_time_elasped + patch_time_elasped_low_level
    print(f"Patching took {patch_time_elasped} seconds")

    # 7. feature extraction for low level the setting of the extraction param is the same for two extractors

    extraction_params = config['extraction_params']
    extractor_name = extraction_params.pop('extractor_name')
    extractor_input_size = extraction_params.pop('extractor_input_size')
    extraction_params.update({'device': device.type,
                              # 'hdf5_path': f"{config['main_params']['save_folder']}/features.h5"
                              })

    if extractor is None:
        extractor, img_transforms = get_encoder(extractor_name, target_img_size=extractor_input_size)
    extractor.to(device.type)
    bag_feature_low_level, extraction_time_elaps_low_level = feature_extract(whole_slide_image, patch_tuple_low_level,
                                                                             extractor, img_transforms,
                                                                             **extraction_params)
    extraction_params.update({'extractor_name': extractor_name, 'extractor_input_size': extractor_input_size})
    # bag_feature should have dimensions (Number_of_Patches, Feature_Dim)
    result_dict.update({'extraction_time_elaps_low_level': extraction_time_elaps_low_level})
    total_time_elasp += extraction_time_elaps_low_level
    print(
        f"Low level Feature extraction using extractor {extraction_params['extractor_name']} took {extraction_time_elaps_low_level} seconds")
    # 8. regression step for low level and select index
    ### Start regression timer
    start_time = time.time()

    predictor_low_level_params = config['predictor_low_level_params']

    if predictor_low_level is None:
        predictor_low_level = load_model(predictor_low_level_params, device=device.type)

    sample_size = round(bag_feature_low_level.size()[0] * predictor_low_level_params['percentage'])
    result_dict.update(
        {'patch_selection_number_low_level': sample_size, 'patch_number_low_level': bag_feature_low_level.size()[0]})

    if predictor_low_level_params['n_classes'] > 1:
        with torch.no_grad():
            _, _, _, bag_low_level_output, _ = predictor_low_level(bag_feature_low_level)
        bag_low_level_output = torch.softmax(bag_low_level_output, dim=1)
    else:
        with torch.no_grad():
            _, _, _, bag_low_level_output = predictor_low_level(bag_feature_low_level)
    selected_index = torch.topk(bag_low_level_output[0, :], sample_size, dim=0)[1]
    selected_index_list = selected_index.cpu().numpy().tolist()
    selected_patch_tuple = contour_filering_from_low_level_selection(patch_tuple_low_level, patch_tuple,
                                                                     selected_index_list)
    result_dict.update(
        {'patch_number_before_selection': len(patch_tuple[0]['coords']),
         'patch_number_after_selection': len(selected_patch_tuple[0]['coords'])})
    patch_selection_time_elasped = time.time() - start_time
    result_dict.update({'patch_selection_time_elasped': patch_selection_time_elasped})
    total_time_elasp += patch_selection_time_elasped

    print(
        f"Patch Selection took {patch_selection_time_elasped} seconds")

    # 9. feature extraction

    extraction_params = config['extraction_params']
    extractor_name = extraction_params.pop('extractor_name')
    extractor_input_size = extraction_params.pop('extractor_input_size')
    extraction_params.update({'device': device.type,
                              # 'hdf5_path': f"{config['main_params']['save_folder']}/features.h5"
                              })

    if extractor is None:
        extractor, img_transforms = get_encoder(extractor_name, target_img_size=extractor_input_size)
    extractor.to(device.type)
    bag_feature, extraction_time_elaps = feature_extract(whole_slide_image, selected_patch_tuple, extractor,
                                                         img_transforms,
                                                         **extraction_params)
    extraction_params.update({'extractor_name': extractor_name, 'extractor_input_size': extractor_input_size})
    whole_slide_image.features.update({'extractor': extractor_name, 'feature': bag_feature.cpu().numpy()})
    # bag_feature should have dimensions (Number_of_Patches, Feature_Dim)
    result_dict.update({'extraction_time_elaps': extraction_time_elaps})
    total_time_elasp += extraction_time_elaps
    print(
        f"Feature extraction using extractor {extraction_params['extractor_name']} took {extraction_time_elaps} seconds")
    # 5. regression step
    ### Start regression timer
    start_time = time.time()

    regression_params = config['regression_params']
    # if regressor is None:
    bag_log_risk = 0
    for pt_file_path in config['regressor_cpkt_paths']:
        regressor = load_model(regression_params, device=device.type, ckpt_path=pt_file_path)
        with torch.no_grad():
            bag_log_risk_temp, _, _, _ = regressor(bag_feature)
        del regressor
        bag_log_risk += bag_log_risk_temp
    bag_log_risk /= len(config['regressor_cpkt_paths'])

    whole_slide_image.log_risk.update({'regressor': regression_params['model_type'],
                                       'predicted_log_risk': bag_log_risk.item()})
    result_dict.update({'predicted_log_risk': whole_slide_image.log_risk['predicted_log_risk']})
    regression_time_elasped = time.time() - start_time
    result_dict.update({'regression_time_elasped': regression_time_elasped})
    total_time_elasp += regression_time_elasped
    print(f"Regression using regressor {regression_params['model_type']} took {regression_time_elasped} seconds")
    # 6. save the predicted risk
    # will add more to save
    result_dict.update({'case_id': whole_slide_image.name})
    # save_folder = config['main_params']['save_folder']
    # save_path = f"{save_folder}/{result_dict['case_id']}.json"
    # save_dict_as_json(result_dict, save_path)

    # if save_params:
    #     save_dict_as_json(seg_params, f"{save_folder}/seg_params.json")
    #     save_dict_as_json(filter_params, f"{save_folder}/filter_params.json")
    #     save_dict_as_json(contour_params, f"{save_folder}/contour_params.json")
    #     save_dict_as_json(extraction_params, f"{save_folder}/extraction_params.json")
    #     save_dict_as_json(regression_params, f"{save_folder}/regression_params.json")

    return result_dict['predicted_log_risk'], extractor, img_transforms, result_dict['case_id']


def inference(wsi_path):
    config_path = 'resources/config_inference.yaml'
    config_dict = yaml.safe_load(open(config_path, 'r'))
    regressor_cpkt_paths = sorted(glob.glob(f"{config_dict['regression_params']['cpkt_folder']}/*.pt"))
    config_dict.update({'regressor_cpkt_paths': regressor_cpkt_paths})
    # config_dict['main_params'].update({'wsi_path', wsi_path})
    # main_params = config_dict['main_params']
    ### get the paths of WSIs to be evaluated
    # if main_params['eval_csv_path'] is not None:
    #     eval_df = pd.read_csv(main_params['eval_csv_path'])
    #     eval_ids = eval_df['case_id'].tolist()
    #     eval_paths = [f"{main_params['wsi_folder']}/{id}.tif" for id in eval_ids]
    # else:
    #     eval_paths = sorted(glob.glob(f"{main_params['wsi_folder']}/*.tif"))

    # if not os.path.isdir(main_params['save_folder']):
    #     os.mkdir(main_params['save_folder'])
    extractor, img_transforms, regressor = None, None, None
    # for i in range(len(eval_paths)):
    #     if i == 0:
    #         save_params = True
    #     else:
    save_params = True
    log_risk, extractor, img_transforms, case_id = interence_pipeline(config_dict, wsi_path,
                                                                                 extractor=extractor,
                                                                                 img_transforms=img_transforms,
                                                                                 regressor=regressor,
                                                                                 save_params=save_params)
    # if not os.path.isdir(main_params['save_folder']):
    #     os.mkdir(main_params['save_folder'])
    return np.exp(-1 * log_risk), case_id


