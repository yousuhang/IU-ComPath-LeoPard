import math
import time
import numpy as np
import cv2
import multiprocessing as mp
from resources.wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, \
    Contour_Checking_fn


def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def scaleHolesDim(contours, scale):
    return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]


def isInHoles(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0] + patch_size / 2, pt[1] + patch_size / 2), False) > 0:
            return 1

    return 0


def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
    if cont_check_fn(pt):
        if holes is not None:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0


def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
    if isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
        return coord
    else:
        return None


def segmentTissue(WSI_object, seg_params={'seg_level': 0, 'sthresh': 8, 'sthresh_up': 255, 'mthresh': 7, 'close': 4,
                                          'use_otsu': False},
                  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8}, ref_patch_size=512, exclude_ids=[],
                  keep_ids=[]):
    """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
    """

    img = np.array(
        WSI_object.read_region((0, 0), seg_params['seg_level'], WSI_object.level_dimensions[seg_params['seg_level']]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], seg_params['mthresh'])  # Apply median blurring

    # Thresholding
    if seg_params['use_otsu']:
        _, img_otsu = cv2.threshold(img_med, 0, seg_params['sthresh_up'], cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, seg_params['sthresh'], seg_params['sthresh_up'], cv2.THRESH_BINARY)

    # Morphological closing
    if seg_params['close'] > 0:
        kernel = np.ones((seg_params['close'], seg_params['close']), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    scale = WSI_object.level_downsamples[seg_params['seg_level']]

    scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale ** 2))
    filter_params = filter_params.copy()
    filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
    filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    if filter_params: foreground_contours, hole_contours = filter_contours(contours, hierarchy,
                                                                           filter_params)  # Necessary for filtering out artifacts

    WSI_object.contours_tissue = scaleContourDim(foreground_contours, scale)
    WSI_object.holes_tissue = scaleHolesDim(hole_contours, scale)

    # exclude_ids = [0,7,9]
    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(WSI_object.contours_tissue))) - set(exclude_ids)

    WSI_object.contours_tissue = [WSI_object.contours_tissue[i] for i in contour_ids]
    WSI_object.holes_tissue = [WSI_object.holes_tissue[i] for i in contour_ids]


def process_contour(WSI_Object, cont, contour_holes, patch_level, patch_size=256, step_size=256,
                    contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
    start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (
        0, 0, WSI_Object.level_dimensions[patch_level][0], WSI_Object.level_dimensions[patch_level][1])

    patch_downsample = int(WSI_Object.level_downsamples[patch_level])
    ref_patch_size = (patch_size * patch_downsample, patch_size * patch_downsample)

    img_w, img_h = WSI_Object.level_dimensions[0]
    if use_padding:
        stop_y = start_y + h
        stop_x = start_x + w
    else:
        stop_y = min(start_y + h, img_h - ref_patch_size[1] + 1)
        stop_x = min(start_x + w, img_w - ref_patch_size[0] + 1)

    print("Bounding Box:", start_x, start_y, w, h)
    print("Contour Area:", cv2.contourArea(cont))

    if bot_right is not None:
        stop_y = min(bot_right[1], stop_y)
        stop_x = min(bot_right[0], stop_x)
    if top_left is not None:
        start_y = max(top_left[1], start_y)
        start_x = max(top_left[0], start_x)

    if bot_right is not None or top_left is not None:
        w, h = stop_x - start_x, stop_y - start_y
        if w <= 0 or h <= 0:
            print("Contour is not in specified ROI, skip")
            return {}, {}
        else:
            print("Adjusted Bounding Box:", start_x, start_y, w, h)

    if isinstance(contour_fn, str):
        if contour_fn == 'four_pt':
            cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'four_pt_hard':
            cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)
        elif contour_fn == 'center':
            cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size[0])
        elif contour_fn == 'basic':
            cont_check_fn = isInContourV1(contour=cont)
        else:
            raise NotImplementedError
    else:
        assert isinstance(contour_fn, Contour_Checking_fn)
        cont_check_fn = contour_fn

    step_size_x = step_size * patch_downsample
    step_size_y = step_size * patch_downsample

    x_range = np.arange(start_x, stop_x, step=step_size_x)
    y_range = np.arange(start_y, stop_y, step=step_size_y)
    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

    num_workers = mp.cpu_count()
    if num_workers > 4:
        num_workers = 4
    pool = mp.Pool(num_workers)

    iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
    results = pool.starmap(process_coord_candidate, iterable)
    pool.close()
    results = np.array([result for result in results if result is not None])

    print(f'Extracted {len(results)} coordinates.')

    if len(results) > 0:
        results = results.astype('int32')
        asset_dict = {'coords': results}

        attr = {'patch_size': patch_size,  # To be considered...
                'patch_level': patch_level,
                'downsample': WSI_Object.level_downsamples[patch_level],
                'downsampled_level_dim': tuple(np.array(WSI_Object.level_dimensions[patch_level])),
                'level_dimensions': WSI_Object.level_dimensions[patch_level],
                'name': WSI_Object.name}

        attr_dict = {'coords': attr}
        return asset_dict, attr_dict

    else:
        return {}, {}


def process_contours(WSI_object, patch_level=0, patch_size=256, step_size=256, contour_fn='four_pt', use_padding=True,
                     top_left=None, bot_right=None):
    #    save_path_hdf5 = os.path.join(save_path, str(WSI_object.name) + '.h5')
    print("Creating patches for: ", WSI_object.name, "...", )
    elapsed = time.time()
    n_contours = len(WSI_object.contours_tissue)
    print(f"======Total {n_contours} contours to process======")
    fp_chunk_size = math.ceil(n_contours * 0.05)
    init = True
    assest_total_dict, attr_total_dict = None, None
    for idx, cont in enumerate(WSI_object.contours_tissue):
        if (idx + 1) % fp_chunk_size == fp_chunk_size:
            print(f'Processing contour {idx}/{n_contours}')

        asset_dict, attr_dict = process_contour(WSI_object, cont, WSI_object.holes_tissue[idx], patch_level, patch_size,
                                                step_size, contour_fn=contour_fn, use_padding=use_padding,
                                                top_left=top_left, bot_right=bot_right)

        if len(asset_dict) > 0:
            if init:
                assest_total_dict = asset_dict
                attr_total_dict = attr_dict
                init = False
            else:
                for key in asset_dict.keys():
                    assest_total_dict[key] = np.concatenate((assest_total_dict[key], asset_dict[key]), axis=0)

    return assest_total_dict, attr_total_dict


def create_contour(coord_list):
    return np.array([[[int(float(coord.attributes['X'].value)),
                       int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype='int32')


def in_patch(coordinate_candidate, coordinate_conditioned, patch_level_conditioned=3, patch_size_conditioned=224):
    patch_range_conditioned = patch_size_conditioned * (4 ** patch_level_conditioned)
    diff_location = coordinate_candidate - coordinate_conditioned
    if 0 <= diff_location[0] < patch_range_conditioned and 0 <= diff_location[1] < patch_range_conditioned:
        return True
    else:
        return False
#TODO add index selection in the training file and save. 
