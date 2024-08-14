import timm
import torch
from resources.utils.constants import MODEL2CONSTANTS
from resources.utils.transform_utils import get_eval_transforms

# if f'{os.environ["HOME"]}/OneDrive/LEOPARD-Submission' == f'{os.getcwd()}':
#     UNI_PATH = f'{os.getcwd()}/resources/extractor_n_weights/UNI/pytorch_model.bin'
# else:
#     UNI_PATH = f'{os.environ["HOME"]}/CLAM/extractor_n_weights/UNI/pytorch_model.bin'

UNI_PATH = "/opt/app/resources/extractor_n_weights/UNI/pytorch_model.bin"


def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'uni_v1':
        # HAS_UNI, UNI_CKPT_PATH = has_UNI()
        # assert HAS_UNI, 'UNI is not available'
        UNI_CKPT_PATH = UNI_PATH
        model = timm.create_model("vit_large_patch16_224",
                                  init_values=1e-5,
                                  num_classes=0,
                                  dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size=target_img_size)

    return model, img_transforms