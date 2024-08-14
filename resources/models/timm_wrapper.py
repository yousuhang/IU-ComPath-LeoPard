import torch
import timm

RESNET50_TV_IN1K_PATH = 'resources/extractor_n_weights/resnet50_tv_in1k/pytorch_model.bin'
ckpt_path_dict = {'resnet50.tv_in1k' :RESNET50_TV_IN1K_PATH}
# class TimmCNNEncoder(torch.nn.Module):
#     def __init__(self, model_name: str = 'resnet50.tv_in1k',
#                  kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0},
#                  pool: bool = True):
#         super().__init__()
#         assert kwargs.get('pretrained', False), 'only pretrained models are supported'
#         self.model = timm.create_model(model_name, **kwargs)
#         self.model_name = model_name
#         if pool:
#             self.pool = torch.nn.AdaptiveAvgPool2d(1)
#         else:
#             self.pool = None
#
#     def forward(self, x):
#         out = self.model(x)
#         if isinstance(out, list):
#             assert len(out) == 1
#             out = out[0]
#         if self.pool:
#             out = self.pool(out).squeeze(-1).squeeze(-1)
#         return out


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k',
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': False, 'num_classes': 0},
                 pool: bool = True):
        super().__init__()
        # assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model.load_state_dict(torch.load(ckpt_path_dict[model_name], map_location='cpu'), strict=False)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out