import torch
from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []

	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))#, interpolation=transforms.InterpolationMode.NEAREST
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms = transforms.Compose(trsforms)

	return trsforms

