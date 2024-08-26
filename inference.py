"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import os
from pathlib import Path
import json
import glob
print(os.getcwd())
print(os.environ['HOME'])
from resources.pipeline import inference

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
# RESOURCE_PATH = Path("resources")

print('the input path is ', INPUT_PATH)

def run():
    # Read the input
    input_prostatectomy_tissue_whole_slide_path = get_img_path(
        location=INPUT_PATH / "images/prostatectomy-wsi",
    )

    # input_prostatectomy_tissue_mask = load_image_file_as_array(
    #     location=INPUT_PATH / "images/prostatectomy-tissue-mask",
    # )

    _show_torch_cuda_info()

    # Predictions
    output_overall_survival_years, case_id = inference(input_prostatectomy_tissue_whole_slide_path)
    # OUTPUT_FOLDER = OUTPUT_PATH / case_id
    # os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "overall-survival-years.json",
        content=output_overall_survival_years
    )
    
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def get_img_path(*, location):
    print('image location', location)
    input_files = glob.glob(str(location / "*.tif"))
    return input_files[0]


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
