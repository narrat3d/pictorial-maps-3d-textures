# Texture inpainting for human figures (from pictorial maps)

This is a fork from https://github.com/saic-vul/coordinate_based_inpainting (MIT License, Copyright by Samsung Electronics Co., Ltd.)

## Usage

### Training

* The code for training the network was not provided by the original authors.

### Inference 

* Prepare our data with texture_inpainting.py (see folder '0_data_processing').
* Run infer_sample.py to generate textures.

## Notes 

* Attention: A total PC power-off can occur during inference sometimes. 
* The original model is trained with humans from photos, therefore textures may look quite realistic (e.g., faces).
