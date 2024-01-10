# Texture inpainting for human figures (from pictorial maps)

This is code for the article [Inferring Implicit 3D Representations from Human Figures on Pictorial Maps](https://doi.org/10.1080/15230406.2023.2224063). Visit the [main repository](https://github.com/narrat3d/pictorial-maps-3d-humans) and the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/) for more information.  
  
It is a fork from https://github.com/saic-vul/coordinate_based_inpainting (MIT License, Copyright by Samsung Electronics Co., Ltd.). 

## Usage

### Training

* The code for training the network was not provided by the original authors.

### Inference 

* Prepare our data with texture_inpainting.py (see folder '0_data_processing').
* Run infer_sample.py to generate textures.

## Notes 

* Attention: A total PC power-off can occur during inference sometimes. 
* The original model is trained with humans from photos, therefore textures may look quite realistic (e.g., faces).

© 2022-2023 ETH Zurich, Raimund Schnürer
