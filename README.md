# Deep Image Prior Inpainting Project

This project implements image inpainting using the Deep Image Prior (DIP) framework, a learning-free approach to restore corrupted images. It removes text overlays or fills missing regions in images, achieving high-quality results without pre-training. The implementation is based on the original DIP paper by Ulyanov et al. (2018).

## Features

- Removes text overlays and fills large missing regions in images
- Uses a skip network architecture (skip_depth6) with configurable depth and input channels
- Supports various regularization techniques to prevent overfitting
- Tested on benchmark images like `kate.png`, `peppers.png`, and `library.png`

##Install Dependencies
Ensure you have Python 3.6+ and PyTorch installed.

##Download the Dataset

Place your images and masks in the data/inpainting/ directory. Example files:

- `kate.png` and `kate_mask.png`
- `peppers.png` and `peppers_mask.png`
- `library.png` and `library_mask.png`

##Project Structure

- `inpainting.py`: Main script for running the inpainting process
- `models/skip.py`: Defines the skip network architecture
- `utils/inpainting_utils.py`: Utility functions for image processing
- `data/inpainting/`: Directory for input images and masks
- `output.png`: Default output file for inpainted images

##Acknowledgments

This project was developed as part of a university course on computer vision, with thanks to the open-source DIP framework.
