# DeepStyleX

**DeepStyleX** is an image transformation family of models that applies a style to the image.
This model is inspired from these two papers: 

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022)


## Getting started
```bash
python3 -m venv .venv
source .venv/bin/activate
```

To install the dependencies:
```bash
pip install -r requirements.txt
```

If it doesn't work, try
```bash
pip install torch pillow torchvision numpy tqdm
```