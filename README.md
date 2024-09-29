# show-attend-and-tell
<!-- markdownlint-disable MD033 MD045 -->

<p align="center">
  <img src=nice_18.png height="300"/>
</p>

Implementation of [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044).

The model has been trained on double digit MNIST dataset obtained from [here](https://github.com/shaohua0116/MultiDigitMNIST). To train the model from scratch download the dataset (see below) and check [this notebook](https://github.com/souvikshanku/show-attend-and-tell/blob/main/train_and_viz.ipynb).

```bash
# Download, unzip and move
cd show-attend-and-tell
gdown https://drive.google.com/uc?id=1NMLh34zDjrI-bOIK6jgLJAqRrUY3uETC
unzip double_mnist.zip
mv labels.csv data/labels.csv
```
