# LLM4CP
B. Liu, X. Liu, S. Gao, X. Cheng and L. Yang, "LLM4CP: Adapting Large Language Models for Channel Prediction," in Journal of Communications and Information Networks, vol. 9, no. 2, pp. 113-125, June 2024, doi: 10.23919/JCIN.2024.10582829. [[paper]](https://ieeexplore.ieee.org/document/10582829)
<br>

## Dependencies and Installation
- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation

### Method 1: Automatic Download (Recommended)

```bash
# Install dependencies
pip install huggingface_hub

# Download dataset
python download_data.py
```

The script will automatically download and organize the dataset into the following structure:

```
./data/
├── train/
│   ├── H_U_his_train.mat
│   └── H_U_pre_train.mat
└── test/
    ├── H_U_his_test.mat
    ├── H_U_pre_test.mat
    ├── H_D_pre_test.mat
    └── Umi/                    # Zero-shot testing scenario
        ├── H_U_his_test.mat
        ├── H_U_pre_test.mat
        └── H_D_pre_test.mat
```

### Method 2: Manual Download

The datasets can also be downloaded manually:
- [[Baidu Drive - Training]](https://pan.baidu.com/s/19DtLPftHomCb6_1V2lREtw?pwd=3gbv)
- [[Baidu Drive - Testing]](https://pan.baidu.com/s/10KzmwC1jncozOGNZ02Hlaw?pwd=sxfd)
- [HuggingFace Dataset](https://huggingface.co/datasets/liuboxun/LLM4CP-dataset)

After downloading, place the files in the `./data/` directory as shown above.

## Dataset Generation
We generate dataset via [QuaDRiGa](https://quadriga-channel-model.de/). To assist researchers in the field of channel prediction, we have provided a runnable demo file in the `data_generation` folder. For more detailed information about the QuDRiGa generator, please refer to its user documentation `uadriga_documentation_v2.8.1-0.pdf`.


## Get Started
Training and testing codes are in the current folder. 

-   The code for training is in `train.py`, while the code for test is in `test_tdd_full.py` and `test_fdd_full.py`. we also provide our pretrained model in [[Weights]](https://pan.baidu.com/s/1lysOqCyw44SGDQrH33Os5Q?pwd=nmqw).
👉 Alternative Huggingface download link:[Model weights](https://huggingface.co/liuboxun/LLM4CP).
    
-   For full shot training, run `train.py` directly (paths are pre-configured to `./data/train/`).

-   For few shot training, modify `train.py` and set `is_few=1`:
    ```python
    train_set = Dataset_Pro(train_TDD_r_path, train_TDD_t_path, is_few=1)
    ```

-   For testing, run `test_tdd_full.py` (Figure 7 results) or `test_fdd_full.py` (Figure 8 results). Paths are pre-configured to `./data/test/`.

-   For zero-shot testing, place Umi scenario data in `./data/test/Umi/` and update the paths in test scripts.

## Citation
If you find this repo helpful, please cite our paper.
```latex
@article{liu2024llm4cp,
  title={LLM4CP: Adapting Large Language Models for Channel Prediction},
  author={Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={arXiv preprint arXiv:2406.14440},
  year={2024}
```
