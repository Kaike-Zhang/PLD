# PLD

This paper has been accepted by WWW 2025. [Link to the paper on Arxiv][https://arxiv.org/pdf/2410.22844](https://arxiv.org/pdf/2502.00348)

## Authors
- **Kaike Zhang**
- Qi Cao
- Yunfan Wu
- Fei Sun
- Huawei Shen
- Xueqi Cheng

## Abstract
While implicit feedback is foundational to modern recommender systems, factors such as human error, uncertainty, and ambiguity in user behavior inevitably introduce significant noise into this feedback, adversely affecting the accuracy and robustness of recommendations. To address this issue, existing methods typically aim to reduce the training weight of noisy feedback or discard it entirely, based on the observation that noisy interactions often exhibit higher losses in the overall loss distribution. However, we identify two key issues: (1) there is a significant overlap between normal and noisy interactions in the overall loss distribution, and (2) this overlap becomes even more pronounced when transitioning from pointwise loss functions (e.g., BCE loss) to pairwise loss functions (e.g., BPR loss). This overlap leads traditional methods to misclassify noisy interactions as normal, and vice versa. To tackle these challenges, we further investigate the loss overlap and find that for a given user, there is a clear distinction between normal and noisy interactions in the user's personal loss distribution. Based on this insight, we propose a resampling strategy to Denoise using the user's Personal Loss distribution, named PLD, which reduces the probability of noisy interactions being optimized. Specifically, during each optimization iteration, we create a candidate item pool for each user and resample the items from this pool based on the user's personal loss distribution, prioritizing normal interactions. Additionally, we conduct a theoretical analysis to validate PLD's effectiveness and suggest ways to further enhance its performance. Extensive experiments conducted on three datasets with varying noise ratios demonstrate PLD's efficacy and robustness.

## Environment
- python >= 3.8
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0
- torch >= 1.10.1


## Usage (Quick Start)
1. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired backbone model and dataset:

    ```bash
    python main.py --model=<backbone model> --dataset=<dataset>
    ```

   Replace `<backbone model>` with the name of your model, and `<dataset>` with the name of your dataset.


## Citation
If you find our work useful, please cite our paper using the following BibTeX:

```bibtex
@article{zhang2025personalized,
  title={Personalized Denoising Implicit Feedback for Robust Recommender System},
  author={Zhang, Kaike and Cao, Qi and Wu, Yunfan and Sun, Fei and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2502.00348},
  year={2025}
}


