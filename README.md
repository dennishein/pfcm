# Poisson Flow Consistency Models 

This repository contains the codebase for [Poisson Flow Consistency Models](https://arxiv.org/abs/2402.08159), to appear in IEEE TMI. This repo heavily builds on [consistency models](https://github.com/openai/consistency_models/blob/main/.gitignore) and [PFGM++](https://github.com/Newbeeer/pfgmpp). 

# Pre-trained models

Here are the download links for each the models in Table IV:

 * EDM: [edm.pt](https://drive.google.com/file/d/1zrZ5LytOxASjimqB_BfcKsJyhuGXgOyv/view?usp=share_link)
 * PFGM++: [pfgmpp_2048.pt](https://drive.google.com/file/d/1CHpDSH5i9GWjwdWf8hgvEjDHDbm-LeIf/view?usp=share_link)
 * CM: [cd.pt](https://drive.google.com/file/d/1B9JuKSZhNhZXIwdHJFy3OR2Z1yFy1tqo/view?usp=share_link****)
 * PFCM: [pfcd_128.pt](https://drive.google.com/file/d/1YoKtSi5_S6w8RvZtPO5dIc9n4lxDtZmG/view?usp=share_link)

# Dependencies

To install with Docker, run the following commands:
```sh
cd docker && make build 
```

# Model training and sampling

We provide examples of EDM training, consistency distillation, consistency training, single-step generation, and multistep generation in [scripts/launch.sh](scripts/launch.sh).

# Evaluations

To compare different generative models, we use FID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples stored in `.npz` (numpy) files. One can evaluate samples with [cm/evaluations/evaluator.py](evaluations/evaluator.py) in the same way as described in [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with reference dataset batches provided therein.

# Citation

If you find this method and/or code useful, please consider citing

```bibtex
@misc{hein2025pfcmpoissonflowconsistency,
      title={PFCM: Poisson flow consistency models for low-dose CT image denoising}, 
      author={Dennis Hein and Grant Stevens and Adam Wang and Ge Wang},
      year={2025},
      eprint={2402.08159},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2402.08159}, 
}
```
