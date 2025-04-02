# Poisson Flow Consistency Models <br>

PyTorch implementation of [PFCM: Poisson flow consistency models for
low-dose CT image denoising](https://arxiv.org/abs/2402.08159) by Dennis Hein, Grant Stevens, Adam Wang, and Ge Wang (to appear in IEEE TMI). 

Abstract: X-ray computed tomography (CT) is widely used for medical diagnosis and treatment planning; however, concerns about ionizing radiation exposure drive efforts to optimize image quality at lower doses. This study introduces Poisson Flow Consistency Models (PFCM), a novel family of deep generative models that combines the robustness of PFGM++ with the efficient single-step sampling of consistency models. PFCM are derived by generalizing consistency distillation to PFGM++ through a change-of-variables and an updated noise distribution. As a distilled version of PFGM++, PFCM inherit the ability to trade off robustness for rigidity via the hyperparameter D ∈ (0, ∞). A fact that we exploit to adapt this novel generative model for the task of low-dose CT image denoising, via a “task-specific” sampler that “hijacks” the generative process by replacing an intermediate state with the low-dose CT image. While this “hijacking” introduces a severe mismatch—the noise characteristics of low-dose CT images are different from that of intermediate states in the Poisson flow process—we show that the inherent robustness of PFCM at small D effectively mitigates this issue. The resulting sampler achieves excellent performance in terms of LPIPS, SSIM, and PSNR on the Mayo low-dose CT dataset. By contrast, an analogous sampler based on standard consistency models is found to be significantly less robust under the same conditions, highlighting the importance of a tunable D afforded by our novel framework. To highlight generalizability, we show effective denoising of clinical images from a prototype photon-counting system reconstructed using a sharper kernel and at a range of energy levels.

![schematic](pfcm_process.png)

# Outline

This implementation is build heavily on  [consistency models](https://github.com/openai/consistency_models/) and [PFGM++](https://github.com/Newbeeer/pfgmpp). All models are trained on the [Mayo low-dose CT data](https://www.aapm.org/grandchallenge/lowdosect/). Trained weights for EDM, PFGM++, CM, and PFCM in Table IV are available here:
 * EDM: [edm.pt](https://drive.google.com/file/d/1zrZ5LytOxASjimqB_BfcKsJyhuGXgOyv/view?usp=share_link)
 * PFGM++: [pfgmpp_2048.pt](https://drive.google.com/file/d/1CHpDSH5i9GWjwdWf8hgvEjDHDbm-LeIf/view?usp=share_link)
 * CM: [cd.pt](https://drive.google.com/file/d/1B9JuKSZhNhZXIwdHJFy3OR2Z1yFy1tqo/view?usp=share_link****)
 * PFCM: [pfcd_128.pt](https://drive.google.com/file/d/1YoKtSi5_S6w8RvZtPO5dIc9n4lxDtZmG/view?usp=share_link)

# Dependencies

To install with Docker, run the following commands:
```sh
cd docker && make build 
```

# Model sampling


# Model training

# Processing the Mayo low-dose CT data 

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
