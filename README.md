# Touch-R1 · Project website

Official website: https://laiyingxin2.github.io/Touch-R1

GitHub Pages redirects the short display URL to the canonical directory URL ending in `/`; the page metadata consistently uses that canonical URL.

Paper: https://arxiv.org/abs/2605.27154

The previous URL https://laiyingxin2.github.io/Projects/ is retained as a redirect.

This repository contains the **website source**, not a release of the model training code or dataset.

## Research resources

- Dataset: https://huggingface.co/datasets/laiyingxin/TouchReason
- Model: https://huggingface.co/laiyingxin/Touch-R1
- Data, research code, and model weights: coming soon.

## Citation

```bibtex
@misc{lai2026touchr1reinforcingtouchreasoning,
  title         = {{Touch-R1}: Reinforcing Touch Reasoning in {MLLMs}},
  author        = {Yingxin Lai and Yafei Zhou and Fucai Zhu and Siyu Zhu and Weihao Yuan},
  year          = {2026},
  eprint        = {2605.27154},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2605.27154}
}
```

## Local preview

```sh
python3 -m http.server 8000 --directory docs
```

Visit http://localhost:8000/. No build step or package installation is needed.

## Deployment

GitHub Pages is deployed from `docs/` by `.github/workflows/static.yml`.
Set Pages → Source to GitHub Actions. Push to `main` to deploy.
