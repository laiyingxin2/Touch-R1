# Touch-R1 · Project website

Official website: https://laiyingxin2.github.io/Touch-R1/

Paper: https://arxiv.org/abs/2605.27154

The previous URL https://laiyingxin2.github.io/Projects/ is retained as a redirect.

This repository contains the **website source**, not a release of the model training code or dataset.

## Local preview

```sh
python3 -m http.server 8000 --directory docs
```

Visit http://localhost:8000/. No build step or package installation is needed.

## Deployment

GitHub Pages is deployed from `docs/` by `.github/workflows/static.yml`.
Set Pages → Source to GitHub Actions. Push to `main` to deploy.
