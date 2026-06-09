# Convolutional VAE on CelebA

![Faces sampled from the prior — none of these people exist](assets/prior_samples.png)

A from-scratch PyTorch implementation of a vanilla Variational Autoencoder for 64×64 RGB faces, built as a foundational component on the road to world models (VQ-VAE → DreamerV2 → IRIS).

The architecture maps directly onto the math:

- **Encoder** (conv stack) → produces `μ`, `log σ²`, the parameters of `q(z|x)`
- **Reparameterization** → `z = μ + σ ⊙ ε` with `ε ~ N(0, I)` so gradients flow through `μ`, `σ`
- **Decoder** (mirrored deconv) → outputs per-pixel logits for `p(x|z)`
- **Loss** → reconstruction (BCE-with-logits, Bernoulli likelihood) + closed-form Gaussian KL to the standard-normal prior

## Architecture

```
Encoder: (3, 64, 64) → (32, 32, 32) → (64, 16, 16) → (128, 8, 8) → (256, 4, 4) → flatten (4096)
                                                                              ↘ fc_μ      → μ  (32)
                                                                              ↘ fc_logvar → logσ² (32)
Reparam: z = μ + exp(½ logσ²) · ε,  ε ~ N(0, I)

Decoder: z (32) → fc_dec → (256, 4, 4) → (128, 8, 8) → (64, 16, 16) → (32, 32, 32) → (3, 64, 64) logits
```

Every conv / deconv is `kernel=4, stride=2, padding=1` for clean ÷2 / ×2 spatial steps. BatchNorm + ReLU after each layer except the final decoder layer, which is bare (it emits logits — the loss applies sigmoid internally, the viz applies it manually).

`latent_dim = 32` — a tight bottleneck. Identity-level information fits; pore-level detail doesn't. This is the same size used in the original World Models paper.

## Loss

```
ℒ = recon + KL,   averaged over the batch

recon = BCE-with-logits(logits, x)         summed over all pixels
KL    = ½ Σ (μ² + σ² − log σ² − 1)         summed over latent dims, the closed-form
                                            Gaussian KL to N(0, I)
```

Both terms are **per-example sums** then batch-averaged. Mixing reductions (e.g. mean-recon + sum-KL) lets the KL swamp the reconstruction and causes posterior collapse — the decoder ignores `z` and outputs an average face.

## Repo layout

```
.
├── vae.py            # ConvVAE class + vae_loss method + smoke test
├── train_vae.py      # data pipeline, training loop, checkpoint save
├── visualize.py      # reconstructions, prior samples, latent interpolation
├── checkpoint/       # vae.pth (created by training)
├── viz/              # output PNGs (created by visualize.py)
├── assets/           # images embedded in this README
└── celeba_root/
    └── img_align_celeba/   # ~200k CelebA jpgs
```

## Setup

```bash
pip install torch torchvision
```

**Dataset.** `torchvision.datasets.CelebA` pulls from Google Drive, which usually fails with a "quota exceeded" error. Fallback: download `img_align_celeba` from the Kaggle dataset `jessicali9530/celeba-dataset`, unzip, and arrange the folder so `ImageFolder` finds it:

```
celeba_root/
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

`ImageFolder("./celeba_root", ...)` then treats the inner folder as a single "class" and returns `(image, 0)` — the training loop discards the label.

## Train

```bash
python train_vae.py
```

10 epochs at `batch_size=128`, `lr=3e-4` (Adam), on Apple Silicon (`mps`) takes roughly an hour. The loss should start around ~8500 on the first few batches, drop fast through epoch 0, and then ease down — for example:

```
epoch 0: 6656.6
epoch 1: 6543.6
...
epoch 9: 6508.3
```

The big drop is epoch 0; the slow grind after that translates to *sharpness* in the reconstructions more than to a dramatic change in the loss number.

The trained model is saved to `./checkpoint/vae.pth`.

## Visualize

```bash
python visualize.py
```

Produces three PNGs in `./viz/`, each one testing a different theoretical claim:

**`reconstructions.png`** — top row originals, bottom row their reconstructions through `encode → reparameterize → decode`. Identity should be preserved (same person, same pose); fine detail will be softer. The softness isn't a bug — it's the inevitable consequence of the 32-D bottleneck plus the Bernoulli likelihood (independent-pixel assumption with no sharpness term) plus the KL pulling latents toward N(0, I). This is the well-known VAE-vs-GAN tradeoff: blurrier outputs in exchange for a principled, samplable latent space.

![Reconstructions — top row originals, bottom row reconstructions](assets/reconstructions.png)

**`prior_samples.png`** — 64 fresh `z ~ N(0, I)` decoded into faces. None of those people exist. The fact that random latent noise decodes to *recognizably face-shaped images* is the entire VAE thesis: the latent space is continuous and samplable. A plain autoencoder fails this test — random `z` lands in the holes between training codes and produces garbage. Yours doesn't, because the KL-to-prior term in the loss spent training pushing every `q(z|x)` toward `N(0, I)`. The smudgy samples are where the prior sampled a low-density region of the aggregate posterior — a real (and well-known) limitation of vanilla VAEs and one of the motivations for VQ-VAE's discrete prior.

![64 faces decoded from z ~ N(0, I) — none of these people exist](assets/prior_samples.png)

**`interpolation.png`** — encode two real faces to `μ₀`, `μ₁`, decode `z = (1−α)μ₀ + αμ₁` for `α ∈ [0, 1]` in 10 steps. A smooth morph (hair, pose, expression drifting gradually) means the straight line between two encoded points stays inside the "this means face" region the whole way — i.e. the latent space is *continuous*, not holey.

![Latent interpolation between two encoded faces — smoothness proves the space is continuous](assets/interpolation.png)

## Implementation notes

A few details that matter and aren't always obvious:

- **The decoder outputs logits, not pixels.** Sigmoid happens inside `binary_cross_entropy_with_logits` (numerically stable) and is applied manually for visualization. Putting a `Sigmoid` in the decoder *and* using the with-logits loss double-sigmoids and breaks training.
- **The network emits `log σ²`, not `σ`.** Avoids a positivity constraint and is numerically stable. `σ = exp(0.5 · log σ²)` is always positive.
- **Reduction balance.** Both reconstruction and KL are per-example sums then batch-averaged. This keeps them on the same scale at β = 1.
- **`eval()` for visualization.** BatchNorm uses batch statistics during training and running statistics during eval. Forget `model.eval()` before viz and even a well-trained model produces broken images.
- **Reparameterization lives inside `forward`** with fresh `ε` per call. Gradients flow through `μ` and `σ`; `ε` is detached noise.

## What's next

This VAE is the perception front-end of a much larger build:

- **β-VAE** — weight the KL by β > 1; individual latent dimensions start to correspond to disentangled factors (pose, lighting, smile).
- **VQ-VAE** — replace the continuous Gaussian latent with a discrete codebook. Same encode→quantize→decode skeleton, but quantization is non-differentiable, so gradients go through the **straight-through estimator** — the same trick that powers categorical latents in DreamerV2.
- **DreamerV2 → IRIS** — IRIS is essentially DreamerV2 with the recurrent dynamics replaced by an autoregressive Transformer over VQ-VAE tokens. VQ-VAE is the missing piece between Dreamer-style RL and IRIS-style RL.

## References

- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014 — the original VAE paper.
- Rezende, Mohamed & Wierstra, *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*, ICML 2014 — concurrent work introducing the reparameterization trick.
- Ha & Schmidhuber, *World Models*, 2018 — where this exact architecture (z = 32, 64×64) first appeared as the perception module of a world-model RL agent.
- Liu, Luo, Wang & Tang, *Deep Learning Face Attributes in the Wild*, ICCV 2015 — the CelebA dataset.
