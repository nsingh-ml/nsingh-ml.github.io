---
title: "Starting a Research Journal"
date: 2026-04-18
draft: false
tags: ["meta", "world-models", "JEPA"]
math: true
---

This blog documents my weekly reading notes on World Models and 
self-supervised representation learning.

**North Star:** Learning compact, abstract world representations 
through latent-space prediction — without pixel reconstruction — 
for sequential decision-making.

## Testing LaTeX

The KL term is $D_{KL}(q \| p)$

And the RSSM prior:

$$
z_t \sim p_\phi(z_t \mid h_t)
$$


The VAE objective (ELBO) is:

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z \vert x)}[\log p_\theta(x \vert z)] - D_{KL}(q_\phi(z \vert x) \Vert p(z))
$$


## Testing Code

```python
import torch
# MDN-RNN logvar collapse check
logvar = model.logvar.detach()
print(f"logvar mean: {logvar.mean():.4f}")  # Should not be near zero
```