# JAX-PF
## Differentiable pattern forming simulations with finite difference and pseudospectral methods implemented in jax.

Inspiration from:
- [JAX-CFD](https://github.com/google/jax-cfd)
- [Diffrax](https://github.com/patrick-kidger/diffrax)
- [condensate](https://github.com/biswaroopmukherjee/condensate)

In particular, much of the code was modeled after jax-cfd, including Domain class, the equation classes, and the timestepping and anti-aliasing utilities.

Many of these models are commonly used in materials science. For example, the Gross-Pitaevskii equation is used to model Bose-Einstein condensates and general ultracold quantum gases. In addition, the Cahn-Hilliard equation occurs in phase separation in liquids and solids, including biomolecular condensates in cells and lithium ion battery electrode materials.
