# MultiviewBoosting
This code is an implementation of the algorithm proposed in [Multiview Boosting WIth Information Propagation for Classification](https://ieeexplore.ieee.org/document/7805338) by Peng et al.

## Structure
`Evaluation.ipynb` runs an experiment to evaluate the different models proposed on MNIST as the experiment (5) proposed in the paper.

### Models
This folder contains the three models proposed in the paper:
- Boost.SH: Multiview boosting which shared weights.
- rBoost.SH: Randomized approach using adversarial bandit theory.
- eBoost.SH: Expert guided appraoch

## Dependencies
Code has been run and tested with python3.4 and relies on scikit-learn and pandas libraries.