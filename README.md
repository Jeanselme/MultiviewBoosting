# MultiviewBoosting
This code is an implementation of the algorithms proposed in [Multiview Boosting WIth Information Propagation for Classification](https://ieeexplore.ieee.org/document/7805338) by Peng et al. \[1\] and in [A Boosting Approach to Multiview Classification with Cooperation](https://link.springer.com/chapter/10.1007/978-3-642-23783-6_14) \[2\]

## Structure
`Evaluation.ipynb` runs an experiment to evaluate the different models proposed on MNIST as the experiment (5) proposed in the paper \[1\].

### Models
This folder contains multiple multiview boosting algorithms proposed in the papers:
- Boost.SH: Multiview boosting with shared weights.
- rBoost.SH: Randomized approach using adversarial bandit theory.
- eBoost.SH: Expert guided appraoch - NOT IMPLEMENTED
- Mumbo: Multiview boosting with individual weights. - NOT IMPLEMENTED

## Dependencies
Code has been run and tested with python3.4 and relies on scikit-learn and pandas libraries.