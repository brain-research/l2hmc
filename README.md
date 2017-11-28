# L2HMC: Automatic Training of MCMC Samplers

TensorFlow open source implementation for training MCMC samplers from the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com)

---

Given an analytically described distributions (implemented as in `utils/distributions.py`), L2HMC enables training of fast-mixing samplers. We provide an example, in the case of the Strongly-Correlated Gaussian, in the notebook `SCGExperiment.ipynb` --other details are included in the paper.


## Contact

***Code author:*** Daniel Levy

***Pull requests and issues:*** @daniellevy

## Citation

If you use this code, please cite our paper:
```
@article{levy2017generalizing,
  title={Generalizing Hamiltonian Monte Carlo with Neural Networks},
  author={Levy, Daniel and Hoffman, Matthew D. and Sohl-Dickstein, Jascha},
  journal={arXiv preprint arXiv:1711.09268},
  year={2017}
}
```

## Note

This is not an official Google product.
