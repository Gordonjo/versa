Meta-Learning Probabilistic Inference For Prediction
====================================================

This repository implements the models and algorithms necessary to reproduce the experiments carried out in the paper
`Meta-Learning Probabilistic Inference For Prediction, Gordon and Bronskill et al. <https://arxiv.org/abs/1805.09921>`_
It includes code for running few-shot classification experiments with Omniglot and miniImageNet, as well as for reproducing
the ShapeNet view reconstruction experiments.

The code has been authored by: John Bronskill, Jonathan Gordon, and Matthias Bauer.

The main components of the repository are:

* ``run_classifier.py``: script to run classification experiments on Omniglot and miniImageNet
* ``train_view_reconstruction.py``: script to train view recovery models using ShapeNet objects
* ``evaluate_view_reconstruction.py``: script to test view recovery models using ShapeNet objects
* ``features.py``: deep neural networks for feature extraction and image generation
* ``inference.py``: amortized inference networks for various versions of Versa
* ``utilities.py``: assorted functions to support the repository

Dependencies
------------
This code requires the following:

* python 2 or python 3
* TensorFlow v1.0+

Data
----
For Omniglot, miniImagenet, and ShapeNet data, see the usage instructions in ``data/save_omniglot_data.py``, ``data/save_mini_imagenet_data.py``, and ``data/save_shapenet_data.py``, respectively.

Usage
-----

* To run few-shot classification, see the usage instructions at the top of ``run_classifier.py``.
* To run view reconstruction, see the usage instructions at the top of ``train_view_reconstruction.py`` and  ``evaluate_view_reconstruction.py``.

Contact
-------
To ask questions or report issues, please open an issue on the issues tracker.

Extending the Model
-------------------

There are a number of ways the repository can be extended:

* **Data**: to use alternative datasets, a class must be implemented to handle the new dataset. The necessary methods for the class are: ``__init__``, ``get_batch``, ``get_image_height``, ``get_image_width``, and ``get_image_channels``. For example signatures see ``omniglot.py``, ``mini_imagenet.py`` or ``omniglot.py``. Note that the code currently handles only image data. Finally, add the initialization of the class to the file ``data.py``.

* **Feature extractors**: to use alternative feature extractors, simply implement a desired feature extractor in ``features.py`` and change the function call in ``run_classifier.py``. For the required signature of a feature extractor see the function ``extract_features`` in ``features.py``.

Citation
--------

If you use this code for your research, please cite our `paper <https://arxiv.org/abs/1805.09921>`_:
::

  @inproceedings{gordon2018metalearning,
    title={Meta-Learning Probabilistic Inference for Prediction},
    author={Jonathan Gordon and John Bronskill and Matthias Bauer and Sebastian Nowozin and Richard Turner},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=HkxStoC5F7},
  }
