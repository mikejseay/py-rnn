# py-rnn

A Python package for creating sparse, randomly-connected, recurrent neural networks, training them using an "innate trajectory" approach, and performing associated experiments.

## Getting Started

This package is written in Python 3. To get started, install Python 3 on your computer (we recommend the Conda package manager), then clone this repo onto your computer.

### Prerequisites

This package depends on:

* matplotlib
* numpy
* scipy
* tqdm

To install them, you can use pip or conda. For pip, type

```
pip install matplotlib
```

For conda, type

```
conda install matplotlib
```

Repeat this process for each package in the list above.

### Usage

The file test_main.py is a script that sets experimental parameters and performs the experiment. To run this, set the repo as your current directory and type

```
python test_main.py
```

The output figures will be saved as a PDF that will be placed into the figs subdirectory of your repo.

## Authors

* **Michael Seay** - [mikejseay](https://github.com/mikejseay)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This code would not be possible without

* Laje, R., & Buonomano, D. V. (2013). Robust timing and motor patterns by taming chaos in recurrent neural networks. Nature Neuroscience, 16(7), 925–933. https://doi.org/10.1038/nn.3405
* Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557. https://doi.org/10.1016/j.neuron.2009.07.018

## Citation

This code is the product of work carried out in the group of [Dean Buonomano at the University of California Los Angeles](http://www.buonomanolab.com/). If you find our code helpful to your work, consider citing us in your publications:

* Laje, R., & Buonomano, D. V. (2013). Robust timing and motor patterns by taming chaos in recurrent neural networks. Nature Neuroscience, 16(7), 925–933. https://doi.org/10.1038/nn.3405
