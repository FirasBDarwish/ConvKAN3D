![Python build & test](https://github.com/FirasBDarwish/ConvKAN3D/actions/workflows/build.yaml/badge.svg)

<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">ConvKAN</h3>

  <p align="center">
    Kolmogorov-Arnold Networks (KANs) meet 3D Convolutional Layers (Conv3D)
    <br />
    <a href="https://arxiv.org/abs/2404.19756"><strong>Read the KANs Paper »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/FirasBDarwish/ConvKAN3D/issues">Report Bug</a>
    ·
    <a href="https://github.com/FirasBDarwish/ConvKAN3D/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Python package for 3D convolutional layers using Kolmogorov-Arnold Networks (KANs). I am new to packaging and distributing software in this manner--especially ML packages--so please go easy on me.

<!-- GETTING STARTED -->
## Getting Started

Using this packaage should be fairly simple!

### Installation

```sh
   $ pip install ConvKAN3D
```

Once you've installed the package and once in Python, make sure to import:

```python
   from ConvKAN3D.ConvKAN3D import effConvKAN3D
```

Now, you're all good to use effConvKAN3D just as you would with Conv3D (with some added hyperparameters for the underlying KAN module being used).

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/catiaspsilva/README-template/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
## License

Distributed under the GNU General Public License. See `LICENSE` for more information.

<!-- Authors -->
## Authors

Your Name - [firasbdarwish](https://www.linkedin.com/in/firasbdarwish/) - fbd2014@nyu.edu

Project Link: [https://github.com/FirasBDarwish/ConvKAN3D](https://github.com/FirasBDarwish/ConvKAN3D)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Official KAN Repo](https://github.com/KindXiaoming/pykan/tree/master)
* [AwesomeKAN Repo](https://github.com/mintisan/awesome-kan)
* [EffcientKAN Implementation](https://github.com/Blealtan/efficient-kan)

## Thank you