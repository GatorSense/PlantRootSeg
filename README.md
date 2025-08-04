# Overcoming Small Minirhizotron Datasets Using Transfer Learning

_Weihuang Xu, Guohao Yu, Alina Zare, Brendan Zurweller, Diane Rowland,Joel Reyes-Cabrera, Felix B Fritschi, Roser Matamala, Thomas E. Juenger_

If you refer the results, cite it: Weihuang Xu, Guohao Yu, Alina Zare, Brendan Zurweller, Diane Rowland,Joel Reyes-Cabrera, Felix B Fritschi, Roser Matamala & Thomas E. Juenger. (2019, Sept.). GatorSense/PlantRootSeg: Initial Release (Version v1.0). [![DOI](https://zenodo.org/badge/202222435.svg)](https://zenodo.org/badge/latestdoi/202222435)

[[`arXiv`](https://arxiv.org/pdf/1903.09344.pdf)] [[`BibTeX`](#CitingRootSeg)]

In this repository, we provide the Jupyter Notebook demo to show the results presented in paper "Overcoming Small Minirhizotron Datasets Using Transfer Learning"

## Demo
For running a demo, after forking the repository, open the RootShow.ipynb in Jupyter Nootbook.
Note: Since pre-trained models are large files. Please install [[`Git LFS`](https://git-lfs.github.com)] and clone repository with the command:

```bash
git lfs clone git@github.com:GatorSense/PlantRootSeg.git
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2019 W. Xu and A. Zare. All rights reserved.

## <a name="CitingRootSeg"></a>Citing DLMinirhizotronSegmentation

Please cite the following reference using the following BibTeX entries.
```
@article{xu2020overcoming,
  title={Overcoming small minirhizotron datasets using transfer learning},
  author={Xu, Weihuang and Yu, Guohao and Zare, Alina and Zurweller, Brendan and Rowland, Diane L and Reyes-Cabrera, Joel and Fritschi, Felix B and Matamala, Roser and Juenger, Thomas E},
  journal={Computers and Electronics in Agriculture},
  volume={175},
  pages={105466},
  year={2020},
  publisher={Elsevier}
}
```
## Train on your data

To finetune this model on your data, first select and initial model (e.g., from the Models folder).  Update data paths to your test imagery and annotation masks.  Our code is assuming XXX image types.  Apply any necessary preprocessing to your imagery (e.g., resize to be consistent in size, normalize imagery, etc).  Adjust parameters as necessary (e.g., epochs, segmentation threshold, etc). 

