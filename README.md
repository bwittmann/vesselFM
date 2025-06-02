<img src="docs/vesselfm_banner_updated.png">

**TL;DR**: VesselFM is a foundation model for universal 3D blood vessel segmentation. It is trained on three heterogeneous data sources: a large, curated annotated dataset, synthetic data generated through domain randomization, and data sampled from a flow matching-based deep generative model. These data sources provide enough diversity to enable vesselFM to achieve exceptional *zero*-shot blood vessel segmentation, even in completely unseen domains. For details, please refer to our preprint (https://arxiv.org/pdf/2411.17386).

---


## 🟢 Installation
First, set up a conda environment and install dependencies:

    conda create -n vesselfm python=3.9

    conda activate vesselfm

    pip install -e .


## 🟢 *Zero*-Shot Segmentation
If you are solely interested in running vesselFM's inference script for *zero*-shot segmentation of data at hand, adjust the respecitve [config file](vesselfm/seg/configs/inference.yaml) (see `#TODO`) and run:

    python vesselfm/seg/inference.py

Additional information on inference, pre-training, and fine-tuning are available [here](./vesselfm/seg). Checkpoints will be downloaded automatically and are also available on [Hugging Face 🤗](https://huggingface.co/bwittmann/vesselFM).


## 🟢 Data Sources
<img src="docs/data_sources.png">

We also provide individual instructions for generating our three proposed data sources.

$\mathcal{D}_\text{drand}$: Domain randomized synthetic data ([here](./vesselfm/d_drand)).

$\mathcal{D}_\text{flow}$: Synthetic data sampled from our flow matching-based deep generative model ([here](./vesselfm/d_flow)).

$\mathcal{D}_\text{real}$: Real data curated from 17 annotated blood vessel segmentation datasets ([here](./vesselfm/d_real)).


## 🟢 Citing vesselFM
If you find our work useful for your research, please cite:

```bibtex
@article{wittmann2024vesselfm,
  title={vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
  author={Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
  journal={arXiv preprint arXiv:2411.17386},
  year={2024}
}
```

## 🟢 License
Code in this repository is licensed under [GNU General Public License v3.0](LICENSE). Model weights are released under [Open RAIL++-M License](https://huggingface.co/bwittmann/vesselFM/blob/main/LICENSE) and are restricted to research and non-commercial use only.