# AiCDR

-----------------------------------------------------------------------------------------------------

![alt text](./AiCDR.jpg)
AiCDR is generative adversarial network with three discriminators for nanobody CDR3 sequence generation.


AiCDR employs part of the benchmarking platform Texygen, see more:

https://github.com/geek-ai/Texygen


Code organization:
* `main.py` - Main script to initialize and run the AiCDR model.
* `example.sh` - Main script to initialize and run the AiCDR model.
* `utils/` -  Utility functions used by the main script.
* `data/` - Training data for AiCDR.
* `save/` - Files generated during intermediate steps of model training.
* `models/` - Saved AiCDR model checkpoints.
* `Guider1/` - Trained model for random sequence discrimination (Guider1).
* `Guider2/` - Trained model for peptide sequence discrimination (Guider2).
* `Guider/` - Scripts for training and evaluating Guider1 and Guider2.
* `Nanobody_library/` - A library of 5,194 nanobody structures with diverse CDR3 sequences.
* `Epitope_profiling/` - Contains docking results for the six target proteins..
-----------------------------------------------------------------------------------------------------
# Usage

Generate CDR3 sequences

* You can build the AiCDR environment through the AiCDR.yaml, the prefix path should be changed to your own path:

  ```
  conda env -f AiCDR.yaml
  ```
* Generating new CDR3 using following command:
  
  ```
  python main.py -g cdrgan -t real -d data/amp.txt
  ```
-----------------------------------------------------------------------------------------------------
Retrain the model

* First, train two Guider networks using your own positive and negative data:
  
  ```
  python Guider1.py
  python Guider2.py 
  ```

* Then, train AiCDR:

  ```
  python main.py -g cdrgan -t real -d <positive dataset location>
  ```
  * `<positive dataset location>` - positive data to train AiCDR
-----------------------------------------------------------------------------------------------------
```
@article{Unpublished,
  title={Computational nanobody design through deep generative modeling and epitope landscape profiling,
  author={Liyun Huo, Tian Tian, Yanqin Xu, Xinyi Jiang, Qiang Huang},
  journal={Unpublished},
  year={2025},
  publisher={Unpublished}
  }
```
-----------------------------------------------------------------------------------------------------
