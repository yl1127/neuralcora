# NeuralCora

NeuralCora is a research codebase for applying neural network models to (re)forecasting in the CORA. This repository hosts baseline models and training/evaluation pipelines.

## 📋 Table of Contents

- [Motivation](#motivation)  
- [Models & Baselines](#models--baselines)  
- [Usage Examples](#usage-examples)  

## Motivation

This codebase is intended to be modular, extensible, and reproducible.  

## Models & Baselines

Below is a summary of current and prospective model performance (or planned models) for a 1-hour forecast task (Zeta, units in meters).  

| Model                         | Zeta RMSE (1 hour) [m] | Reference                  |
|------------------------------|--------------------------|-----------------------------|
| Persistence (Baseline)       | 0.205                    | Rasp et al. (2020)          |
| Climatology (Baseline)       | 0.425                    | Rasp et al. (2020)          |
| CNN                          | 0.048                    | Rasp et al. (2020)          |
| U-Net                        | (Ongoing)                | Weyn et al. (2020)          |
| ResNet                        | —                        | (Ongoing)                    |
| Vision Transformer (ViT)     | —                        | —                           |

**Notes:**
- “(Ongoing)” indicates work in progress; results are to be added as training and evaluation finish.
- Dashes (—) mean the model is not yet implemented or evaluated.

## Usage Examples

You can find example demos in the notebooks:

- **quickstart-baseline.ipynb** — shows how to train and evaluate the persistence / climatology baselines  
- **quickstart-cnn.ipynb** — shows how to train and evaluate the CNN model
