## Model Configuration Files

This folder contains configuration files for different versions of our models.

### Model Versions

- **v44**: Final model (**ASE-TM**), incorporates both **Mamba 2** and **attention** mechanisms.
- **v14**: Model with **Mamba 2**, but **without attention**.
- **v9**: Model based on **Mamba 1**.

### Notes

- In the `active_SEMamba_denoise/SEMamba_active_RIR_v44.yaml` file, the `training_cfg.predict_future` parameter specifies the future window size for prediction.
- Currently, this setting is **only supported** by the **denoise** model.