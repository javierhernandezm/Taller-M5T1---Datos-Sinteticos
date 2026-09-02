# TSTR Diffusion-TS R61

> **Diagnóstico auxiliar de tres reajustes, no es la tabla canónica del notebook 03.**
> Sirve para medir variabilidad del generador. La comparación oficial conserva el protocolo
> histórico de un ajuste y tres semillas downstream y está en `../nb03/tstr_resumen.csv`.

Comparación estricta: Diffusion-TS recibe exactamente la misma matriz estandarizada `[X60 | y]` que VAE, WGAN-GP y RealNVP. El target es un token especial, no un retorno.

| fit_seed | val_r2 | val_mse | epoca_mejor | sampling_seconds |
| --- | --- | --- | --- | --- |
| 42 | 0.4759 | 0.1008 | 80 | 40.3473 |
| 43 | 0.4591 | 0.1040 | 61 | 40.1077 |
| 44 | 0.4317 | 0.1093 | 74 | 40.1495 |

R² medio: **0.4555**; ratio TSTR/TRTR: **0.904**.

## Comparaciones emparejadas

| comparador | delta_r2 | ci95_inf | ci95_sup | victorias |
| --- | --- | --- | --- | --- |
| real | -0.0484 | -0.1152 | 0.0185 | 0/3 |
| wgan_gp | 0.3305 | 0.2792 | 0.3819 | 3/3 |
| vae | 0.1213 | 0.0849 | 0.1577 | 3/3 |
| realnvp | -0.0215 | -0.0816 | 0.0385 | 0/3 |
| jitter | -0.0097 | -0.0487 | 0.0292 | 1/3 |

## Fidelidad

| fit_seed | sd_x | curtosis_x | acf_abs_lag1 | err_corr_xy | w1_x_col_media | err_corr_xx_spearman | discriminative_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.6898 | 20.2026 | -0.0143 | 0.0363 | 0.1841 | 0.0284 | 0.7875 |
| 43 | 0.6913 | 20.2962 | -0.0154 | 0.0377 | 0.1855 | 0.0285 | 0.7861 |
| 44 | 0.7032 | 19.4106 | -0.0153 | 0.0368 | 0.1764 | 0.0287 | 0.7823 |
