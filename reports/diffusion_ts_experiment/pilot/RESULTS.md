# Experimento Diffusion-TS

**Conclusión automática del protocolo:** incorporar en sustitución de WGAN-GP: supera también VAE.

## Diseño

- Perfil: `pilot`; ajustes: [42, 43, 44].
- Entrenamiento: 3,000 actualizaciones por ajuste sobre 101,151 trayectorias efectivas.
- Muestreo: 102,406 pares por ajuste, DDIM 50 pasos.
- Representación: 60 retornos observados + 21 futuros; el target se recalcula con su fórmula física.
- Fuente adaptada: `machine-learning-for-trading` en `007a829a7494133662693676133e059785e1ba3a`.
- Comparadores: resultados TSTR versionados de los mismos datos, predictor y semillas.

## Reconstrucción

Se recuperaron 101,151 trayectorias (98.77% de las ventanas de train). El error máximo al reconstruir y fue 4.77e-07.

## TSTR

| fit_seed | val_r2 | val_mse | epoca_mejor | sampling_seconds |
| --- | --- | --- | --- | --- |
| 42 | 0.4633 | 0.1032 | 41 | 55.5564 |
| 43 | 0.4691 | 0.1021 | 73 | 55.2997 |
| 44 | 0.4543 | 0.1049 | 91 | 55.3714 |

R² medio Diffusion-TS: **0.4623**; ratio TSTR/TRTR: **0.917**.

### Comparaciones emparejadas por semilla

| comparador | delta_r2 | ci95_inf | ci95_sup | victorias |
| --- | --- | --- | --- | --- |
| real | -0.0417 | -0.0805 | -0.0028 | 0/3 |
| wgan_gp | 0.3373 | 0.3286 | 0.3459 | 3/3 |
| vae | 0.1280 | 0.1157 | 0.1403 | 3/3 |
| realnvp | -0.0148 | -0.0554 | 0.0258 | 1/3 |
| jitter | -0.0030 | -0.0151 | 0.0091 | 1/3 |

### Referencias históricas del notebook 03

| brazo | mean | std | count |
| --- | --- | --- | --- |
| block_bootstrap | 0.3143 | 0.0056 | 3 |
| gaussiana | -0.1953 | 0.0243 | 3 |
| jitter | 0.4653 | 0.0067 | 3 |
| real | 0.5039 | 0.0085 | 3 |
| realnvp | 0.4771 | 0.0104 | 3 |
| vae | 0.3343 | 0.0077 | 3 |
| wgan_gp | 0.1250 | 0.0044 | 3 |

## Fidelidad

| fit_seed | curtosis_x | acf_abs_lag1 | err_corr_xy | w1_x_col_media | err_corr_xx_spearman | discriminative_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 24.1732 | -0.0155 | 0.0384 | 0.2009 | 0.0286 | 0.8052 |
| 43 | 21.8650 | -0.0156 | 0.0371 | 0.1986 | 0.0283 | 0.7957 |
| 44 | 22.0918 | -0.0152 | 0.0361 | 0.1907 | 0.0285 | 0.7840 |

## Límite de inferencia

Este piloto compara sistemas: Diffusion-TS usa los 21 retornos futuros completos y deriva y, mientras los modelos originales reciben directamente el escalar y. Una mejora demuestra que la solución temporal merece avanzar, pero no separa el efecto de arquitectura del efecto de representación.

La decisión definitiva de producción requiere la malla de escasez usando la trayectoria de 81 días como unidad de muestreo; este informe no reutiliza retornos fuera del presupuesto declarado.
