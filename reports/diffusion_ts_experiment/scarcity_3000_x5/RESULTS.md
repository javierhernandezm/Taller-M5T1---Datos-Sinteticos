# Gate de escasez Diffusion-TS

**Conclusión:** supera el control real en las tres semillas: merece entrar en la malla completa.

Cada réplica usa 3,000 trayectorias reales y añade 15,000 sintéticas.
El generador se reajusta desde cero y no ve otras filas de train.

## Comparación emparejada

| seed | diffusion_ts_path81 | solo_real_path81 | delta_r2 |
| --- | --- | --- | --- |
| 42 | 0.3402 | 0.2944 | 0.0458 |
| 43 | 0.2676 | 0.1958 | 0.0718 |
| 44 | 0.4468 | 0.3939 | 0.0529 |

Delta R² medio: **+0.0568**, IC95 t de Student [+0.0234, +0.0902].

## Celda histórica del notebook 04

| generador | test_r2_media | test_r2_sd | n_seeds |
| --- | --- | --- | --- |
| jitter | 0.3875 | 0.0595 | 3 |
| realnvp | 0.4195 | 0.0354 | 3 |
| vae | 0.2723 | 0.0150 | 3 |
| wgan_gp | 0.1677 | 0.1253 | 3 |

La referencia histórica usa ventanas [X|y], mientras Diffusion-TS usa trayectorias de 81 retornos. Sirve para contextualizar magnitudes, no como contraste emparejado.
