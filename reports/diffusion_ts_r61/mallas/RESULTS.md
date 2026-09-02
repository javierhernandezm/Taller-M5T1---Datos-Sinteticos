# Mallas completas — Diffusion-TS R61

Comparacion estricta sobre `[X60 | y]`: WGAN-GP se retira de la lista activa y Diffusion-TS ocupa exactamente sus 129 celdas en las dos mallas.

## Cobertura

- Ratios finos: 333 celdas; Diffusion-TS 54.
- Curvas: 465 celdas; Diffusion-TS 75.
- Ajustes independientes de Diffusion-TS: 15.

## Efecto frente a solo real

Ratios: 7 mejoras, 0 empeoramientos y 11 celdas no concluyentes.
Curvas: 9 mejoras, 1 empeoramientos y 15 celdas no concluyentes.

## Comparacion pareada con generadores

| design | comparator | pairs | delta_r2_mean | delta_r2_sd | wins | losses |
| --- | --- | --- | --- | --- | --- | --- |
| ratios | jitter | 54 | -0.0215 | 0.0611 | 19 | 35 |
| ratios | gaussiana | 54 | 0.0775 | 0.1140 | 42 | 12 |
| ratios | block_bootstrap | 54 | 0.0113 | 0.0631 | 40 | 14 |
| ratios | vae | 54 | 0.0351 | 0.0800 | 42 | 12 |
| ratios | realnvp | 54 | -0.0100 | 0.0804 | 29 | 25 |
| ratios | wgan_gp | 54 | 0.1682 | 0.1479 | 52 | 2 |
| curves | jitter | 75 | -0.0009 | 0.0520 | 39 | 36 |
| curves | gaussiana | 75 | 0.1096 | 0.1041 | 71 | 4 |
| curves | block_bootstrap | 75 | 0.0205 | 0.0635 | 60 | 15 |
| curves | vae | 75 | 0.0639 | 0.1127 | 63 | 12 |
| curves | realnvp | 75 | 0.0003 | 0.0682 | 36 | 39 |
| curves | wgan_gp | 75 | 0.1276 | 0.1532 | 69 | 6 |

La conclusion definitiva debe leerse junto con `comparisons.csv`, los deltas pareados y el diagnostico TSTR; no se decide solo por una media global.
