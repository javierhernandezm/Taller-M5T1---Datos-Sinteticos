# Experimento Diffusion-TS

**Conclusión automática del protocolo:** pendiente: el perfil no ejecutó downstream.

## Diseño

- Perfil: `smoke`; ajustes: [42].
- Entrenamiento: 20 actualizaciones por ajuste sobre 2,048 trayectorias como máximo.
- Muestreo: 512 pares por ajuste, DDIM 5 pasos.
- Representación: 60 retornos observados + 21 futuros; el target se recalcula con su fórmula física.
- Fuente adaptada: `machine-learning-for-trading` en `007a829a7494133662693676133e059785e1ba3a`.
- Comparadores: resultados TSTR versionados de los mismos datos, predictor y semillas.

## Reconstrucción

Se recuperaron 101,151 trayectorias (98.77% de las ventanas de train). El error máximo al reconstruir y fue 4.77e-07.

## TSTR

No ejecutado en este perfil.

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
| 42 | -0.0428 | 0.8578 | 0.6255 | 0.8533 | 0.5398 | 1.0000 |

## Límite de inferencia

Este piloto compara sistemas: Diffusion-TS usa los 21 retornos futuros completos y deriva y, mientras los modelos originales reciben directamente el escalar y. Una mejora demuestra que la solución temporal merece avanzar, pero no separa el efecto de arquitectura del efecto de representación.

La decisión definitiva de producción requiere la malla de escasez usando la trayectoria de 81 días como unidad de muestreo; este informe no reutiliza retornos fuera del presupuesto declarado.
