# Investigación Diffusion-TS

> **Experimento exploratorio R81, superado por la comparación estricta R61.** Este informe
> se conserva por trazabilidad, pero no sostiene la decisión activa porque Diffusion-TS veía
> 20 retornos adicionales. El pipeline completo y el veredicto definitivo están en
> [`../diffusion_ts_r61/FINAL_CONCLUSION.md`](../diffusion_ts_r61/FINAL_CONCLUSION.md).

## Veredicto

**Sí merece incorporarse al pipeline experimental en sustitución de WGAN-GP.**

La recomendación se apoya en dos contrastes complementarios:

1. En TSTR puro, Diffusion-TS conserva el 91,7% del R² del entrenamiento con datos reales,
   supera a WGAN-GP y VAE en las tres semillas y queda estadísticamente empatado con RealNVP.
2. En el gate de escasez con 3.000 trayectorias reales y 15.000 sintéticas, mejora el R² de
   test en las tres semillas. El incremento medio es +0,0568, con IC95 [+0,0234, +0,0902].

La conclusión no autoriza todavía un cambio directo en `main`: el siguiente gate debe ser la
malla completa, redefiniendo la trayectoria de 81 días como unidad muestral para impedir fugas
de presupuesto real.

## Procedencia y adaptación

- Rama: `codex/research-diffusion-ts`, creada desde `main` en `f8fff012706897bdf2fbe9b4fe2c013f0859bce6`.
- Referencia externa: [`05_diffusion_ts.py`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/007a829a7494133662693676133e059785e1ba3a/05_synthetic_data/05_diffusion_ts.py), fijada al commit `007a829a7494133662693676133e059785e1ba3a`.
- La implementación local es una adaptación univariante: predicción de `x0`, Transformer
  bidireccional, tendencia polinómica, componente Fourier, pérdida temporal y espectral,
  calendario coseno, EMA y muestreo DDIM.
- No se aplicó corrección posterior de varianza, winsorización ni calibración con validación.

## Representación y control de fuga

Cada observación se reconstruye como:

```text
R81 = [60 retornos observados | 21 retornos futuros]
X   = R81[:60]
y   = log(sqrt(252 * mean(R81[60:] ** 2)))
```

Se recuperaron 101.151 de 102.406 trayectorias de train (98,77%). Se verifican activo,
tramo, fechas y solapamiento numérico; el error máximo al reconstruir el target fue
`4,77e-07`. La etiqueta sintética queda así determinada por los retornos generados, sin un
modelo auxiliar.

## Resultado TSTR

Se entrenaron tres generadores independientes durante 3.000 actualizaciones. Cada uno produjo
102.406 pares sintéticos mediante DDIM de 50 pasos, y con ellos se entrenó el GRU congelado del
notebook 02.

| Semilla | R² Diffusion-TS | R² WGAN-GP | R² VAE | R² RealNVP |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0,4633 | 0,1240 | 0,3406 | 0,4848 |
| 43 | 0,4691 | 0,1298 | 0,3366 | 0,4653 |
| 44 | 0,4543 | 0,1211 | 0,3256 | 0,4810 |
| **Media** | **0,4623** | **0,1250** | **0,3343** | **0,4771** |

Comparaciones emparejadas sobre R²:

| Comparador | Delta medio Diffusion-TS − comparador | IC95 | Victorias |
| --- | ---: | ---: | ---: |
| WGAN-GP | +0,3373 | [+0,3286, +0,3459] | 3/3 |
| VAE | +0,1280 | [+0,1157, +0,1403] | 3/3 |
| RealNVP | −0,0148 | [−0,0554, +0,0258] | 1/3 |
| Datos reales | −0,0417 | [−0,0805, −0,0028] | 0/3 |

Por tanto, la difusión es inequívocamente mejor que el modelo que se pretende retirar, mejora
también al VAE y no puede distinguirse de RealNVP con solo tres pares.

## Gate de escasez

Para cada semilla se seleccionaron por fechas 3.000 anclas `R81`, se reajustó Diffusion-TS
exclusivamente con ellas y se añadieron 15.000 pares sintéticos. El control usa exactamente las
mismas 3.000 anclas sin aumento.

| Semilla | Solo real | Real + Diffusion-TS | Delta R² |
| ---: | ---: | ---: | ---: |
| 42 | 0,2944 | 0,3402 | +0,0458 |
| 43 | 0,1958 | 0,2676 | +0,0718 |
| 44 | 0,3939 | 0,4468 | +0,0529 |

El delta medio es **+0,0568**, con IC95 de Student **[+0,0234, +0,0902]**. Esta es la evidencia
directa de que el modelo no solo produce un TSTR alto: también añade utilidad en el régimen de
escasez para el que se diseñó el pipeline.

## Fidelidad y riesgos restantes

La utilidad predictiva es buena, pero la distribución todavía tiene defectos:

| Métrica media | Diffusion-TS | Real |
| --- | ---: | ---: |
| Desviación de retornos estandarizados | 0,675 | 1,000 |
| Kurtosis en exceso | 22,71 | 28,37 |
| ACF de `|r|`, lag 1 | −0,0154 | 0,0627 |
| Desviación del target | 0,841 | 1,000 |
| AUC discriminativa | 0,795 | 0,500 ideal |

- Sigue existiendo infradispersión, aunque no se ocultó con un factor de reescalado.
- Recupera bastante mejor las colas que una difusión gaussiana suavizada, pero no reproduce el
  clustering de volatilidad de lag corto.
- El buen TSTR puede convivir con una fidelidad imperfecta: el generador conserva la relación
  útil `X-y`, pero sigue siendo distinguible de los datos reales.
- Diffusion-TS ve los 21 retornos futuros que determinan `y`; los modelos originales solo ven el
  escalar. Esta investigación compara soluciones completas, no identifica de forma aislada el
  efecto de la arquitectura.

## Coste observado

En una RTX 4070 Laptop, por semilla y con el dataset completo:

- Ajuste de 3.000 pasos: 42,6 s de media.
- Muestreo de 102.406 observaciones, DDIM-50: 55,4 s de media.
- Entrenamiento TSTR del GRU: aproximadamente 155 s.

El coste del generador no es un impedimento para el notebook 03. Sí debe presupuestarse en la
malla, donde se reajusta por celda.

## Siguiente decisión de integración

La malla completa debería:

1. sustituir `wgan_gp` por un identificador nuevo, por ejemplo `diffusion_ts_path81`;
2. contar anclas `R81`, no filas `[X|y]`, como presupuesto de datos reales;
3. comparar al menos los niveles de 1.000 y 3.000 reales y ratios 1, 3 y 5;
4. conservar tres reajustes completos del generador por celda;
5. mantener sin corrección la variante actual y ensayar cualquier calibración de dispersión como
   una ablation separada ajustada exclusivamente con train;
6. añadir una comparación de representación común si se desea atribuir la mejora exclusivamente
   a la arquitectura.

## Reproducción

```powershell
python scripts/run_diffusion_ts_experiment.py --profile smoke
python scripts/run_diffusion_ts_experiment.py --profile pilot
python scripts/run_diffusion_ts_scarcity.py
```

Los checkpoints y matrices sintéticas se guardan localmente, pero están excluidos de Git. Los
resultados numéricos y las curvas de entrenamiento están versionados en las subcarpetas
[`pilot`](pilot/RESULTS.md) y [`scarcity_3000_x5`](scarcity_3000_x5/RESULTS.md).
