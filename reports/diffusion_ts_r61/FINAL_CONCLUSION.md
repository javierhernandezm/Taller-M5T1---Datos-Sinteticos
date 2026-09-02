# Diffusion-TS R61 — conclusión definitiva de integración

## Veredicto ejecutivo

**Sí: Diffusion-TS R61 merece sustituir a WGAN-GP en el pipeline activo.** La decisión se
apoya en tres pruebas concordantes:

1. TSTR oficial: mejora el R² histórico del WGAN en **+0,3523** y gana 3/3 semillas.
2. Malla de ratios: gana al WGAN en **52/54** comparaciones y lo supera en +0,1682 R²
   medio.
3. Malla de curvas: gana al WGAN en **69/75** comparaciones y lo supera en +0,1276 R²
   medio.

La recomendación tiene un límite importante: **Diffusion-TS no es un campeón absoluto**.
Empata globalmente con RealNVP y jitter, es más caro que el segundo y reproduce peor que
RealNVP la escala y el clustering de volatilidad. Debe entrar como sustituto del WGAN y
candidato fuerte, no como única opción por defecto.

## Por qué el experimento definitivo es R61 y no R81

La unidad de comparación del taller es:

```text
R61 = [60 retornos pasados estandarizados | y estandarizado]
```

Todos los generadores originales reciben esos 61 valores. El primer piloto de
Diffusion-TS usó `R81 = [60 retornos pasados | 21 retornos futuros]` y reconstruyó `y` a
partir del segundo tramo. Era una prueba útil de viabilidad, pero no una comparación justa:
Diffusion veía veinte grados de libertad adicionales y una representación más informativa
del futuro.

La implementación definitiva elimina esa ventaja:

- los 60 retornos son tokens temporales;
- `y` es un token especial, con proyección y cabeza de salida propias;
- la atención bidireccional aprende la distribución conjunta `p(X, y)`;
- la pérdida espectral se aplica solo a X, porque `y` no es un retorno temporal;
- la entrada y salida siguen siendo exactamente `(n, 61)`.

R81 fue una exploración previa y **no se versiona**: ni su informe ni sus runners entran
en `main`, porque ninguna cifra canónica de los notebooks 03/04 procede de ellos. Queda
recuperable desde el historial de la rama `codex/research-diffusion-ts`.

## Procedencia y adaptación

- Rama aislada: `codex/research-diffusion-ts`, creada desde
  `main@f8fff012706897bdf2fbe9b4fe2c013f0859bce6`.
- Código de referencia fijado al commit upstream `007a829a7494133662693676133e059785e1ba3a`:
  [`05_diffusion_ts.py`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/007a829a7494133662693676133e059785e1ba3a/05_synthetic_data/05_diffusion_ts.py).
- Trabajo original: [Diffusion-TS: Interpretable Diffusion for General Time Series Generation](https://openreview.net/forum?id=4h1apFjO99).

La adaptación local conserva las ideas relevantes del modelo de referencia: predicción de
`x0`, Transformer bidireccional, tendencia polinómica, componente Fourier, residuo temporal,
calendario coseno, EMA y muestreo DDIM. Se elimina cualquier dependencia del formato del
notebook upstream y se implementa la interfaz común `fit/sample/save/load` del taller.

No se aplicó reescalado posterior, winsorización ni calibración con validación real. Los
defectos de dispersión se dejan visibles.

## Protocolo experimental

### Notebook 03 — fidelidad y TSTR

- 102.406 ventanas reales de train.
- Diffusion-TS: 3.000 actualizaciones, 50 pasos DDIM.
- Un ajuste del generador con semilla 42, igual que el protocolo histórico del notebook.
- Tres semillas downstream: 42, 43 y 44.
- 21 filas activas: TRTR + 6 generadores × 3 semillas.
- Manifiesto con firma de receta, configuración y hash del checkpoint.

El ensayo auxiliar con tres reajustes completos se conserva en `tstr/`, pero no se mezcla
con el resultado oficial porque responde a un protocolo distinto.

### Notebook 04 — mallas completas

| Diseño | N real | Presupuesto sintético | Celdas activas | Diffusion |
|---|---|---|---:|---:|
| ratios finos | 250 · 1.000 · 3.000 | 0,25× · 0,5× · 0,75× · 1× · 3× · 5× | 333 | 54 |
| curvas | 250 · 1.000 · 3.000 · 10.000 · 20.000 | 500 · 1.000 · 3.000 · 7.000 · 20.000 | 465 | 75 |

Se realizaron 15 ajustes independientes de Diffusion-TS, uno por `(N_real, semilla)`. De
cada ajuste se generó una muestra máxima y se reutilizaron prefijos **idénticos bit a bit**
entre presupuestos solapados. Son 117 evaluaciones downstream únicas representadas en 129
celdas, porque 12 presupuestos coinciden entre ambas mallas.

Coste observado en una RTX 4070 Laptop:

- ajuste de 3.000 pasos: 48,0 s de media;
- muestreo DDIM-50 de 20.000 R61: 8,2 s de media;
- pipeline completo de las celdas nuevas: 50,2 min, dominado por el downstream.

## Resultados

### TSTR oficial

| Brazo | R² medio | sd | TSTR/TRTR |
|---|---:|---:|---:|
| real | 0,5039 | 0,0085 | 1,0000 |
| **Diffusion-TS R61** | **0,4773** | **0,0033** | **0,9472** |
| RealNVP | 0,4771 | 0,0104 | 0,9467 |
| jitter | 0,4653 | 0,0067 | 0,9233 |
| VAE | 0,3343 | 0,0077 | 0,6633 |
| block bootstrap | 0,3143 | 0,0056 | 0,6237 |
| gaussiana | −0,1953 | 0,0243 | −0,3876 |

Diffusion queda primero, pero la distancia a RealNVP es solo **+0,00025 R²**: un empate
práctico. La diferencia con WGAN-GP sí es material: su referencia histórica era 0,1250.

### Fidelidad

| Métrica | Real | Diffusion | RealNVP |
|---|---:|---:|---:|
| sd(X) | 1,0028 | **0,6919** | 0,9226 |
| sd(y) | 0,9883 | **0,8138** | 0,9809 |
| curtosis de X | 25,24 | 19,85 | 23,24 |
| ACF(|r|), lag 1 | 0,0618 | **−0,0135** | 0,0424 |
| AUC discriminativa | 0,500 ideal | **0,7866** | 0,6547 |
| W1 por columna | 0 ideal | **0,1868** | 0,0567 |

Diffusion conserva bien las colas y la relación predictiva, pero produce trayectorias
infradispersas y sin clustering. El resultado TSTR no debe leerse como fidelidad total.

### Malla de ratios

Diffusion-TS: **7 mejoras, 0 empeoramientos, 11 no concluyentes**.

| N real | Lectura |
|---:|---|
| 250 | solo 5× concluye: +0,0765 R²; las otras cinco dosis son inestables |
| 1.000 | las seis medias son positivas (+0,095 a +0,158), ninguna supera el ruido de tres semillas |
| 3.000 | mejoran las seis: +0,0136 con 0,25× hasta +0,0629 con 5× |

Media de ΔR² sobre las 18 configuraciones:

| Generador | ΔR² medio |
|---|---:|
| jitter | +0,0590 |
| RealNVP | +0,0475 |
| **Diffusion-TS** | **+0,0375** |
| block bootstrap | +0,0262 |
| VAE | +0,0025 |
| gaussiana | −0,0400 |

### Malla de curvas

Sobre R² del test, Diffusion-TS obtiene **9 mejoras, 1 empeoramiento y 15 resultados no
concluyentes**. La única pérdida es aislada: 20.000 reales + 500 sintéticos, ΔR² −0,0037.

| Comparador | ΔR² medio Diffusion − comparador | Victorias |
|---|---:|---:|
| WGAN histórico | **+0,1276** | **69/75** |
| gaussiana | +0,1096 | 71/75 |
| VAE | +0,0639 | 63/75 |
| block bootstrap | +0,0205 | 60/75 |
| RealNVP | **+0,0003** | 36/75 |
| jitter | **−0,0009** | 39/75 |

La equivalencia con RealNVP/jitter es muy estrecha. El efecto de los tres se desvanece al
llegar a 10.000–20.000 reales: a partir de ahí la elección del generador tiene poco valor
práctico.

## Interpretación

1. **La sustitución no es marginal.** Diffusion no mejora al WGAN por una semilla o una
   métrica: lo hace en TSTR y en dos diseños de malla.
2. **La representación justa importa.** R81 demostró viabilidad; R61 demuestra que la
   arquitectura sigue funcionando sin información adicional.
3. **Utilidad y fidelidad no son equivalentes.** Diffusion empata el TSTR de RealNVP pese a
   una fidelidad claramente peor. La auditoría detecta descartes, no siempre ordena los
   candidatos fuertes.
4. **El régimen manda.** El caso más reproducible está alrededor de 3.000 reales. Con 250 y
   1.000 el potencial es alto, pero la varianza entre semillas también; con 10.000–20.000
   la ganancia es demasiado pequeña para justificar el coste.
5. **No hay evidencia de un óptimo interior estable** dentro de 0,25×–5× o hasta 20.000
   sintéticos. No se extrapola fuera de ese rango.

## Recomendación de pipeline

- Mantener activos `jitter`, `gaussiana`, `block_bootstrap`, `vae`, `diffusion_ts` y
  `realnvp`; conservar WGAN solo como implementación histórica reproducible.
- Usar Diffusion-TS como candidato principal cuando haya aproximadamente 3.000 ventanas
  reales y se contemple una ampliación de 1×–6,7×.
- Mantener RealNVP como referencia neuronal de fidelidad y jitter como baseline obligatorio
  de coste mínimo.
- No pagar el coste de difusión por defecto con 10.000 o más ventanas reales salvo que una
  nueva validación demuestre una mejora material.

## Limitaciones

1. Tres semillas son pocas en los regímenes N=250 y N=1.000.
2. Solo se ha probado una configuración de Diffusion-TS, un split temporal y un target.
3. El umbral `|t| >= 2,5` es una regla operativa con n=3, no una prueba confirmatoria.
4. No se ha hecho una ablación de la tendencia, Fourier, EMA o número de pasos DDIM.
5. La privacidad está fuera de alcance porque los precios son públicos; en datos sensibles
   haría falta una auditoría separada.

## Reproducción y artefactos

```powershell
python scripts/run_diffusion_ts_r61_nb03.py
python scripts/run_diffusion_ts_r61_mallas.py
python scripts/promote_diffusion_ts_r61_results.py
python scripts/update_diffusion_ts_r61_notebooks.py
python scripts/run_diffusion_ts_r61_notebook03.py
```

El quinto paso no es opcional. Los cuatro primeros calculan Diffusion-TS y
promueven las mallas, pero ninguno produce la tabla de fidelidad publicada:
`run_diffusion_ts_r61_nb03.py` solo recalcula su propia fila de auditoría y
hereda las otras cinco de la ejecución anterior. El notebook 03 es el único
sitio donde los seis generadores se ajustan en una misma corrida, así que es el
que hace comparables entre sí las cifras de la sección «Fidelidad». Sin él, un
clon limpio termina con una auditoría mezclada de dos ejecuciones y no
reproduce lo publicado.

El notebook reajusta los seis generadores, pero reutiliza el TSTR cacheado
mientras el manifiesto del paso 1 siga siendo válido: no repite los 21
entrenamientos downstream. El paso deja además `reports/diffusion_ts_r61/nb03/`
en espejo con `data/processed/`, que es la invariante que protege
`tests/test_artefactos_r61.py`.

- `data/processed/tstr_nb03_manifest.json`: receta y checkpoint del TSTR oficial.
- `data/processed/malla_*_delta_pareado.csv`: inferencia por semilla de ambas mallas.
- `reports/diffusion_ts_r61/nb03/`: auditoría y TSTR R61.
- `reports/diffusion_ts_r61/mallas/`: configuración, costes, comparaciones y archivos del
  experimento; checkpoints y matrices sintéticas quedan ignorados por Git.
- `notebooks/03_generadores.ipynb` y `04_malla_sintetica.ipynb`: narrativa y outputs
  canónicos ya regenerados.

