"""Actualiza las celdas narrativas afectadas por la sustitución WGAN -> Diffusion-TS.

El experimento numérico se ejecuta en los runners R61. Este script solo mantiene
las conclusiones de los notebooks 03/04 sincronizadas con sus CSV canónicos y
añade al notebook 04 el contraste paralelo en validación y test.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _cell(nb: dict, cell_id: str) -> dict:
    return next(cell for cell in nb["cells"] if cell.get("id") == cell_id)


def _save(path: Path, nb: dict) -> None:
    path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def update_nb03() -> None:
    path = ROOT / "notebooks" / "03_generadores.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    _cell(nb, "26")["source"] = _source(
        r"""
## 5 · Análisis crítico — experimento R61 definitivo

La comparación activa es estrictamente homogénea: todos los generadores reciben y producen
`[X60 | y]`. En Diffusion-TS los 60 retornos son tokens temporales y `y` es un token
especial con proyección y cabeza propias. El prototipo R81 que reconstruía 20 observaciones
adicionales queda como exploración y **no interviene en ningún resultado de abajo**.

### 5.1 Fidelidad

Referencia real: desviación de X 1,003 · curtosis 25,24 · ACF(|r|) lag-1 0,062.

| Generador | sd(X) | Curtosis | ACF\|r\| lag1 | AUC | W1 por columna | Lectura |
|---|---:|---:|---:|---:|---:|---|
| jitter | 1,000 | 23,57 | **0,058** | **0,507** | **0,026** | casi indistinguible porque recicla train |
| gaussiana | 1,001 | 0,03 | −0,015 | 0,958 | 0,236 | falla colas y clustering |
| block_bootstrap | 0,990 | 26,16 | 0,125 | 0,760 | 0,057 | colas buenas, dependencia exagerada |
| VAE | 0,909 | 1,50 | −0,005 | 0,900 | 0,157 | sobre-suavizado |
| **Diffusion-TS R61** | **0,693** | 20,32 | −0,013 | 0,784 | 0,187 | buenas colas, clara contracción y sin clustering |
| RealNVP | 0,940 | 22,48 | **0,042** | **0,662** | **0,054** | mejor fidelidad neuronal global |

Diffusion-TS recupera el 81 % de la curtosis real, pero reduce la dispersión de X un 31 %,
la de `y` un 18 % y no reproduce el clustering de volatilidad. Por eso su AUC 0,784 es
peor que el 0,662 de RealNVP. El diagnóstico impide confundir una utilidad predictiva alta
con una réplica completa de la distribución financiera.

### 5.2 Utilidad TSTR con el protocolo oficial del notebook

Un único ajuste de cada generador y tres semillas del modelo downstream; selección de
época simétrica y validación real común.

| Brazo | R² validación | sd | ratio TSTR/TRTR |
|---|---:|---:|---:|
| real (TRTR) | 0,5039 | 0,0085 | 1,000 |
| **Diffusion-TS R61** | **0,4773** | **0,0033** | **0,9472** |
| RealNVP | 0,4771 | 0,0104 | 0,9467 |
| jitter | 0,4653 | 0,0067 | 0,9233 |
| VAE | 0,3343 | 0,0077 | 0,6633 |
| block bootstrap | 0,3143 | 0,0056 | 0,6237 |
| gaussiana | −0,1953 | 0,0243 | −0,3876 |

Diffusion-TS encabeza el TSTR activo por **0,00025 R²** frente a RealNVP: es un empate
práctico, no evidencia de superioridad. Frente al WGAN-GP retirado, cuya media histórica
bajo este mismo protocolo era 0,1250, mejora **+0,3523 R²** y gana en las tres semillas.

### 5.3 Qué se decide aquí y qué exige la malla

El TSTR da una respuesta clara a la sustitución: **Diffusion-TS conserva mucha más señal
que WGAN-GP y merece pasar a la malla**. No basta para declararlo generador por defecto:
RealNVP reproduce mejor escala y dependencia, mientras que TSTR ordena Diffusion primero.
Esa tensión es precisamente la razón de ejecutar las 798 celdas del notebook 04.

La correlación AUC–TSTR baja a Spearman −0,49 al introducir Diffusion-TS: una sola métrica
de fidelidad ya no ordena toda la utilidad. La auditoría sigue siendo un filtro eficaz
para descartar generadores tóxicos, pero la decisión final debe combinar fidelidad, TSTR y
ganancia pareada en la malla.
"""
    )
    _save(path, nb)


def update_nb04() -> None:
    path = ROOT / "notebooks" / "04_malla_sintetica.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))

    code = "".join(_cell(nb, "f11b32ab")["source"])
    marker = "display(pd.crosstab(par_curvas.n_real, par_curvas.veredicto))\n"
    addition = r"""

# El CSV de conclusiones usa el R² del test intocado. Mostrar ambos evita
# interpretar una diferencia de selección (val_mse) como una contradicción.
par_curvas_test = delta_pareado(df_curvas, metrica="test_r2",
                                presupuesto="n_synth", mas_es_mejor=True)
print("\nVeredictos paralelos en R² del test real:")
display(pd.crosstab(par_curvas_test.n_real, par_curvas_test.veredicto))
"""
    if "par_curvas_test" not in code:
        code = code.replace(marker, marker + addition)
    _cell(nb, "f11b32ab")["source"] = _source(code)

    _cell(nb, "31e4e6be")["source"] = _source(
        r"""
#### Lectura

La malla son **465 celdas**: 5 niveles de reales × 5 presupuestos sintéticos × 6
generadores × 3 semillas, más 15 referencias de solo-real. WGAN-GP ya no forma parte de
la matriz activa; Diffusion-TS R61 ocupa exactamente sus 75 celdas.

**1 · El generador importa menos a medida que crece el dato real.** El cociente entre el
mejor y el peor `val_mse` cae de **2,12× con 250 reales a 1,26× con 20.000**. La elección
es crítica en escasez y casi marginal al final de la curva.

**2 · Diffusion-TS ayuda sin introducir una zona sistemáticamente tóxica.** Sobre
`val_mse` obtiene **8 mejoras, 0 empeoramientos y 17 celdas no concluyentes**. En el R² del
test real son **9 mejoras, 1 empeoramiento y 15 no concluyentes**; la única pérdida es
pequeña y aislada (20.000 reales + 500 sintéticos, ΔR² −0,0037). En N=3.000 la mejora
crece de +0,002 a +0,061 R² al pasar de 500 a 20.000 sintéticos.

**3 · El efecto se concentra entre escasez severa y el régimen medio.** Con 250 reales,
Diffusion-TS mejora el test desde 1.000 sintéticos; con 1.000 reales los cinco estimadores
son positivos pero cuatro quedan bajo el ruido entre semillas; con 3.000 hay tres mejoras
claras desde 3.000 sintéticos. A partir de 10.000 reales el margen práctico se comprime a
milésimas.

**4 · No aparece un óptimo interior estable.** La pendiente de `val_mse` frente a
`log(n_synth)` es negativa para Diffusion-TS en los cinco regímenes, igual que para jitter
y RealNVP. Eso apoya usar más sintético dentro del rango ensayado cuando hay pocos reales,
pero no autoriza extrapolar más allá de 20.000 ventanas.
"""
    )

    _cell(nb, "cf44673b")["source"] = _source(
        r"""
## 3 · Análisis crítico y decisión

Se han ejecutado dos diseños con la `gru_l` congelada: **333 celdas** de ratios finos y
**465 celdas** de presupuestos absolutos. En ambos, cada semilla conserva el mismo
submuestreo real al comparar solo-real contra real+sintético. Diffusion-TS se ajustó 15
veces, una por pareja `(N_real, semilla)`, y la muestra DDIM máxima se reutilizó por
prefijos idénticos para cubrir presupuestos solapados sin alterar una sola observación.

### Resultado específico de Diffusion-TS R61

| Evidencia | Resultado | Lectura |
|---|---:|---|
| TSTR | R² 0,4773; 94,72 % de TRTR | empata con RealNVP y supera ampliamente al WGAN histórico |
| Ratios finos | **7 mejora / 0 empeora / 11 NC** | las 6 dosis mejoran con N=3.000 |
| Curvas, test R² | **9 mejora / 1 empeora / 15 NC** | fuerte con pocos/mid datos; marginal con 10k–20k |
| Frente a WGAN | +0,168 R² medio en ratios; +0,128 en curvas | gana 52/54 y 69/75 comparaciones pareadas |
| Frente a RealNVP | −0,010 en ratios; +0,0003 en curvas | empate global |
| Frente a jitter | −0,0215 en ratios; −0,0009 en curvas | el baseline simple sigue siendo muy competitivo |

En N=3.000, el resultado más estable, Diffusion-TS mejora **todos** los ratios: desde
ΔR² +0,0136 con 0,25× hasta +0,0629 con 5×. En N=1.000 las seis medias también son
positivas (+0,095 a +0,158), pero la gran dispersión del solo-real impide declararlas
concluyentes con tres semillas. Con N=250 el efecto es no monótono y solo 5× concluye
como mejora (+0,0765).

### Contraste de hipótesis

**H1 · "El sintético ayuda sobre todo con pocos reales" → MATIZADA.** El potencial bruto
es mayor con 250–1.000 reales, pero también lo es el ruido de optimización. La evidencia más
limpia aparece con 3.000; a partir de 10.000 la ganancia se reduce a milésimas.

**H2 · "Existe un óptimo intermedio" → NO OBSERVADO EN EL RANGO.** Para N=3.000 la
ganancia de Diffusion crece de forma sostenida entre 0,25× y 5×, y en la malla absoluta
hasta 20.000 sintéticos. La irregularidad con N=250 impide convertirlo en una ley general.

**H3 · "La gaussiana será difícil de batir" → REFUTADA.** En ratios acumula 0 mejoras y
3 empeoramientos; su falta de colas y clustering se transforma en sesgo downstream.

**H4 · "Las redes generativas pierden donde hacen falta" → REFUTADA COMO CLASE.** El VAE
no aporta mejoras en ratios, pero Diffusion-TS y RealNVP son los dos modelos aprendidos más
fuertes. El fallo era del WGAN-GP concreto, no de que el generador fuese neuronal.

**H5 · "Jitter es difícil de batir" → CONFIRMADA.** Promedia +0,059 R² en ratios, por
delante de RealNVP (+0,0475) y Diffusion (+0,0375). En curvas los tres empatan en la práctica
(+0,0508, +0,0496 y +0,0499 respectivamente).

### La auditoría predice el descarte, no el ganador absoluto

Al cruzar los seis generadores activos con la ganancia de ratios, el discriminative AUC es
el mejor predictor global (Spearman **−0,94**), seguido de W1 por columna (−0,83), error
de correlación de rangos (−0,77) y TSTR (+0,71). El TSTR ordena Diffusion primero, pero la
malla de ratios la coloca tercera. La conclusión metodológica es más sobria: la auditoría
detecta bien generadores peligrosos; entre candidatos fuertes hace falta la malla.

### Decisión definitiva

> **Sí merece la pena sustituir WGAN-GP por Diffusion-TS R61 en el pipeline activo.** La
> mejora es grande y consistente en TSTR y en ambas mallas. **No merece convertirse en el
> único generador por defecto:** globalmente empata con RealNVP y jitter, cuesta más que el
> baseline y conserva peor la escala y el clustering que RealNVP. Su caso de uso más claro
> es la ampliación con aproximadamente 3.000 ventanas reales y presupuestos sintéticos de
> 1×–6,7×; con 10.000–20.000 reales el beneficio práctico casi desaparece.

### Limitaciones

1. Solo hay tres semillas; en N=250 y N=1.000 muchos efectos quedan bajo el ruido.
2. Se ha probado una configuración de Diffusion-TS, un split temporal y un target.
3. La reutilización por prefijos es exacta y auditable, pero comparte el mismo ajuste del
   generador entre presupuestos: los tests downstream sí son independientes por semilla.
4. R61 garantiza igualdad de información con los demás modelos; R81 queda excluido porque
   habría dado a Diffusion veinte retornos adicionales que ningún rival ve.
"""
    )
    _save(path, nb)


if __name__ == "__main__":
    update_nb03()
    update_nb04()
    print("Notebooks 03/04 actualizados para Diffusion-TS R61")
