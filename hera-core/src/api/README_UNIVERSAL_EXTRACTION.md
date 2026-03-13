# Universal Extraction Node (Hera API)

## Visión General
Endpoint unificado (`/v1/hera/extract`) que recibe archivos (Excel, PDF, Imágenes) y devuelve JSON estructurado vía SSE.

**Archivo principal:** `universal_extract.rs`

## Filosofía: LLM para Inteligencia, Rust para Velocidad

> ⚡ **Regla de oro:** El LLM entiende semántica (qué columna es qué). Rust procesa datos (miles de filas). Nunca pasar datos completos al LLM.

### Flujo Excel (~2s total)
```
Excel → Calamine (parse instant) → Headers al LLM (~1-2s) → Rust mapea filas (instant)
```
1. `calamine` extrae filas/columnas como texto pipe-delimited — **instant**
2. LLM recibe SOLO los encabezados (~20 tokens): `0: "cups", 1: "servicio", 2: "tarifa particular"...`
3. LLM responde con mapeo de índices (~10 tokens): `{"name":1,"originalPrice":2,"moviloPrice":3,"discountPercentage":-1}`
4. Rust usa esos índices para mapear TODAS las filas — **instant**

### Flujo OCR (tabular: ~2s, texto libre: ~30s)
```
Imagen/PDF → Tesseract OCR → ¿Texto tabular?
    ├── SÍ → Headers al LLM → Rust mapea filas (~2s total)
    └── NO → LLM full structuring (~30s, token-a-token)
```

### ¿Por qué no fuzzy matching con keywords hardcoded?
Porque **falla**. Un matcher de keywords no puede entender que:
- "CUPS" es un código, no un nombre
- "TARIFA PREFERENCIAL" es el precio bajo, no el alto
- "VALOR IPS" = precio original en contexto colombiano

Solo un LLM tiene la capacidad semántica de mapear correctamente.

## LLM Header Mapper
El LLM recibe una instrucción fija y solo varía la lista de encabezados:

**Input:** `Mapea estos encabezados: 0: "cups", 1: "servicio", 2: "tarifa particular", 3: "tarifa movilo"`

**Output:** `{"name":1,"originalPrice":2,"moviloPrice":3,"discountPercentage":-1}`

- `name`: columna con nombre del servicio/procedimiento
- `originalPrice`: columna con precio normal/regular/particular (el más alto)
- `moviloPrice`: columna con precio preferencial/convenio/club (el más bajo)
- `discountPercentage`: columna con porcentaje de descuento (-1 si no existe)

## SSE Events
| Step          | Descripción                                    |
|---------------|------------------------------------------------|
| `classifying` | Archivo recibido, detectando formato           |
| `routing`     | Ruta seleccionada (Excel / OCR)                |
| `structuring` | LLM mapeando headers o procesando texto        |
| `finished`    | JSON listo con `data` y tiempo en ms            |
| `error`       | Error en cualquier fase                        |

## Esquema JSON de Salida
```json
[
  {
    "name": "Consulta General",
    "originalPrice": 150000,
    "moviloPrice": 120000,
    "discountPercentage": 20,
    "category": "",
    "description": ""
  }
]
```
