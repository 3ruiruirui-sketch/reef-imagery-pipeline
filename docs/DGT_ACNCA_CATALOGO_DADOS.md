# Catálogo de Dados — Serviço DGT/ACNCA
**Data de investigação:** 2026-06-16  
**STAC API Base URL:** `https://dgt-be.a.incd.pt:8081/`  
**JupyterHub Base URL:** `https://dgt-jupyterhub.d.acnca.pt/`  
**Utilizador JupyterHub:** `jsoares`

> **Nota de segurança:** As credenciais de acesso (tokens, chaves S3) estão guardadas no ficheiro `.env` da raiz do projeto. Este ficheiro está excluído do git via `.gitignore`.

---

## Resumo Executivo

O serviço expõe **39 coleções** de dados geoespaciais via STAC API, organizadas em 4 categorias principais:

| Categoria | Coleções | Cobertura |
|-----------|----------|-----------|
| LiDAR — Nuvem de Pontos | 3 | Portugal Continental + Açores |
| LiDAR — Modelos Digitais | 10 | Portugal Continental + Açores |
| Ortofotos Aéreas | 8 | Portugal Continental |
| Mosaicos Sentinel-2 | 11 | Portugal Continental |
| Açores — Modelos Digitais | 6 | Arquipélago dos Açores |

---

## 1. 🌄 LiDAR — Nuvem de Pontos

| Coleção | Título | Cobertura | Formato | Tamanho típico |
|---------|--------|-----------|---------|----------------|
| `LAZ` | LiDAR – Nuvem de pontos LAS | Portugal Continental | `.laz` | ~107 MB |
| `COPC` | Nuvem de pontos LIDAR - COPC | Portugal Continental | COPC | N/D |
| `ACORES-LAZ` | Nuvem de pontos LIDAR – LAZ | Açores | `.laz` | ~478 MB |

**Propriedades por item:**
- `pc:count` — número de pontos
- `pc:type` — tipo de nuvem
- `pc:encoding` — codificação (ex: LASzip)
- `pc:statistics` — estatísticas de altitude (Zmin, Zmax, média)
- `pc:schemas` — esquema de dimensões (X, Y, Z, Intensity, ReturnNumber, Classification, etc.)
- `proj:wkt2` — projeção cartográfica

**Backend S3:** `stor-001.a.acnca.pt:9000` → buckets `lidar/LAZ/`, `acores/ACORES-LAZ/`

---

## 2. 🏔️ LiDAR — Modelos Digitais (Portugal Continental)

| Coleção | Tipo | Resolução | Tamanho típico/tile |
|---------|------|-----------|---------------------|
| `MDT-2m` | Modelo Digital do Terreno | 2 metros | ~1 MB |
| `MDS-2m` | Modelo Digital de Superfície | 2 metros | ~1 MB |
| `MDT-50cm` | Modelo Digital do Terreno | 50 centímetros | ~16 MB |
| `MDS-50cm` | Modelo Digital de Superfície | 50 centímetros | ~16 MB |

**Formato:** GeoTIFF / COG (Cloud Optimized GeoTIFF)  
**Cobertura:** Portugal Continental  
**Backend S3:** `stor-002.a.acnca.pt:9000` → buckets `lidar/MDT2m/`, `lidar/MDS2m/`, `lidar/MDT50cm/`, `lidar/MDS50cm/`

**Propriedades por item:**
- `eo:bands` — bandas espectrais
- `file:size` — tamanho do ficheiro
- `proj:wkt2` / `proj:projjson` — projeção cartográfica

**Usos principais:**
- MDA (Modelo Digital de Altura) = `MDS - MDT`
- Hillshade, declive, exposição solar
- Análise de inundação (zonas baixas)
- Curvas de nível

---

## 3. 🏝️ LiDAR — Modelos Digitais (Açores)

| Coleção | Tipo | Resolução | Cobertura |
|---------|------|-----------|-----------|
| `ACORES-MDS-1` | Modelo Digital de Superfície | 1 metro | Açores |
| `ACORES-MDE-1` | Modelo Digital de Elevação | 1 metro | Açores |
| `ACORES-MDS-5` | Modelo Digital de Superfície | 5 metros | Açores |
| `ACORES-MDE-5` | Modelo Digital de Elevação | 5 metros | Açores |
| `ACORES-MDS-10` | Modelo Digital de Superfície | 10 metros | Açores |
| `ACORES-MDE-10` | Modelo Digital de Elevação | 10 metros | Açores |

**Extensão temporal:** 2024 (campanha de aquisição: Setembro 2024)  
**Extensão espacial:** `[-32.124°, 36.397°, -24.060°, 40.172°]`  
**Backend S3:** `stor-002.a.acnca.pt:9000` → buckets `acores/ACORES-MDS-*/`, `acores/ACORES-MDE-*/`

---

## 4. 📷 Ortofotos — Portugal Continental

| Coleção | Ano | Resolução | Bandas | Fonte |
|---------|-----|-----------|--------|-------|
| `ORTOS-1995` | 1995 | N/D | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2004` | 2004–2006 | 50 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2007` | 2007 | 50 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2010` | 2010 | 50 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2012` | 2012 | 50 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2015` | 2015 | 50 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2018` | 2018 | 25 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOS-2021` | 2021 | 25 cm | RGBI | Câmara aerofotogramétrica |
| `ORTOSAT-2023` | 2023 | 30 cm | RGBI | Satélite multifonte |

**Formato:** GeoTIFF / COG, compressão JPEG  
**Backend S3:** `stor-002.a.acnca.pt:9000` → bucket `orto/ortos<ano>/`  
**Cobertura temporal:** 1995 → 2023 (série temporal para análise de mudança)

---

## 5. 🛰️ Mosaicos Sentinel-2 (2015–2025)

| Coleção | Ano | Nível |
|---------|-----|-------|
| `MosaicoS2-2015` | 2015 | Level 1C/2A |
| `MosaicoS2-2016` | 2016 | Level 1C/2A |
| `MosaicoS2-2017` | 2017 | Level 1C/2A |
| `MosaicoS2-2018` | 2018 | Level 1C/2A |
| `MosaicoS2-2019` | 2019 | Level 1C/2A |
| `MosaicoS2-2020` | 2020 | Level 1C/2A |
| `MosaicoS2-2021` | 2021 | Level 1C/2A |
| `MosaicoS2-2022` | 2022 | Level 1C/2A |
| `MosaicoS2-2023` | 2023 | Level 1C/2A |
| `MosaicoS2-2024` | 2024 | Level 1C/2A |
| `MosaicoS2-2025` | 2025 (parcial) | Level 1C/2A |

**Extensão espacial:** Portugal Continental `[-9.84°, 36.56°, -5.87°, 42.33°]`  
**Backend:** `stratus.d.incd.pt:8080` → `sentinel/DGT<ano>/`  
**Missão:** Copernicus Sentinel-2

---

## 6. 🗂️ Infraestrutura de Acesso

### Endpoints e Backends S3/MinIO

| Backend | Endpoint | Conteúdo principal |
|---------|----------|--------------------|
| LAZ | `https://stor-001.a.acnca.pt:9000` | Nuvens de pontos (LAZ/COPC) |
| MDT/MDS/Ortos | `https://stor-002.a.acnca.pt:9000` | Rasters (MDT, MDS, Ortofotos) |
| Sentinel | `https://stratus.d.incd.pt:8080` | Mosaicos Sentinel-2 |

### Acesso programático (Python)

```python
import os
from pystac_client import Client
import boto3

# Abrir catálogo STAC
STAC_URL = os.getenv("DGT_STAC_BASE_URL", "https://dgt-be.a.incd.pt:8081/")
api = Client.open(STAC_URL)

# Listar coleções
collections = [c.id for c in api.get_collections()]

# Pesquisar itens por bbox (ex: Lisboa)
bbox = [-9.229, 38.691, -9.089, 38.829]
items = list(api.search(
    collections=["MDT-2m"],
    bbox=bbox,
    max_items=50
).items())

# Cliente S3 para MDT/MDS (Backend 2)
s3_mdt = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL2"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID2"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY2"),
)

# Cliente S3 para LAZ (Backend 1)
s3_laz = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID1"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY1"),
)
```

### Acesso via JupyterHub API

```python
import requests
import os

JHUB_API = os.getenv("JUPYTERHUB_HUB_API_URL")
TOKEN = os.getenv("JUPYTERHUB_API_TOKEN")
headers = {"Authorization": f"token {TOKEN}"}

# Listar conteúdo de uma pasta no servidor Jupyter
user = os.getenv("JUPYTERHUB_USERNAME", "jsoares")
r = requests.get(
    f"https://dgt-jupyterhub.d.acnca.pt/user/{user}/api/contents/Exemplos",
    headers=headers
)
contents = r.json()["content"]
```

### Variáveis de ambiente (`.env`)

```bash
# JupyterHub
JUPYTERHUB_BASE_URL=https://dgt-jupyterhub.d.acnca.pt
JUPYTERHUB_HUB_API_URL=https://dgt-jupyterhub.d.acnca.pt/hub/api
JUPYTERHUB_USER_SERVER_URL=https://dgt-jupyterhub.d.acnca.pt/user/jsoares/
JUPYTERHUB_API_TOKEN=<ver .env>
JUPYTERHUB_USERNAME=jsoares

# STAC API
DGT_STAC_BASE_URL=https://dgt-be.a.incd.pt:8081/

# S3 Backend LAZ
AWS_ENDPOINT_URL1=https://stor-001.a.acnca.pt:9000
AWS_ACCESS_KEY_ID1=<ver .env>
AWS_SECRET_ACCESS_KEY1=<ver .env>

# S3 Backend MDT/MDS
AWS_ENDPOINT_URL2=https://stor-002.a.acnca.pt:9000
AWS_ACCESS_KEY_ID2=<ver .env>
AWS_SECRET_ACCESS_KEY2=<ver .env>
AWS_S3_BUCKET=lidar
```

---

## 7. 💡 Relevância para o Projeto reef_imagery_pipeline

| Dado | Coleção(ões) | Aplicabilidade ao projeto |
|------|-------------|--------------------------|
| Altimetria costeira de alta resolução | `MDT-50cm`, `MDS-50cm` | Transição terra-mar, batimetria litoral |
| Altimetria regional | `MDT-2m`, `MDS-2m` | Cobertura alargada Portugal Continental |
| Nuvem de pontos LiDAR | `LAZ`, `COPC` | Reconstrução 3D de zonas costeiras |
| Nuvem de pontos Açores | `ACORES-LAZ`, `ACORES-MDE-1` | Ilhas com ecossistemas de recife potencialmente relevantes |
| Ortofoto temporal | `ORTOS-2007` → `ORTOS-2021` | Análise de mudança costeira multitemporal |
| Ortofotos recentes | `ORTOSAT-2023` | Referência visual de alta resolução (30 cm) |
| Imagem multiespectral | `MosaicoS2-2015` → `2025` | NDWI, deteção de recifes, turbidez da água |

### Fluxos de trabalho sugeridos

1. **Batimetria costeira:** `STAC search MDT-50cm` → streaming COG → análise de zona intertidal
2. **Mudança costeira:** `ORTOS-2007/2012/2021` → comparação temporal de linhas de costa
3. **Análise multiespectral:** `MosaicoS2-2020/2024` → cálculo NDWI/NDVI → deteção de recifes/algas
4. **Açores:** `ACORES-LAZ` + `ACORES-MDE-1` → pipeline adaptado para ilhas atlânticas

---

## 8. 📚 Notebooks de Referência (JupyterHub)

Localizados em `Exemplos/` no JupyterHub:

| Notebook | Conteúdo |
|----------|----------|
| `00_Guia_de_Utilizacao.ipynb` | Porta de entrada, percurso recomendado |
| `01_Downloads.ipynb` | Download de ficheiros LAZ/MDT/MDS por ID ou bbox |
| `02_Pesquisa_STAC.ipynb` | Exploração do catálogo STAC, pesquisa por bbox e data |
| `03_Areas_Administrativas.ipynb` | Pesquisa por unidades CAOP (Distrito/Município/Freguesia) |
| `04_Processamento_LAZ.ipynb` | Estatísticas, histogramas, visualização 3D de nuvens LAZ |
| `05_Processamento_MDT_MDS.ipynb` | Streaming COG, hillshade, MDA, declive, curvas de nível |
| `06_Risco_Inundacao.ipynb` | Análise exploratória de zonas baixas por freguesia |
