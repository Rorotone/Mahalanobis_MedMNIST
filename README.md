# Clasificación de Imágenes Dermatológicas — ResNet18 + Distancia de Mahalanobis

Proyecto de Machine Learning / Computer Vision que implementa un pipeline completo de clasificación de imágenes médicas sobre el dataset **DermaMNIST**, combinando fine-tuning de ResNet18 con clasificación métrica basada en la distancia de Mahalanobis.

---

## Resultados

| Método | Accuracy (Test) |
|--------|----------------|
| ResNet18 + Softmax (baseline) | 73.02% |
| ResNet18 + Mahalanobis | **79.40%** |
| **Mejora** | **+6.38 pp** |

### Reporte de clasificación (Mahalanobis)

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.51 | 0.38 | 0.43 | 66 |
| 1 | 0.65 | 0.63 | 0.64 | 103 |
| 2 | 0.66 | 0.51 | 0.58 | 220 |
| 3 | 0.62 | 0.35 | 0.44 | 23 |
| 4 | 0.56 | 0.48 | 0.52 | 223 |
| 5 | 0.86 | 0.94 | 0.90 | 1341 |
| 6 | 0.83 | 0.52 | 0.64 | 29 |
| **weighted avg** | **0.78** | **0.79** | **0.78** | **2005** |

---

## Descripción del enfoque

### 1. Fine-tuning de ResNet18
Se parte de ResNet18 preentrenado en ImageNet. Se reemplaza la capa fully connected final para adaptarla a las 7 clases de DermaMNIST y se entrena con:
- Optimizador: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Scheduler: StepLR (step=7, gamma=0.1)
- Epochs: 10
- GPU: NVIDIA T4 (Google Colab)

### 2. Extracción de embeddings
Se elimina la capa `fc` del modelo entrenado para usarlo como extractor de características, generando embeddings de 512 dimensiones por imagen.

### 3. Clasificador por Distancia de Mahalanobis
En lugar de usar softmax directamente, se calculan los centroides de cada clase en el espacio de embeddings y se clasifica cada muestra asignándola a la clase con menor distancia de Mahalanobis, considerando la covarianza de cada clase.

---

## Dataset

**DermaMNIST** — parte del benchmark [MedMNIST v2](https://medmnist.com/)

- 7 clases de lesiones cutáneas
- ~10,000 imágenes de 28×28 px (redimensionadas a 224×224)
- Split: train / val / test

---

## Tecnologías

- Python 3.11
- PyTorch 2.6 + torchvision
- scikit-learn
- medmnist
- Google Colab (GPU T4)

---

## Cómo ejecutar

### Opción 1 — Google Colab (recomendado)

1. Abre el notebook directamente en Colab
2. Activa el entorno de ejecución con GPU: `Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU`
3. Ejecuta todas las celdas en orden

### Opción 2 — Local

```bash
# Instalar dependencias
pip install torch torchvision medmnist scikit-learn numpy pandas

# Abrir el notebook
jupyter notebook Mahalanobis_MedMNIST.ipynb
```

---

## Estructura del notebook

```
1. Instalación de dependencias
2. Carga y preprocesamiento del dataset DermaMNIST
3. Definición del modelo (ResNet18 fine-tune)
4. Entrenamiento y validación (20 épocas, baseline softmax)
5. Extracción de embeddings
6. Clasificador por Distancia de Mahalanobis
7. Evaluación y reporte de clasificación
```

---

## Autor

**Rodrigo Molina Castillo**  
Ingeniería Civil Informática — Universidad Andrés Bello  
[linkedin.com/in/rodrigo-molina-817194176](https://linkedin.com/in/rodrigo-molina-817194176)
