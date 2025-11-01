# Status Classification

Clasificación de series temporales EDM para predecir el estado (Normal, NPT, OD) usando TCN.

## Descripción

Este módulo clasifica series completas de perforación EDM en una de tres categorías:
- **Normal**: Perforación normal
- **NPT**: No product time
- **OD**: Over drill

Usa una arquitectura TCN (Temporal Convolutional Network) con pooling global para producir una predicción por chunk.

## Estructura

```
Status Classification/
├── config.py           # Configuración del experimento
├── model.py           # Modelo TCN para clasificación
├── preprocessing.py   # Preprocesamiento de datos
├── train.py           # Script de entrenamiento
├── evaluate.py        # Script de evaluación
├── run_experiment.py  # Script principal
└── README.md         # Este archivo
```

## Características

### Preprocesamiento
- Carga datos de Option 1 (Voltage, Z)
- Crea chunks superpuestos (length=600, stride=200)
- Opción para incluir derivadas (dV/dt, dZ/dt)
- Normalización con StandardScaler
- Split 90/10 train/validation

### Modelo
- TCN con convoluciones dilatadas
- Global average pooling
- Salida: (batch, 3) - probabilidades por clase

### Entrenamiento
- CrossEntropyLoss con class weights
- Adam optimizer
- ReduceLROnPlateau scheduler
- Early stopping
- Data augmentation (opcional) con ruido gaussiano

## Uso

### Ejecución simple

```bash
cd "Status Classification"
conda activate edm_plotting
python run_experiment.py
```

### Configuración

Edita `config.py` para ajustar hiperparámetros:

```python
# Experiment info
experiment_name = "status_exp_001"

# Preprocessing
chunk_length = 600
stride = 200
normalize = True
include_derivatives = False  # Agregar dV/dt, dZ/dt

# Architecture
channels = [64, 64, 128, 128]
dilations = [1, 2, 4, 8]
kernel_size = 5
dropout = 0.25

# Training
batch_size = 256
learning_rate = 1e-4
num_epochs = 200
early_stopping_patience = 10
```

## Resultados

Los resultados se guardan en:
```
results/{experiment_name}/
├── models/
│   ├── {experiment_name}_best_model.pth
│   └── {experiment_name}_final_model.pth
├── results/
│   └── {experiment_name}_history.json
└── {experiment_name}_config.json
```

## Diferencia con Stage Segmentation

- **Stage Segmentation**: Predice el stage para **cada timestamp** (segmentación temporal)
- **Status Classification**: Predice el status de la serie **completa** o de cada chunk

## Referencias

- TCN: Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)


