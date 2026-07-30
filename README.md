# Artificial Intelligence Course Labs

Dos demostraciones reproducibles de inteligencia artificial:

1. clasificación del conjunto Iris con una red neuronal;
2. preguntas y respuestas extractivas con Transformers.

> **English summary:** Reproducible AI course labs for neural-network
> classification and extractive transformer question answering.

## Mejoras aplicadas

- funciones reutilizables en lugar de scripts monolíticos;
- semillas aleatorias configuradas;
- normalización entrenada únicamente con el conjunto de entrenamiento;
- capa `Input` explícita en Keras;
- carga diferida del modelo Transformer;
- inyección de la tubería de QA para permitir pruebas sin descargar modelos;
- argumentos de línea de comandos.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Clasificador Iris

```powershell
$env:PYTHONPATH='src'
python -m ai_labs.iris_classifier --epochs 60
```

## Preguntas y respuestas

```powershell
$env:PYTHONPATH='src'
python -m ai_labs.transformer_qa
```

La primera ejecución descargará el modelo de Hugging Face. El contexto y las
preguntas de ejemplo están en inglés porque el modelo predeterminado fue
ajustado con SQuAD2 en inglés. El nombre del modelo puede cambiarse con
`--model`.

## Pruebas

Las pruebas no requieren acceso a la red:

```powershell
.\run-tests.ps1
```
