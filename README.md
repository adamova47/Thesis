# Recurrent Self-Organizing Networks
Code investigating Self-Organizing Maps (SOMs) and two recurrent extensions: Recurrent Self-Organizing Maps (RSOMs) and Merge Self-Organizing Maps (MSOMs).

The project explores how self-organizing maps can represent static data and temporal sequences, how different model configurations affect map quality, and whether learned map representations can be used for sequence reconstruction.

The repository contains model implementations, experiment scripts, datasets, saved results, visualizations, and documentation accompanying the thesis.

## Overview
A Self-Organizing Map (SOM) is an unsupervised neural network that maps high-dimensional input data onto a lower-dimensional grid. 

During training, neurons develop prototype vectors that represent patterns in the input data. Nearby neurons are encouraged to represent similar inputs, helping preserve topological relationships.

Traditional SOMs process inputs independently. Recurrent variants extend this idea by incorporating temporal context, allowing the representation to depend on both the current input and information from previous steps.

This project investigates three architectures:
- SOM: establishes a baseline for learning representations of static data.
- RSOM: incorporates recurrent activity and contextual information into the matching process.
- MSOM: incorporates a context descriptor derived from previous map activity when selecting the winning neuron.

The models are evaluated using multiple datasets and configurations, with particular attention to quantization error, temporal behavior, neuron utilization, and sequence reconstruction.

For the theoretical background and detailed research results, consult the thesis PDF included in the repository.

## Materials
- **Thesis (PDF):** [Recurrent_self-organizing_networks.pdf](./docs/Diplomova_Praca.pdf)
- **Pre-defense presentation (PPTX):** [Recurrent-Self-Organizing-Networks.pptx](./docs/Recurrent-Self-Organizing-Networks.pptx)

## Repository Structure
Simplified overview of the main components:
```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── Diplomova_Praca.pdf
│   ├── Recurrent-Self-Organizing-Networks.pptx
│   └── PROJECT_GUIDE.md
├── src/
│   └── models/
│       ├── SOM.py
│       ├── SOM_vectorized.py
│       ├── SOM_cupy.py
│       ├── RSOM.py
│       ├── RSOM_cp_vectorized.py
│       ├── MSOM.py
│       ├── MSOM_cp_vectorized.py
│       └── utils.py
└── experiments/
    ├── seeds/
    ├── mackey_glass/
    ├── nico/
    └── reconstruction/
        ├── RSOM/
        └── MSOM/
```

## Main components
src/models/
Contains the implementations of SOM, RSOM, and MSOM, including vectorized and GPU-oriented variants.
The shared utils.py module provides helper functionality used by the experiment scripts, including plotting and data-processing utilities.

experiments/seeds/
Contains experiments using the Seeds dataset to investigate SOM configurations and evaluate the resulting maps.

experiments/mackey_glass/
Contains experiments using the Mackey–Glass time series to investigate how recurrent map architectures represent temporal dynamics.

experiments/nico/
Contains experiments using NICO grasp-motion sequences. The scripts load sequences, normalize their features, train recurrent models, and evaluate map utilization and temporal representation.

experiments/reconstruction/
Contains utilities and experiments for reconstructing sequences or map-state histories from trained models.
The RSOM and MSOM subdirectories contain different reconstruction approaches. See the documentation in those directories for details.

docs/
Contains the thesis, presentation, and supplementary documentation.

## Requirements
- Python compatible with the versions required by the project's dependencies.
- Dependencies listed in requirements.txt.
- A compatible NVIDIA GPU and CUDA environment for GPU-based experiments using CuPy.
- Dataset files required by the selected experiment.

## Evaluation Metrics
The experiment scripts use several measures to evaluate trained maps.

### Quantization Error (QE)
Quantization error measures how far input vectors are from their corresponding winning prototype vectors, typically using Euclidean distance.
A lower QE generally indicates that the prototypes represent the evaluated inputs more closely. Comparisons are meaningful only when preprocessing, evaluation data, and metric definitions are compatible.

### Temporal Quantization Error
Temporal quantization error is used by the recurrent implementations to evaluate representation quality while accounting for temporal context.
Its precise definition depends on the implementation. Consult the corresponding model's evaluation code before comparing temporal quantization error across architectures.

### Best Matching Unit (BMU)
The Best Matching Unit is the neuron selected as the best match for an input under the model's matching rule.
A sequence of BMUs represents a sequence of map states and can be used to examine how a model responds to temporal data.

### Hit Maps and Dead Neurons
A hit map records how frequently each neuron wins during evaluation.
- Neurons with many hits represent frequently encountered patterns.
- Neurons with few hits may represent uncommon patterns.
- Neurons that never win during the evaluation are commonly called dead neurons.
Hit maps help assess how effectively the map is being utilized.

### Entropy
Entropy summarizes the distribution of winning neurons. A more distributed set of BMU hits generally produces higher entropy than a distribution concentrated on a few neurons.
Entropy should be interpreted alongside QE, map dimensions, dataset characteristics, and other measures. Higher entropy does not necessarily mean a better representation.

## Sequence Reconstruction
The reconstruction experiments investigate how trained models can be used to infer input sequences or plausible histories of map activity.
The repository contains separate approaches for RSOM and MSOM.
- **RSOM reconstruction:** investigates inverse mapping from learned representations to estimated inputs.    
- **MSOM backward decoding:** explores predecessor BMU chains that lead to a selected target neuron.
