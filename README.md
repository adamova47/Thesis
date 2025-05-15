# Recurrent self-organising networks

This repository contains the implementation of Recurrent Self-Organizing Maps (RSOM) and Merge Self-Organizing Maps (MSOM) in Python. The goal of this project is to implement, test, and visualize these algorithms, focusing on sequence learning, sequence reconstruction, and continuous learning to address catastrophic forgetting.

The repository includes:
- Source code for SOM, RSOM and MSOM implementations
- Experiment logs and results
- Project timeline and task calendar

This project is conducted as part of the MSc thesis in Applied Informatics at Comenius University in Bratislava.

17.2. - 16.3.
- Studying Python libraries necessary for implementation.
- Learning the basics of neural networks and algorithms.
- Reviewing articles by Kohonen and Strickert & Hammer (Merge SOM).
- Analyzing the NeuroNet code in C++ with a focus on SOM and RecSOM implementation.

17.3. - 23.3.
- Starting the implementation of the basic 2D Kohonen SOM.
- Initial attempts at weight initialization ('uniform').
- Basic setup of grid metrics ('euclid').
- Implementation of the basic training algorithm without metric monitoring.

24.3. - 30.3.
- Extending the SOM with flexible initialization methods: 'data_range', 'sample'.
- Adding additional grid metrics: 'manhattan', 'chebyshev'.
- Implementation of basic neighborhood functions: 'gaussian', 'bubble'.
- Testing the SOM performance on a small dataset.

31.3. - 6.4.
- Implementing further initialization methods: 'pca', 'kmeans'.
- Adjustable learning rate and radius.
- Monitoring quantization error and weight update history, and adding quantization error visualization.

7.4. - 13.4.
- Additional SOM testing (membership maps, U-matrix, attribute heatmaps).
- Adding more neighborhood functions: 'epanechnikov', 'triangular', 'inverse'.
- Implementation of a toroidal grid.
- Setting up the CUDA environment and testing basic parallelization using multiprocessing.

14.4. - 27.4.
- Extending SOM to RecSOM, implementing the context layer.
- Reimplementing the BMU algorithm for RecSOM.
- Testing basic RecSOM functionality (without detailed testing).

28.4. -
- Implementing MSOM, merging input and context weights.
- Testing the BMU algorithm for MSOM on a small dataset.
