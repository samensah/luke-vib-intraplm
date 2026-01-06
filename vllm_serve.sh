#!/bin/bash
# Serve model with vLLM
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-mini-instruct \
    --port 8000 \
    --host 0.0.0.0


Surveys
* Papadakis, G., Skoutas, D., Thanos, E., & Palpanas, T. (2020). A Survey of Blocking and Filtering Techniques for Entity Resolution. ACM Computing Surveys.
* Christophides, V., Efthymiou, V., Palpanas, T., Papadakis, G., & Stefanidis, K. (2020). An Overview of End-to-End Entity Resolution for Big Data. ACM Computing Surveys.
* Barlaug, N., & Gulla, J. A. (2021). Neural Networks for Entity Matching: A Survey.
Deep Learning Methods
* Mudgal, S., et al. (2018). Deep Learning for Entity Matching: A Design Space Exploration. SIGMOD.
* Li, Y., et al. (2020). Deep Entity Matching with Pre-Trained Language Models. PVLDB.
* Thirumuruganathan, S., et al. (2021). Deep Learning for Blocking in Entity Matching: A Design Space Exploration. PVLDB.
* Wu, R., et al. (2020). ZeroER: Entity Resolution using Zero Labeled Examples. SIGMOD.
LLM Methods
* Peeters, R., & Bizer, C. (2023). Using ChatGPT for Entity Matching. ADBIS.
* Peeters, R., & Bizer, C. (2024). Entity Matching using Large Language Models. arXiv.
* Peeters, R., & Bizer, C. (2024). Fine-tuning Large Language Models for Entity Matching. arXiv.
Benchmarks
* Peeters, R., Der, R. C., & Bizer, C. (2024). WDC Products: A Multi-Dimensional Entity Matching Benchmark. EDBT.
* Köpcke, H., Thor, A., & Rahm, E. (2010). Evaluation of Entity Resolution Approaches on Real-World Match Problems. PVLDB.
* Konda, P., et al. (2016). Magellan: Toward Building Entity Matching Management Systems. PVLDB.
Knowledge Graph Alignment
* Wang, Z., et al. (2018). Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks. EMNLP.
* Sun, Z., et al. (2020). Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation. AAAI.
* Cao, Y., et al. (2019). Multi-Channel Graph Neural Network for Entity Alignment. ACL.
Blocking Methods
* Wang, R., et al. (2024). Neural Locality Sensitive Hashing for Entity Blocking. SIAM SDM.
* Brinkmann, A., Shraga, R., & Bizer, C. (2023). SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines.
* Wang, R., & Zhang, Y. (2024). Pre-trained Language Models for Entity Blocking: A Reproducibility Study. NAACL.
