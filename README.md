# HRIME-MSP
Hierarchical RIME algorithm with multiple search preferences for extreme learning machine training

## Highlights
• We propose a hierarchical RIME algorithm with multiple search preferences (HRIME-MSP) for solving complex optimization problems.  
• The proposed HRIME-MSP partitions the swarm population into the superior, the borderline, and the inferior layer.  
• Individuals in the different layers have distinctive search preferences.  
• Comprehensive experiments on CEC benchmarks, engineering problems, and extreme learning machine (ELM) training tasks are conducted.  
• The experimental results and statistical analyses confirm the efficiency and effectiveness of our proposed HRIME-MSP.  

## Abstract
This paper introduces a hierarchical RIME algorithm with multiple search preferences (HRIME-MSP) to tackle complex optimization problems. Although the original RIME algorithm is recognized as an efficient metaheuristic algorithm (MA), its reliance on a single, simplistic search operator poses limitations in maintaining population diversity and avoiding premature convergence. To address these challenges, we propose a hierarchical partition strategy that categorizes the population into superior, borderline, and inferior layers based on their fitness values. Individuals in the superior layer utilize an exploitative local search operator, individuals in the borderline layer inherit the expert-designed soft- and hard-rime search operators from the original RIME algorithm, and individuals in the inferior layer employ the explorative OBL method. We conduct comprehensive numerical experiments on the CEC2017 and CEC2022 benchmarks, six engineering problems, and extreme learning machine (ELM) training tasks to evaluate the performance of HRIME-MSP. Twelve popular and high-performance MA approaches are used as competitor algorithms. The experimental results and statistical analyses confirm the effectiveness and efficiency of HRIME-MSP across various optimization tasks. These findings practically support the scalability and applicability of HRIME-MSP as an advanced optimization technique for diverse real-world applications.

## Citation
@article{Zhong:25,  
title = {Hierarchical RIME algorithm with multiple search preferences for extreme learning machine training},  
journal = {Alexandria Engineering Journal},  
volume = {110},  
pages = {77-98},  
year = {2025},  
issn = {1110-0168},  
doi = {https://doi.org/10.1016/j.aej.2024.09.109 },  
author = {Rui Zhong and Chao Zhang and Jun Yu},  
}

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. ELM model and datasets in classification are provided by mafese==0.1.8 and intelelm==1.0.3 libraries.

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
