# Generative-AI-Bioinformatics-Biology
This repo contains tutorials, large language model-based tools and approaches for applying generative AI to Bioinformatics and biology





# Bioinformatics 

 [Centre of Bioinformatics Research and Technology (CBIRT)] (https://cbirt.net/)
 

* scGPT


  This is the official codebase for scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI.

This package is built on the foundation of the scGPT model, which is the first single-cell foundation model built through generative pre-training on over 33 million cells. The scGPT model incorporates innovative techniques to overcome methodology and engineering challenges specific to pre-training on large-scale single-cell omic data. By adapting the transformer architecture, we enable the simultaneous learning of cell and gene representations, facilitating a comprehensive understanding of cellular characteristics based on gene expression.

This package provides a set of functions for data preprocessing, visualization, and model evaluation that are compatible with the scikit-learn library. These functions enable users to preprocess their single-cell RNA-seq data, visualize the results, and evaluate the performance of the scGPT model on downstream tasks such as multi-batch integration, multi-omic integration, cell-type annotation, genetic perturbation prediction, and gene network inference.


 | Model Name               | Description                                        | Download |
|--------------------------|----------------------------------------------------|----------|
| whole-human (recommended)| Pretrained on 33 million normal human cells.       | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
| continual pretrained     | For zero-shot cell embedding related tasks.        | [link](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing) |
| brain                    | Pretrained on 13.2 million brain cells.            | [link](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |
| blood                    | Pretrained on 10.3 million blood and bone marrow cells. | [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
| heart                    | Pretrained on 1.8 million heart cells              | [link](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |
| lung                     | Pretrained on 2.1 million lung cells               | [link](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |
| kidney                   | Pretrained on 814 thousand kidney cells            | [link](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |
| pan-cancer               | Pretrained on 5.7 million cells of various cancer types | [link](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |

   
  https://scgpt.readthedocs.io/en/latest/
  
  https://github.com/bowang-lab/scGPT






* RiboNucleic Acid (RNA) Language Model

  RiboNucleic Acid Language Model - RiNALMo.

  Ribonucleic acid (RNA) plays a variety of crucial roles in fundamental biological processes. Recently, RNA has become an interesting drug target, emphasizing the need to improve our understanding of its structures and functions. Over the years, sequencing technologies have produced an enormous amount of unlabeled RNA data, which hides important knowledge and potential. Motivated by the successes of protein language models, we introduce RiboNucleic Acid Language Model (RiNALMo) to help unveil the hidden code of RNA. RiNALMo is the largest RNA language model to date with 650 million parameters pre-trained on 36 million non-coding RNA sequences from several available databases. RiNALMo is able to extract hidden knowledge and capture the underlying structure information implicitly embedded within the RNA sequences. RiNALMo achieves state-of-the-art results on several downstream tasks. Notably, we show that its generalization capabilities can overcome the inability of other deep learning methods for secondary structure prediction to generalize on unseen RNA families.


   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/94e735d6-0356-437c-9f1b-a559287efbd9)


  https://github.com/lbcb-sci/RiNALMo

  https://sikic-lab.github.io/





* Protein language models scaling laws

  The goal of this project is to uncover the best approach to scale large protein language models (ie., learn scaling laws for protein language models) and then publicly release a suite of optimally-trained large protein language models.

  https://github.com/OpenBioML/protein-lm-scaling
  



* ProLLaMA

  A Protein Large Language Model for Multi-Task Protein Language Processing

  Large Language Models (LLMs), including GPT-x and LLaMA2, have achieved remarkable performance in multiple Natural Language Processing (NLP) tasks. Under the premise that protein sequences constitute the protein language, Protein Large Language Models (ProLLMs) trained on protein corpora excel at de novo protein sequence generation. However, as of now, unlike LLMs in NLP, no ProLLM is capable of multiple tasks in the Protein Language Processing (PLP) field. We introduce a training framework to transform any general LLM into a ProLLM capable of handling multiple PLP tasks. Specifically, our framework utilizes low-rank adaptation and employs a two-stage training approach, and it is distinguished by its universality, low overhead, and scalability. Through training under this framework, we propose the ProLLaMA model, the first known ProLLM to handle multiple PLP tasks simultaneously. Experiments show that ProLLaMA achieves state-of-the-art results in the unconditional protein sequence generation task. In the controllable protein sequence generation task, ProLLaMA can design novel proteins with desired functionalities. In the protein property prediction task, ProLLaMA achieves nearly 100% accuracy across many categories. The latter two tasks are beyond the reach of other ProLLMs.


   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/2b6a32b8-8ea0-4795-a3c4-5e22a72428b0)


  https://github.com/PKU-YuanGroup/ProLLaMA
  

  
* PocketGen

  PocketGen: Generating Full-Atom Ligand-Binding Protein Pockets


    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fb98e4b7-195d-42a0-a57b-333406c76d28)

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4c9df7c9-5df7-43a3-82a5-f4fc45f81df6)

  

  https://github.com/zaixizhang/PocketGen




* PTM-Mamba

 A PTM-Aware Protein Language Model with Bidirectional Gated Mamba Blocks

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/2072de29-4af0-484f-9762-b4428a7c9523)


 https://github.com/programmablebio/ptm-mamba

 https://huggingface.co/ChatterjeeLab/PTM-Mamba
 



* PNAbind

  A python package and collection of scripts for computing protein surface meshes, chemical, electrostatic, geometric features, and building/training graph neural network models of protein-nucleic acid binding



    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/42a3df23-01e6-48d4-b97f-acdc40e2c050)


  https://github.com/jaredsagendorf/pnabind

  

* GenePT


  GenePT is a single-cell foundation model that leverages ChatGPT embeddings to tackle gene-level and cell-level biology tasks. This project is motivated by the significant recent progress in using large-scale (e.g., tens of millions of cells) gene expression data to develop foundation models for single-cell biology. These models implicitly learn gene and cellular functions from the gene expression profiles, which requires extensive data curation and resource-intensive training. By contrast, GenePT offers a complementary approach by using NCBI text descriptions of individual genes with GPT-3.5 to generate gene embeddings. From there, GenePT generates single-cell embeddings in two ways: (i) by averaging the gene embeddings, weighted by each gene’s expression level; or (ii) by creating a sentence embedding for each cell, using gene names ordered by the expression level.

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fb252dee-2b14-4f60-ab34-2acecb940131)


  https://github.com/yiqunchen/GenePT



  
* pLMs-interpretability


  https://github.com/zzhangzzhang/pLMs-interpretability
  

* DNABERT_S
  
  DNABERT_S: Learning Species-Aware DNA Embedding with Genome Foundation Models
  
  DNABERT-S is a foundation model based on DNABERT-2 specifically designed for generating DNA embedding that naturally clusters and segregates genome of different species in the embedding space, which can greatly benefit a wide range of genome applications, including species classification/identification, metagenomics binning, and understanding evolutionary relationships.

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/207ef2c5-659e-4b76-95cc-3e5dd52b4345)

 https://github.com/MAGICS-LAB/DNABERT_S


* GenerRNA

 GenerRNA is a generative RNA language model based on a Transformer decoder-only architecture. It was pre-trained on 30M sequences, encompassing 17B nucleotides.

Here, you can find all the relevant scripts for running GenerRNA on your machine. GenerRNA enable you to generate RNA sequences in a zero-shot manner for exploring the RNA space, or to fine-tune the model using a specific dataset for generating RNAs belonging to a particular family or possessing specific characteristics.

The newest version with trained model is placed in the following [repository]: (https://huggingface.co/pfnet/GenerRNA)

  https://github.com/pfnet-research/GenerRNA


* DeepGO-SE
  
  DeepGO-SE: Protein function prediction as Approximate Semantic Entailment

  DeepGO-SE, a novel method which predicts GO functions from protein sequences using a pretrained large language model combined with a neuro-symbolic model that exploits GO axioms and performs protein function prediction as a form of approximate semantic entailment.

This repository contains script which were used to build and train the DeepGO-SE model together with the scripts for evaluating the model's performance.

 https://github.com/bio-ontology-research-group/deepgo2



* ChatDrug

  Conversational Drug Editing Using Retrieval and Domain Feedback


    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/3c8e52f9-bf68-4406-9816-8a3fff2363c8)
  

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/66303821-4296-4054-958c-9dfeb6344816)


  https://github.com/chao1224/ChatDrug


* reguloGPT

  reguloGPT: Harnessing GPT for Knowledge Graph Construction of Molecular Regulatory Pathways

   Molecular Regulatory Pathways (MRPs) are crucial for understanding biological functions. Knowledge Graphs (KGs) have become vital in organizing and analyzing MRPs, providing structured representations of complex biological interactions. Current tools for mining KGs from biomedical literature are inadequate in capturing complex, hierarchical relationships and contextual information about MRPs. Large Language Models (LLMs) like GPT-4 offer a promising solution, with advanced capabilities to decipher the intricate nuances of language. However, their potential for end-to-end KG construction, particularly for MRPs, remains largely unexplored.
  

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/184bdd45-2551-4240-95b4-1fdbed7d0937)

  
  
  https://github.com/Huang-AI4Medicine-Lab/reguloGPT

  
* nanoBERT

  The model is a heavy sequence specfic transformer to predict amino acids in a given position in a query sequence.
  
  https://huggingface.co/NaturalAntibody


* GPCR-BERT

  This repository contains the necessary codes for running the GPCR-BERT.
  

  https://github.com/Andrewkimmm/GPCR-BERT


 * Lingo3DMol

  Lingo3DMol: Generation of a Pocket-based 3D Molecule using a Language Model

  Lingo3DMol is a pocket-based 3D molecule generation method that combines the ability of language model with the ability to generate 3D coordinates and geometric deep learning to produce high-quality molecules.

   Online Service: https://sw3dmg.stonewise.cn/

  https://github.com/stonewiseAIDrugDesign/Lingo3DMol
  
  
* ColabFold

  Making Protein folding accessible to all!
  
    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/62878af8-586e-4a50-8193-15f0bafe9831)

  https://colabfold.mmseqs.com/
  
  https://github.com/sokrypton/ColabFold

* OmegaFold

   OmegaFold: High-resolution de novo Structure Prediction from Primary Sequence

  
  OmegaFold is the first computational method to successfully predict high-resolution protein structure from a single primary sequence alone. Using a new combination of a protein language model that allows us to make predictions from single sequences and a geometry-inspired transformer model trained on protein structures, OmegaFold outperforms RoseTTAFold and achieves similar prediction accuracy to AlphaFold2 on recently released structures.

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/69a41650-375c-475c-8a39-16ed6dd09ae2)

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/2326767c-2d4b-4e65-8a93-3de456da0635)
 
   
  https://github.com/HeliXonProtein/OmegaFold
  

* esm

  Evolutionary Scale Modeling (esm): Pretrained language models for proteins

  https://github.com/facebookresearch/esm
  

* ViSNet

  AI-powered ab initio biomolecular dynamics simulation
  
  ViSNet (shorted for “Vector-Scalar interactive graph neural Network”) is an equivariant geometry-enhanced graph neural for molecules that significantly alleviate the dilemma between computational costs and sufficient utilization of geometric information.

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/636e954b-3266-49d9-8ebf-5284b29ed591)


  https://github.com/microsoft/AI2BMD/tree/ViSNet

