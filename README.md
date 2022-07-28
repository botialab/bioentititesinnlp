# Machine Learning for Entity Detection in Biological Natural Language Questions

## Authors
- [Antonio Arques Acosta](https://github.com/Antonio4132)

## Introduction to our Work

In this repository we show the research for a entity detection system in natural languaje. For example in the sentence

> Can malaria be found in mice's blood?

Some entities can be detected:

- **Phenotype**: Malaria is a disease found in the question.
- **Tissue**: Blood is a tissue, also found in its main name in the question.
- **Specie**: Mus Musculus is the technical name for mouse, which in this question is found under a more common name, mice.

So as to be expected, natural languaje has some tricks that need to be comprehended and tackled in the designed systems. For instance, some entities may not appear directly in the sentence and thus need to be infered from the context, other may be under the use of a synonym or a common name... 

This is part of a large biomedical workflow, which tries to solve natural languaje biomedical questions, detecting the desired entities, linking them to data structures such as ontologies and enventually deploying automated workflows to solve the question. As we advance on our research, we design experiments in order to answer the questions that appear, for instance, evaluating the performance of NER models, questions generation techniques... This experiments are briefly explained in this repository, the code and results are presented with a tutorial about their usage.

It is worth mentioning that this experiments are made under an unified framework called Nextlow. This provide a clear pipeline to divide de experiment in independent processes that comunicate with channels. The tool makes the experiemnts portable, reproducible and easy to follow. 

## Initial Set-Up

Even if the project is meant to be portable, it needs some programs to run the workflow. 

- Nextflow: This pipeline is used to run the processes correctly, so it needs to be installed. A quick tutorial can be found [here](https://www.nextflow.io/).
- Anaconda: Nextflow autonomously creates anaconda enviorments to every process that needs it. This makes the whole set-up quite quick. To install anaconda, full version can be found [here](https://www.anaconda.com/). For a lightwheight although functional version, use [miniconda](https://docs.conda.io/en/latest/miniconda.html).
- CUDA and Pytorch: Some experiments use Pytorch with a GTX 3060 Ti GPU. It must be noted that the PyTorch version is selected accordingly to the CUDA drivers version for the foresaid GPU. If changes were needed, a different PyTorch and CUDA version should be specified in the .yml files. 


## Experiments

Here we explain the main experiments done, the reason behind them, how they work and their results. 

### EE-NER

This first experiment explores the use of Named Entity Recognition Models, or NER, to try and solve the task. Specifically, it tries to create a labeled dataset of questions using NER models in conjunction with the ontology. This diagram explains how it works:

![EE-NER DFD](/images/EE_NER-DFD.png)

The following processes are launched:

- **Detect Entitie**s: Using Stanza biomedical NER models, entities are detected from a small corpus of hand-made questions. 

> Is breast cancer a terminal disease for women? (phenotype: breast cancer, specie: homo sapiens)

- **Create Template**: Once the questions are created, they are transformed into a template, replacing entities with keywords. 

> Is **PHENOTYPE** a terminal disease for **SPECIE**?

- **Create Questions**: Using the ontology and the Owlready2 python package to move through it, new questions are created by replacing the keywords with real values from real single cell projects, ensuring the veracity of the resultant questions. 

> Is **Lung Cancer** a terminal disease for **Mus Musculus**?


This experiments aims to expand and label an initial, raw corpus of questions. In the initial version, a 30 question corpus is used, but if good results were achieved, this would be upgraded with a bigger raw corpus of collected questions via web scrapping or other thechniques.

The detection system proved ineffective, detecting 16% of species and 20% of phentypes. Even if the detecting results were good, the overall design of NER models present flaws. The pretrained models need to be quite specific, and even if so, it can't create labeled questions with implicit entities or synonyms.

#### Usage Tutorial

The next command is used to launch the nextflow script:

```
nextflow run ee-ner.nf
```

This experiment create some csv files as it proceeds with the overall creation of the augmented dataset.


### EE-Specie

This experiments tries to detect entities using transformers, to create the labeled dataset, and to classify the entities. For simplicity, specie will be the only entity detected. Looking at the experiment diagram 

![EE-Specie DFD](/images/EE_Specie.png)

It can be seen as 3 blocks:

- **Abstract Extraction**: Using E-Utils API Web, biomedical abstracts are extracted from the Pubmed repository for every specie.
- **Question Generation**: Questions from every abstract are created, so that if a question is created from a Homo Sapiens question, it has that specie as label. The questions are generated using QG Transformers, from [this Github](https://github.com/patil-suraj/question_generation).
- **Model Fine-Tuning**: Once we have a labeled dataset, a base BERT model is fine tuned by adding a classification (softmax) neural network for the existing species. 

The question generation aproach proves to be quite interesting, as it can create a lot of questions on the biomedical subject. Even if so, most of them lack quality or context. This eventually leds to a quite poor model performance, as shown in the following graph. 


#### Usage Tutorial

It's used as follows:

```
nextflow run E0.nf
```

No more tweaks are needed, as it just only works for the species entity in a pre-defined enviorment. 

### EE-Bert-T5

This last experiment tries a different transformer approach, focusing on the semantic knowledge of them. Using [Sentence Transformers](https://www.sbert.net/), we can detect which label of a said entity is closest to a question, if any. This is done using embeddings and cosine similarity. With sentence transformers, the embeddings are created by pooling the encoder output of the transformer. The next diagram shows the process:

![EE-BERT-T5 DFD](/images/EE_BERT_T5_DFD.png)

Overall, it shares the main architecture as the last experiment, but it has some modifications so that questions can be created from any entity present in the ontolgy. Instead of the fine-tuned BERT, a model is evaluated with Top 1, 3 and 5 accuraccy in the created dataset. A filter is also designed, so that any question that is over 2 Standard Errors from the mean Cosine Similary with its question is discarde. This filter tries to tackle the low quality of some questions. It will prove to be useful but better semantic filters will be created in later upgrades. 

#### Usage Tutorial

When executing the nextflow command, you have the following parameters:
1. entity: This parameter specifies the name of the folder in which the values are stored
of the entity and other files.
2. kv: kv or keyword, is the name of the ontology class from which to extract information.
3. model: specifies the model to use, the values are ”bert” or ”t5”.
4. finetune: this parameter can be activated with yes, in which case the program searches for a model
trained with model adaptation to use.

For instance:

```
nextflow run E1.nf –entity Specie –kw Specie –model bert –finetune yes
```
