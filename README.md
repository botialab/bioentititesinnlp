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



## Experiments

Here we explain the main experiments done, the reason behind them, how they work and their results. 

### EE-NER

This first experiment explores the use of Named Entity Recognition Models, or NER, to try and solve the task. Specifically, it tries to create a labeled dataset of questions using NER models in conjunction with the ontology. This diagram explains how it works:

![EE-NER DFD](/images/EE_NER-DFD.png)

The following processes are launched:

- **Detect Entitie**s: Using Stanza biomedical NER models, entities are detetec from a small corpus of hand-made questions. 

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

### EE-Bert-T5
