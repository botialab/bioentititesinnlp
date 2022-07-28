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




### EE-Specie

### EE-Bert-T5
