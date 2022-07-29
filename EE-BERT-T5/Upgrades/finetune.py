from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime

import pandas as pd 
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

trainSp = pd.read_csv('QuestionsSpecie/qSp-Train.csv')
valSp = pd.read_csv('QuestionsSpecie/qSp-Val.csv')
testSp = pd.read_csv('QuestionsSpecie/qSp-Test.csv')

trainPh = pd.read_csv('QuestionsPhenotype/qPh-Train.csv')
valPh = pd.read_csv('QuestionsPhenotype/qPh-Val.csv')
testPh = pd.read_csv('QuestionsPhenotype/qPh-Test.csv')

trainTs = pd.read_csv('QuestionsTissue/qTs-Train.csv')
valTs = pd.read_csv('QuestionsTissue/qTs-Val.csv')
testTs = pd.read_csv('QuestionsTissue/qTs-Test.csv')

dataset = dict()

train = pd.concat([trainSp,trainPh,trainTs])
val = pd.concat([valSp,valPh,valTs])
test = pd.concat([testSp,testPh,testTs])

#dataset['Specie'] = [trainSp,valSp,testSp]
#dataset['Phenotype'] = [trainPh,valPh,testPh]
#dataset['Tissue'] = [trainTs,valTs,testTs]

dataset['suffle'] = [train,val,test]

# Read the dataset
model_name = 'pritamdeka/S-BioBert-snli-multinli-stsb'
#model_name = 'models/tsdae-modelSpecieSmall-v2'
train_batch_size = 32
num_epochs = 5
model_save_path = 'BioBert-FineTune-Base-multitask-suffle'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read Biomed Questions train dataset")

for ds in dataset.keys():

    train = dataset[ds][0]
    val = dataset[ds][1]
    test = dataset[ds][2]

    print('Fine Tune for %s' % ds)

    train_samples = []
    dev_samples = []
    test_samples = []

    for index,row in train.iterrows():
        inp_example = InputExample(texts=[row['Question'], row['Label']], label=float(row['score']))
        train_samples.append(inp_example)

    for index,row in val.iterrows():
        inp_example = InputExample(texts=[row['Question'], row['Label']], label=float(row['score']))
        dev_samples.append(inp_example)

    for index,row in test.iterrows():
        inp_example = InputExample(texts=[row['Question'], row['Label']], label=float(row['score']))
        test_samples.append(inp_example)



    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    # Development set: Measure correlation between cosine score and gold labels
    logging.info("Read Evaluation Biomed Questions dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='biomed-questions-test')
    test_evaluator(model, output_path=model_save_path)