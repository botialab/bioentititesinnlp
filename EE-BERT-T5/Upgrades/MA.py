import pandas as pd

df = pd.read_csv("AbstractsEspecieMA.csv")

dataset = pd.DataFrame()
#dataset['id'] = df.index
dataset['text'] = df['Title']

print(dataset.head())

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader

# Define your sentence transformer model using CLS pooling
model_name = 'pritamdeka/S-BioBert-snli-multinli-stsb'

word_embedding_model = models.Transformer(model_name)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:0")

model.to('cuda')

# Define a list with sentences (1k - 100k sentences)
train_sentences = [str(item).replace("\n",'') for item in dataset.text.tolist()]

# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)

model.save('tsdae-modelSpecieSmall-v2')