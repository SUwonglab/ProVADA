from sklearn.linear_model import LogisticRegression
from datasets import load_dataset, DownloadMode
import numpy as np
from sklearn.metrics import classification_report
from proto_tools.tools import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    run_esm2_embeddings,
)
import pickle
from provada.utils.setup import seed_everything
import os
import pandas as pd


# Set the seed
seed_everything(42)

# Load the dataset
dataset = load_dataset(
    path="Xiaowei0402/uniprot_subcellular_localization",
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
)


dfs = []
for split in dataset.keys():
    df = dataset[split].to_pandas()
    df["split"] = split
    dfs.append(df)

# Concatenate the dataframes
df = pd.concat(dfs)

# Embed all the sequences. ESM2 runs in proto-tools' own environment; the
# checkpoint must match LocalizationPredictor.MODEL_CHECKPOINT, since the
# classifier fitted below is only valid for that embedding space.
outputs = run_esm2_embeddings(
    ESM2EmbeddingsInput(sequences=df["sequence"].tolist()),
    ESM2EmbeddingsConfig(
        model_checkpoint="esm2_t33_650M_UR50D",
        batch_size=80,
        return_logits=False,
        device="cuda",
    ),
)
mean_embeddings = np.array([result.mean_embedding for result in outputs.results])
labels = np.array(df["cytosolic_label"].tolist())

# Create index splits
train_indices = df[df["split"] == "train"].index.tolist()
val_indices = df[df["split"] == "validation"].index.tolist()

# Split the mean embeddings
train_embeddings = mean_embeddings[train_indices]
val_embeddings = mean_embeddings[val_indices]

# Split the labels
train_labels = labels[train_indices]
val_labels = labels[val_indices]

# Create a logistic regression model with L2 regularization
model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")

# Fit the model
model.fit(train_embeddings, train_labels)

# Make predictions
val_predictions = model.predict(val_embeddings)

# Evaluate the model
print(classification_report(val_labels, val_predictions))

# Get the directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# save model as pickle
with open(os.path.join(current_dir, "logreg_model.pkl"), "wb") as file:
    pickle.dump(model, file)
