from sklearn.linear_model import LogisticRegression
from datasets import load_dataset, DownloadMode
import numpy as np
from sklearn.metrics import classification_report
from provada.models.esm2 import ESM2Model
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

# Load the ESM2 model
esm2_model = ESM2Model(model_name="esm2_t33_650M_UR50D")


# Embed all the sequences
outputs = esm2_model(
    df["sequence"].tolist(),
    batch_size=80,
    keep_on_device=False,
    return_logits=False,
    device="cuda",
)
# Pull the mean embeddings
mean_embeddings = outputs["mean_embeddings"].numpy()
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
