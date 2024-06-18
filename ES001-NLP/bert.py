# Import the necessary libraries
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Preprocess the data
df = pd.read_csv("data.csv", header=None, names=["text", "label"])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_data = tokenizer.batch_encode_plus(df.text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=256, return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(df.label.values)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)

# Define the training parameters
batch_size = 32
epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(epochs):
    model.train()
    for i in range(0, input_ids.size(0), batch_size):
        optimizer.zero_grad()
        outputs = model(input_ids[i:i+batch_size], attention_mask=attention_masks[i:i+batch_size], labels=labels[i:i+batch_size])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    predictions = torch.argmax(outputs[0], dim=1).flatten()
    accuracy = torch.sum(predictions == labels) / len(labels)

print("Accuracy:", accuracy.item())