import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

# Dataset definition
class OCRDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pair = self.dataset[idx]
        inputs = self.tokenizer(pair[0], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        outputs = self.tokenizer(pair[1], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "labels": outputs["input_ids"].squeeze()}

# Training function
def train(model, train_dataloader, optimizer, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train().to(device)

    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Inference function
def correct_ocr(model, input_text, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":

    # Hyperparameters
    batch_size = 4
    max_length = 32
    learning_rate = 1e-4
    epochs = 10

    with open("ocr_dataset.json", "r") as f:
        dataset_json = json.load(f)

    dataset = [(pair["mistake"], pair["correction"]) for pair in dataset_json]

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Create dataloader and optimizer
    ocr_dataset = OCRDataset(dataset, tokenizer, max_length=max_length)
    train_dataloader = DataLoader(ocr_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_dataloader, optimizer, epochs=epochs)