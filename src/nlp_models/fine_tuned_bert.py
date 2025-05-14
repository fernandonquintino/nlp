import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class FineTunedBertClassifier:
    def __init__(self, model_name_or_path='bert-base-uncased', device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, texts, labels, epochs=3, batch_size=8, lr=2e-5):
        # Ensure texts are list of strings
        texts = list(texts)
        texts = [str(t) for t in texts]

        self.model.train()
        dataset = BertDataset(texts, labels, self.tokenizer)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, texts, batch_size=8):
        texts = list(texts)
        texts = [str(t) for t in texts]

        self.model.eval()
        dummy_labels = [0] * len(texts)
        dataset = BertDataset(texts, dummy_labels, self.tokenizer)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**batch)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        return preds

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path, device=None):
        return cls(model_name_or_path=path, device=device)

# --- Loader function ---
def load_fine_tuned_bert(model_path, device=None):
    return FineTunedBertClassifier.load(model_path, device=device)