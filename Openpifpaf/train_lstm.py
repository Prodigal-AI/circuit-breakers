import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Constants
KEYPOINTS_DIR = 'keypoints_data'
MODEL_DIR = 'models'
EXERCISES = ['leg-raises', 'plank', 'pull-up', 'push-up', 'squat']
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 30  # Use last 30 frames per sample
INPUT_SIZE = 17 * 2   # 17 keypoints, x and y

# Dataset class
class ExerciseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # [samples x frames x 34]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

# LSTM model
class ExerciseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ExerciseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # only last hidden state
        out = self.fc(hn[-1])
        return out

# Training function
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for exercise in EXERCISES:
    print(f"Training model for {exercise}...")

    folder = os.path.join(KEYPOINTS_DIR, exercise)
    all_data = []
    all_labels = []

    for file in os.listdir(folder):
        if file.endswith('.npy'):
            data = np.load(os.path.join(folder, file))  # [frames x 17 x 2]
            if len(data) >= SEQUENCE_LENGTH:
                # use last SEQUENCE_LENGTH frames
                trimmed = data[-SEQUENCE_LENGTH:].reshape(SEQUENCE_LENGTH, -1)  # [30 x 34]
                all_data.append(trimmed)
                all_labels.append(1)  # 1 = correct form (for now)

    if not all_data:
        print(f"No valid data found for {exercise}, skipping.")
        continue

    # Train/test split
    x_train, x_val, y_train, y_val = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

    train_dataset = ExerciseDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Binary classification (correct = 1, incorrect = 0)
    model = ExerciseLSTM(INPUT_SIZE, HIDDEN_SIZE, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer)

    # Save model
    model_path = os.path.join(MODEL_DIR, f'{exercise}_lstm.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}\n")
