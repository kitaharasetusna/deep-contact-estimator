import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, 64, num_layers, batch_first=True)
        self.fc = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, 16)
        # self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x):
        #(num_layers, B, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), 64).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), 64).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out  = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# Define model parameters
input_size = 54  # Number of dimensions
hidden_size = 64
num_layers = 2
output_size = 16

# Create the model
model = LSTMModel(input_size, num_layers, output_size)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Sample input and output shapes
batch_size = 30
seq_length = 150

# Randomly generate sample data
input_data = torch.randn(batch_size, seq_length, input_size)
labels = torch.randint(0, output_size, (batch_size,))
print(labels.shape)

# Forward pass
outputs = model(input_data)

# Calculate the loss
loss = criterion(outputs, labels)

print(f"Output shape: {outputs.shape}")
print(f"Loss: {loss.item()}")
