import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

class CyberLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                          torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_cyber_plot(predictions, actual_values, dates):
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_values,
        name='Actual',
        line=dict(color='#00ff00', width=2),
        mode='lines'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=dates[-len(predictions):],
        y=predictions,
        name='Predicted',
        line=dict(color='#00f3ff', width=2, dash='dash'),
        mode='lines'
    ))
    
    # Update layout with cyberpunk theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Courier New', color='#00f3ff'),
        title=dict(
            text='Neural Network Predictions',
            font=dict(size=24, color='#00f3ff'),
            x=0.5
        ),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(0,255,0,0.1)',
            linecolor='#00ff00'
        ),
        yaxis=dict(
            title='Price',
            gridcolor='rgba(0,255,0,0.1)',
            linecolor='#00ff00'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#00f3ff',
            borderwidth=1
        )
    )
    
    return fig

def train_model(model, train_data, train_labels, epochs=100):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                           torch.zeros(1, 1, model.hidden_layer_size))
        
        y_pred = model(train_data)
        single_loss = loss_function(y_pred, train_labels)
        single_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 10 == 0:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')
    
    return model

def prepare_data(data, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return torch.FloatTensor(X), torch.FloatTensor(y) 