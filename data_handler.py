import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import base64
import os

class CyberDataHandler:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def fetch_crypto_data(self, symbol, period='1y'):
        """Fetch cryptocurrency data from Yahoo Finance"""
        try:
            crypto = yf.Ticker(symbol)
            data = crypto.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def encrypt_data(self, data, filename):
        """Encrypt data and save to file"""
        try:
            # Convert DataFrame to bytes
            data_bytes = data.to_csv().encode()
            # Encrypt the data
            encrypted_data = self.cipher_suite.encrypt(data_bytes)
            # Save to file
            with open(filename, 'wb') as f:
                f.write(encrypted_data)
            return True
        except Exception as e:
            print(f"Error encrypting data: {e}")
            return False

    def decrypt_data(self, filename):
        """Decrypt data from file"""
        try:
            with open(filename, 'rb') as f:
                encrypted_data = f.read()
            # Decrypt the data
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            # Convert back to DataFrame
            data = pd.read_csv(pd.io.common.StringIO(decrypted_data.decode()))
            return data
        except Exception as e:
            print(f"Error decrypting data: {e}")
            return None

    def preprocess_data(self, data):
        """Preprocess the data for neural network"""
        # Normalize the data
        data['Normalized'] = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())
        return data['Normalized'].values

    def generate_future_dates(self, last_date, days):
        """Generate future dates for predictions"""
        return [last_date + timedelta(days=x) for x in range(1, days+1)]

    def save_key(self, filename='encryption_key.key'):
        """Save the encryption key to a file"""
        try:
            with open(filename, 'wb') as f:
                f.write(self.key)
            return True
        except Exception as e:
            print(f"Error saving key: {e}")
            return False

    def load_key(self, filename='encryption_key.key'):
        """Load the encryption key from a file"""
        try:
            with open(filename, 'rb') as f:
                self.key = f.read()
            self.cipher_suite = Fernet(self.key)
            return True
        except Exception as e:
            print(f"Error loading key: {e}")
            return False 