"""
Generate sample bank transaction CSV files for testing prompt tuning system.
Creates 30 CSV files with realistic bank transaction data including anomalies.
"""

import csv
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np


class BankDataGenerator:
    """Generate realistic bank transaction data with anomalies."""

    def __init__(self, num_files: int = 30, transactions_per_file: int = 100):
        self.num_files = num_files
        self.transactions_per_file = transactions_per_file
        self.output_dir = "bank_data"

        # Transaction categories
        self.normal_categories = [
            "Groceries", "Utilities", "Transport", "Entertainment",
            "Restaurants", "Shopping", "Healthcare", "Insurance"
        ]
        self.anomaly_categories = [
            "Suspicious Wire Transfer", "Duplicate Payment", "Unusual Merchant",
            "Late Night ATM", "Foreign Transaction", "High-Risk Vendor"
        ]

    def generate_normal_transaction(self, transaction_id: int, date: datetime) -> Dict:
        """Generate a normal transaction."""
        amount = round(random.uniform(5, 200), 2)
        return {
            "transaction_id": f"TXN{transaction_id:06d}",
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_gbp": amount,
            "category": random.choice(self.normal_categories),
            "merchant": f"Merchant_{random.randint(1, 100)}",
            "description": "Regular transaction",
            "is_anomaly": False,
            "above_250": False
        }

    def generate_high_value_transaction(self, transaction_id: int, date: datetime) -> Dict:
        """Generate a transaction above 250 GBP."""
        amount = round(random.uniform(251, 5000), 2)
        is_anomaly = random.random() < 0.3  # 30% of high-value transactions are anomalies

        return {
            "transaction_id": f"TXN{transaction_id:06d}",
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_gbp": amount,
            "category": random.choice(self.anomaly_categories if is_anomaly else self.normal_categories),
            "merchant": f"Merchant_{random.randint(1, 100)}",
            "description": "High-value transaction" + (" - ANOMALY" if is_anomaly else ""),
            "is_anomaly": is_anomaly,
            "above_250": True
        }

    def generate_anomaly_transaction(self, transaction_id: int, date: datetime) -> Dict:
        """Generate an anomalous transaction."""
        # Anomalies can be various amounts
        amount = round(random.choice([
            random.uniform(5, 100),      # Small anomaly
            random.uniform(251, 10000)   # Large anomaly
        ]), 2)

        anomaly_types = [
            "Multiple transactions in short time",
            "Unusual location",
            "Suspicious merchant category",
            "Round number pattern",
            "Velocity anomaly"
        ]

        return {
            "transaction_id": f"TXN{transaction_id:06d}",
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "amount_gbp": amount,
            "category": random.choice(self.anomaly_categories),
            "merchant": f"SuspiciousMerchant_{random.randint(1, 20)}",
            "description": random.choice(anomaly_types),
            "is_anomaly": True,
            "above_250": amount > 250
        }

    def generate_file(self, file_num: int) -> str:
        """Generate a single CSV file with transactions."""
        transactions = []
        start_date = datetime.now() - timedelta(days=365)

        for i in range(self.transactions_per_file):
            transaction_id = file_num * 1000 + i
            transaction_date = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Distribution: 70% normal, 20% high-value, 10% anomaly
            rand = random.random()
            if rand < 0.70:
                transaction = self.generate_normal_transaction(transaction_id, transaction_date)
            elif rand < 0.90:
                transaction = self.generate_high_value_transaction(transaction_id, transaction_date)
            else:
                transaction = self.generate_anomaly_transaction(transaction_id, transaction_date)

            transactions.append(transaction)

        # Sort by date
        transactions.sort(key=lambda x: x['date'])

        # Write to CSV
        filename = os.path.join(self.output_dir, f"bank_account_{file_num:02d}.csv")
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['transaction_id', 'date', 'amount_gbp', 'category',
                         'merchant', 'description', 'is_anomaly', 'above_250']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transactions)

        return filename

    def generate_all_files(self) -> List[str]:
        """Generate all CSV files."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Generating {self.num_files} CSV files...")
        files = []
        for i in range(self.num_files):
            filename = self.generate_file(i)
            files.append(filename)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.num_files} files")

        print(f"\nAll {self.num_files} files generated successfully in '{self.output_dir}' directory")
        return files

    def get_ground_truth_stats(self) -> Dict:
        """Calculate statistics from generated data."""
        import pandas as pd

        all_data = []
        for i in range(self.num_files):
            filename = os.path.join(self.output_dir, f"bank_account_{i:02d}.csv")
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        stats = {
            "total_transactions": len(combined_df),
            "transactions_above_250": combined_df['above_250'].sum(),
            "total_anomalies": combined_df['is_anomaly'].sum(),
            "anomalies_above_250": combined_df[combined_df['above_250'] == True]['is_anomaly'].sum(),
            "anomalies_below_250": combined_df[combined_df['above_250'] == False]['is_anomaly'].sum()
        }

        return stats


if __name__ == "__main__":
    generator = BankDataGenerator(num_files=30, transactions_per_file=100)
    files = generator.generate_all_files()
    stats = generator.get_ground_truth_stats()

    print("\n" + "="*50)
    print("Ground Truth Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
