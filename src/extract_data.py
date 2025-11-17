# extract_data.py
# Run this script to download sector PDFs, extract top stocks, create CSVs, and fetch stock data.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import DATA_DIR
from scripts.fetch_stockdata import fetch_stock_data
from scripts.stock_name import download_sector_pdfs, extract_top_stocks, create_sector_csv


def main():
    download_sector_pdfs()
    print("Downloaded sector PDFs.")

    top_stocks = extract_top_stocks(num_top=5)
    print("Extracted top stocks from PDFs.")

    create_sector_csv(top_stocks)
    print("Created sector CSV files.")

    ticker_path = os.path.join(DATA_DIR, "top_stocks_code.xlsx")
    print(f"Ticker mapping file created at {ticker_path}.")

    fetch_stock_data(top_stocks, ticker_path)
    print("Fetched stock data for top companies.")


if __name__ == "__main__":
    main()
    print("Data extraction and saving completed.")