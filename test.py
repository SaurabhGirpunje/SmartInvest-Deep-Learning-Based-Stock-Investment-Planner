import os, requests
from scripts.config import DATA_DIR
from scripts.fetch_stockdata import fetch_stock_data, clean_and_scale_stock_data
from scripts.stock_name import download_sector_pdfs, extract_top_stocks, create_sector_csv


def main():
    download_sector_pdfs()
    top_stocks = extract_top_stocks(num_top=10)
    create_sector_csv(top_stocks)
    ticker_path = os.path.join(DATA_DIR, "top_stocks_code.xlsx")
    fetch_stock_data(top_stocks, ticker_path)
    clean_and_scale_stock_data()


if __name__ == "__main__":
    main()