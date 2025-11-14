import os
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from scripts.config import STOCK_DATA_DIR, START_DATE, END_DATE, CLEAN_DATA_DIR


def fetch_stock_data(top_stock: dict, mapping_excel: str):
    # Load mapping file
    mapping = pd.read_excel(mapping_excel)
    mapping.columns = [c.strip() for c in mapping.columns]

    required_cols = ["Sector", "Company Name", "NSE Code", "Weight (%)"]
    for col in required_cols:
        if col not in mapping.columns:
            raise Exception(f"Column '{col}' missing in {mapping_excel}")

    # Remove NOT_FOUND rows
    mapping = mapping[mapping["NSE Code"] != "NOT_FOUND"].copy()

    # Iterate through each sector in top_stock
    for sector, companies in top_stock.items():
        print(f"\nSector: {sector}")

        # Create sector folder if doesn't exist
        sector_dir = os.path.join(STOCK_DATA_DIR, sector)
        os.makedirs(sector_dir, exist_ok=True)

        # Convert list of tuples only company names
        company_names = [name for name, _ in companies]

        # Filter mapping  only required companies under this sector
        df_sector = mapping[
            (mapping["Sector"] == sector) &
            (mapping["Company Name"].isin(company_names))
        ]

        if df_sector.empty:
            print(f"No matching companies found in Excel for sector: {sector}")
            continue

        for _, row in df_sector.iterrows():
            company = row["Company Name"]
            nse_code = row["NSE Code"]
            ticker = f"{nse_code}.NS"
            out_path = os.path.join(sector_dir, f"{nse_code}.csv")

            print(f"Updating {company} ({ticker}) ...")

           
            #  Download latest data
            try:
                new_data = yf.download(
                    ticker,
                    start=START_DATE,
                    end=END_DATE,
                    progress=False,
                    auto_adjust=True  # avoid FutureWarning
                )

                if new_data.empty:
                    print(f" No data returned for {ticker}")
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = ["_".join([str(i) for i in col]).strip() for col in new_data.columns.values]

                # Keep only Close column
                close_col = next((col for col in new_data.columns if 'Close' in col), None)
                if not close_col:
                    print(f" 'Close' column not found for {ticker}")
                    continue

                new_df = new_data[[close_col]].reset_index()
                new_df.rename(columns={'Date': 'date', close_col: 'close'}, inplace=True)

            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue

            
            #  If file exists, load & merge
            if os.path.exists(out_path):
                # Read old CSV with date parsing
                old_df = pd.read_csv(out_path, usecols=['date', 'close'], parse_dates=['date'])

                # Combine old and new
                combined = pd.concat([old_df, new_df], ignore_index=True)

                # Remove duplicates based on date
                combined.drop_duplicates(subset=['date'], keep='last', inplace=True)

                # Ensure date column is datetime
                combined['date'] = pd.to_datetime(combined['date'], errors='coerce')

                # Sort by date
                combined.sort_values(by='date', inplace=True)

                # Save updated file
                combined.to_csv(out_path, index=False)
                print(f" Updated existing file: {out_path} (total {len(combined)} rows)")

            else:
                #  No existing file, create new
                new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
                new_df.sort_values(by='date', inplace=True)
                new_df.to_csv(out_path, index=False)
                print(f" Created new file: {out_path}")

    print("\n  Dataset updated!")






def clean_and_scale_stock_data():
    scaler = StandardScaler()

    for sector in os.listdir(STOCK_DATA_DIR):
        src = os.path.join(STOCK_DATA_DIR, sector)
        if not os.path.isdir(src):
            continue

        dest = os.path.join(CLEAN_DATA_DIR, sector)
        os.makedirs(dest, exist_ok=True)
        print(f"\nCleaning {sector}")

        for file in os.listdir(src):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(src, file)
            df = pd.read_csv(file_path, parse_dates=["date"])  # parse date
            df.set_index("date", inplace=True)

            # Reindex to all business days
            df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="B"))

            # Forward-fill missing prices
            df["close"] = df["close"].ffill()

            # Reset index and rename
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)

            # ---------------------------------------
            # Standard Scaling
            # ---------------------------------------
            df['close_scaled'] = scaler.fit_transform(df[['close']])

            # Save cleaned and scaled CSV
            output_file = os.path.join(dest, file.replace(".csv", "_scaled.csv"))
            df.to_csv(output_file, index=False)
            print(f"Cleaned and scaled: {file}")


