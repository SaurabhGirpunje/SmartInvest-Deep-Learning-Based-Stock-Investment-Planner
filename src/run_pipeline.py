# src/run_pipeline.py
"""
Simple runner script to execute the finetune pipeline.

You ONLY need to edit the section:
    >>> SELECT WHICH MODE TO RUN <<<

No command-line arguments are used.
Everything is controlled inside this script.
"""

import glob
import logging
from pathlib import Path
from finetune_pipeline import process_stock, write_aggregates

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

DATA_ROOT = Path("data") / "stock_data"
OUTPUT_ROOT = Path("output")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def find_stock_csv(sector, stock):
    path = DATA_ROOT / sector / f"{stock}.csv"
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"CSV not found: {path}")


def collect_sector_csvs(sector):
    pattern = str(DATA_ROOT / sector / "*.csv")
    return sorted(glob.glob(pattern))


def collect_all_csvs():
    pattern = str(DATA_ROOT / "*" / "*.csv")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------
# MAIN RUN FUNCTION — EDIT ONLY THIS PART
# ---------------------------------------------------------
def main():
    """
    SELECT EXACTLY ONE MODE BELOW BY UNCOMMENTING.
    """

    # ======================================================
    # MODE 1: Run ONE stock
    # ======================================================
    sector = "Auto"
    stock = "EICHERMOT"
    csvs = [find_stock_csv(sector, stock)]

    # Auto: BAJAJ-AUTO, EICHERMOT, M&M, MARUTI, TATAMOTORS
    # Bank: AXISBANK, HDFCBANK, ICICIBANK, KOTAKBANK, SBIN
    # Chemicals: PIDILITIND, PIIND, SOLARINDS, SRF, UPL
    # FMCG: BRITANNIA, HINDUNILVR, ITC, NESTLEIND, TATACONSUM
    # Healthcare: APOLLOHOSP, CIPLA, DIVISLAB, MAXHEALTH, SUNPHARMA
    # IT: HCLTECH, INFY, TCS, TECHM, WIPRO
    # Media: NAZARA, PVRINOX, SAREGAMA, SUNTV, ZEEL
    # Metal: ADANIENT, HINDALCO, JSWSTEEL, TATASTEEL, VEDL
    # OilGas: BPCL, GAIL, IOC, ONGC, RELIANCE
    # Pharma: CIPLA, DIVISLAB, DRREDDY, LUPIN, SUNPHARMA

    # ======================================================
    # MODE 2: Run ALL stocks inside one sector
    # ======================================================
    # sector = "Auto"
    # csvs = collect_sector_csvs(sector)


    # ======================================================
    # MODE 3: Run ALL stocks in dataset
    # ======================================================
    # csvs = collect_all_csvs()


    # ------------------------------------------------------
    # Run selected CSVs
    # ------------------------------------------------------
    logging.info(f"Running pipeline for {len(csvs)} file(s).")

    records = []
    for csv in csvs:
        logging.info(f"Processing: {csv}")
        try:
            rec = process_stock(
                csv_path=csv,
                output_root=str(OUTPUT_ROOT),
                window=10,
                step=5,
                epochs=30,
                batch_size=16,
                forecast_days=5,
                verbose=0
            )
            records.append(rec)
        except Exception as e:
            logging.exception(f"Error processing {csv}: {e}")

    # Save aggregated metrics
    if records:
        write_aggregates(records, output_root=str(OUTPUT_ROOT))
        logging.info("Aggregation complete.")
    else:
        logging.warning("No successful records to aggregate.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
