# Receipt Data Extraction using Donut Model

This repository contains a Python script that uses the Donut model to extract relevant information from receipt images.

## Requirements

* Python 3.8+
* torch
* numpy
* transformers


## Installation

1. Clone the repository: `git clone https://github.com/rajavignesh/receipt-data-extraction.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the script: `python extraction.py`

## Output

The script will output a JSON file for each receipt image, containing the extracted information and it will generate the report of filename,
processed time and accuracy

Example output:
{'company_name': 'MR D.T.Y. (M) SON BHD', 'date': '11/11/2024', 'address': "101 1251-A & 1851-B. JALUN KPB KILAT AUTO ECO WASH & SHINE ES1000 KILAT' ECO AUTO WASH &MAX WA44-A - 12 WA43 A - 24 KLEENGO AJAIB 99 SERAI WANYI goOG HANDKERCHIEF 71386#2PCS", 'total_amount': 30.9}
