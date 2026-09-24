import requests
import os
import json
import time

headers = {"User-Agent": "yourname@youremail.com"}  # replace with your actual email

os.makedirs("./sec_filings", exist_ok=True)

companies = {
    "AAPL": "0000320193",
    "MSFT": "0000789019", 
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "NVDA": "0001045810",
    "TSLA": "0001318605"
}

def get_latest_10k(cik, ticker):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    filings = data["filings"]["recent"]
    forms = filings["form"]
    urls = filings["primaryDocument"]
    accession = filings["accessionNumber"]
    
    for i, form in enumerate(forms):
        if form == "10-K":
            acc = accession[i].replace("-", "")
            doc = urls[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            
            r = requests.get(filing_url, headers=headers)
            filename = f"./sec_filings/{ticker}_10K.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"Downloaded {ticker}")
            time.sleep(1)  # SEC rate limit — don't remove this
            return
        
for ticker, cik in companies.items():
    get_latest_10k(cik, ticker)

print(f"Done. Files in ./sec_filings/")