from bs4 import BeautifulSoup
import os

input_dir = "./sec_filings"
output_dir = "./sec_filings_clean"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        with open(f"{input_dir}/{filename}", "r", encoding="utf-8") as f:
            raw = f.read()
        
        soup = BeautifulSoup(raw, "html.parser")
        clean_text = soup.get_text(separator="\n")
        
        # Remove excessive blank lines
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
        output_path = f"{output_dir}/{filename}"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        
        print(f"Cleaned {filename} — {len(clean_text)} chars")