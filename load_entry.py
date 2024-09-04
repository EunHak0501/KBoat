import subprocess

def crawl_data_for_year(year):
    subprocess.run(["python", "crawlling_entry.py", str(year)])
    subprocess.run(["python", "crawlling_result.py", str(year)])
    print(f"Crawling completed for {year}.")

crawl_data_for_year(2024)