# pip3 install seleniumbase
import os
from bs4 import BeautifulSoup
from seleniumbase import Driver
from test.test_pickle import getattribute


DATA_DIR = "hltv_data"
MAPS_DIR = os.path.join(DATA_DIR, "maps")
MATCH_DIR = os.path.join(DATA_DIR, "matches")

def get_all_links(html_content, selector):
    # Parse the saved HTML content (just the inner part of the stats table)
    soup = BeautifulSoup(html_content, "html.parser")

    hrefs = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if selector in a["href"]
    ]

    unique_links = [
        href if href.startswith("http") else "https://www.hltv.org" + href
        for href in hrefs
    ]

    return unique_links

        
def scrape_matches():

    url = "https://www.hltv.org/stats/matches?startDate=2025-01-15&endDate=2025-07-15&matchType=BigEvents&rankingFilter=Top20"
    offset = 0
    counter = 0
    
    while True:
        url = f"https://www.hltv.org/stats/matches?startDate=2025-04-15&endDate=2025-07-15&matchType=BigEvents&offset={offset}&rankingFilter=Top20"
        # Open URL using UC mode with 6 second reconnect to bypass detection
        driver.uc_open_with_reconnect(url, reconnect_time=6)

        # do it one time
        if counter == 0:
            # Attempt to click CAPTCHA checkbox if present
            driver.uc_gui_click_captcha()

            # Wait a bit for the page to fully load after CAPTCHA solving
            driver.sleep(2)
            
            try:
                driver.wait_for_element_visible('#CybotCookiebotDialogBodyButtonDecline', timeout=10)
                driver.click('#CybotCookiebotDialogBodyButtonDecline')
                print("Clicked Decline cookies button")
                # Wait a bit for the page to fully load after clicking the decline
                driver.sleep(2)

            except Exception:
                print("Decline button not found or already handled")

    

        html_content = driver.get_attribute(".stats-table.matches-table.no-sort", "innerHTML")
        soup = BeautifulSoup(html_content, "html.parser")
        rows = soup.select("tbody tr")

        if not rows:
            print("No more rows — stopping.")
            break


        save_path = os.path.join(MATCH_DIR, f"hltv_matches_{counter}.html")
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html_content)
        
        offset += 50
        counter += 1
    

def scrape_game(file_path):
    with open(file_path, "r") as f:
        html = f.read()
    
    links = get_all_links(html, "/matches/mapstatsid")
    
    for url in links:
        driver.uc_open_with_reconnect(url, reconnect_time=6)
        html_content = driver.get_attribute(".stats-section.stats-match", "innerHTML")
        file_name = url.split("/")[-1].split("?")[0] + "_" + url.split("/")[-2] + ".html"

        save_path = os.path.join(MAPS_DIR, file_name)
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html_content)

        

    
   
if __name__ == "__main__":
    # Initialize the driver in GUI mode with UC enabled
    driver = Driver(uc=True, headless=False)

    # scrape_matches()
    
    for f in os.listdir(MATCH_DIR):
        file_path = os.path.join(MATCH_DIR, f)
        print(file_path)
        scrape_game(file_path)

    
    driver.close()

