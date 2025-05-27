import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

# Create directory for storing documentation
doc_dir = "fastai_docs"
os.makedirs(doc_dir, exist_ok=True)

# Base URL for fastai documentation
base_url = "https://docs.fast.ai"

# Function to clean text
def clean_text(text):
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to save content to file
def save_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved: {filename}")

# Function to extract all links from the navigation sidebar
def extract_nav_links(soup):
    links = []
    nav_elements = soup.select('.md-nav__item a.md-nav__link')
    
    for link in nav_elements:
        href = link.get('href')
        if href and not href.startswith('#') and not href.startswith('http'):
            full_url = urljoin(base_url, href)
            title = clean_text(link.text)
            links.append((title, full_url))
    
    return links

# Function to scrape and save a single documentation page
def scrape_page(title, url):
    print(f"Scraping: {title} - {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content area
            content_div = soup.select_one('.md-content__inner')
            if content_div:
                # Extract headings, paragraphs, code blocks, etc.
                content_text = ""
                
                # Add title
                content_text += f"# {title}\n\n"
                content_text += f"URL: {url}\n\n"
                
                # Process content elements
                for element in content_div.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol']):
                    if element.name.startswith('h'):
                        level = int(element.name[1])
                        content_text += f"{'#' * level} {clean_text(element.text)}\n\n"
                    elif element.name == 'p':
                        content_text += f"{clean_text(element.text)}\n\n"
                    elif element.name == 'pre':
                        code = element.text.strip()
                        content_text += f"```\n{code}\n```\n\n"
                    elif element.name in ['ul', 'ol']:
                        for li in element.find_all('li'):
                            content_text += f"* {clean_text(li.text)}\n"
                        content_text += "\n"
                
                # Clean filename to be safe for filesystem
                safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
                filename = os.path.join(doc_dir, f"{safe_title}.txt")
                save_to_file(filename, content_text)
                
                return True
            else:
                print(f"Could not find content for {title}")
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
    
    return False

# Main function to scrape all documentation
def scrape_fastai_docs():
    print("Starting to scrape fastai documentation...")
    
    # Get the main page
    main_page = requests.get(base_url)
    if main_page.status_code != 200:
        print(f"Failed to fetch main page: {base_url}")
        return
    
    # Parse the main page
    soup = BeautifulSoup(main_page.text, 'html.parser')
    
    # Extract all links from the navigation
    links = extract_nav_links(soup)
    
    # Save index page
    index_content = "# fastai Documentation Index\n\n"
    for title, url in links:
        index_content += f"* [{title}]({url})\n"
    
    save_to_file(os.path.join(doc_dir, "index.txt"), index_content)
    
    # Scrape each linked page
    success_count = 0
    for title, url in links:
        if scrape_page(title, url):
            success_count += 1
        
        # Be nice to the server - add delay between requests
        time.sleep(1)
    
    print(f"Completed scraping. Successfully scraped {success_count} of {len(links)} pages.")
    print(f"Documentation saved to '{doc_dir}' directory.")

if __name__ == "__main__":
    scrape_fastai_docs() 