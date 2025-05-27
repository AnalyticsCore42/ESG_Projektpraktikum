import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

# Create directory for storing documentation
doc_dir = "fastai_docs/tabular"
os.makedirs(doc_dir, exist_ok=True)

# Base URL for fastai documentation
base_url = "https://docs.fast.ai"

# Specific tabular links to scrape
tabular_links = [
    ("Tabular core", "/tabular.core.html"),
    ("Tabular data", "/tabular.data.html"),
    ("Tabular learner", "/tabular.learner.html"),
    ("Tabular model", "/tabular.model.html"),
    ("Collaborative filtering", "/collab.html")
]

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

# Function to extract code examples and API details
def extract_api_details(soup):
    """Extract detailed API documentation including function signatures, parameters and examples"""
    api_sections = []
    
    # Look for API documentation sections
    for section in soup.select('.doc-class, .doc-function, .doc-method'):
        section_content = ""
        
        # Get the name and signature
        header = section.select_one('.doc-function-name, .doc-class-name, .doc-method-name')
        if header:
            section_content += f"## {header.text.strip()}\n\n"
        
        # Get the signature
        signature = section.select_one('.doc-function-signature, .doc-method-signature')
        if signature:
            section_content += f"```python\n{signature.text.strip()}\n```\n\n"
        
        # Get the documentation
        doc_text = section.select_one('.doc-function-doc, .doc-class-doc, .doc-method-doc')
        if doc_text:
            section_content += f"{doc_text.text.strip()}\n\n"
        
        # Get parameter descriptions
        params = section.select('.doc-param')
        if params:
            section_content += "### Parameters:\n\n"
            for param in params:
                param_name = param.select_one('.doc-param-name')
                param_desc = param.select_one('.doc-param-desc')
                if param_name and param_desc:
                    section_content += f"- **{param_name.text.strip()}**: {param_desc.text.strip()}\n"
            section_content += "\n"
        
        # Get code examples
        examples = section.select('pre')
        if examples:
            section_content += "### Examples:\n\n"
            for example in examples:
                section_content += f"```python\n{example.text.strip()}\n```\n\n"
        
        api_sections.append(section_content)
    
    return "\n".join(api_sections)

# Function to scrape and save a single documentation page
def scrape_page(title, url_path):
    url = urljoin(base_url, url_path)
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
                
                # Extract detailed API documentation
                api_details = extract_api_details(soup)
                if api_details:
                    content_text += "\n# API Details\n\n"
                    content_text += api_details
                
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

# Function to scrape fastai tabular documentation
def scrape_fastai_tabular_docs():
    print("Starting to scrape fastai tabular documentation...")
    
    # Save index page
    index_content = "# fastai Tabular Documentation Index\n\n"
    for title, url_path in tabular_links:
        full_url = urljoin(base_url, url_path)
        index_content += f"* [{title}]({full_url})\n"
    
    save_to_file(os.path.join(doc_dir, "index.txt"), index_content)
    
    # Scrape each tabular page
    success_count = 0
    for title, url_path in tabular_links:
        if scrape_page(title, url_path):
            success_count += 1
        
        # Be nice to the server - add delay between requests
        time.sleep(1)
    
    print(f"Completed scraping. Successfully scraped {success_count} of {len(tabular_links)} pages.")
    print(f"Documentation saved to '{doc_dir}' directory.")

# Function to create a feature importance documentation file
def create_feature_importance_guide():
    """Create a guide for feature importance in fastai tabular learners"""
    content = """# Feature Importance in fastai Tabular Learners

Feature importance is a crucial aspect of tabular models to understand which features have the most significant impact on predictions. Here are different approaches to calculate feature importance with fastai TabularLearner:

## Method 1: Using sklearn's permutation_importance

This method works by randomly shuffling each feature and measuring how much the performance metric drops:

```python
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

def calculate_permutation_importance(learn, valid_dl, n_iter=5):
    # Move model to CPU and set to eval mode
    learn.model.cpu()
    learn.model.eval()
    
    # Get validation dataset
    x_valid, y_valid = valid_dl.dataset.x.items, valid_dl.dataset.y.items
    
    # Convert to numpy for sklearn
    x_valid_np = {}
    for key in x_valid:
        if isinstance(x_valid[key], torch.Tensor):
            x_valid_np[key] = x_valid[key].numpy()
        else:
            x_valid_np[key] = x_valid[key]
    
    y_valid_np = y_valid.numpy() if isinstance(y_valid, torch.Tensor) else y_valid
    
    # Define a predict function for sklearn
    def model_predict(X):
        with torch.no_grad():
            preds = learn.model(X)
        return preds.numpy()
    
    # Calculate permutation importance
    result = permutation_importance(
        estimator=model_predict,
        X=x_valid_np,
        y=y_valid_np,
        n_repeats=n_iter,
        random_state=42
    )
    
    # Create a dictionary of feature importances
    feature_names = valid_dl.dataset.x_names
    importances = {name: imp for name, imp in zip(feature_names, result.importances_mean)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances
```

## Method 2: Using Random Forest for feature importance

For tabular data, you can train a Random Forest model alongside your TabularLearner and use its feature_importances_ attribute:

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def get_rf_feature_importance(df, cat_names, cont_names, target_name):
    # Prepare data for Random Forest
    X = pd.get_dummies(df, columns=cat_names)
    y = df[target_name]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_names = X.columns
    importances = {name: imp for name, imp in zip(feature_names, rf.feature_importances_)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances
```

## Method 3: Using SHAP values

SHAP (SHapley Additive exPlanations) provides a unified approach to explain model outputs:

```python
import shap

def get_shap_importance(learn, valid_dl, background_samples=100):
    # Set model to eval mode
    learn.model.eval()
    
    # Get a sample of data for the explainer
    x_sample, _ = next(iter(valid_dl))
    
    # Create a SHAP explainer
    explainer = shap.DeepExplainer(learn.model, x_sample[:background_samples])
    
    # Get SHAP values for validation data
    x_valid, _ = next(iter(valid_dl))
    shap_values = explainer.shap_values(x_valid)
    
    # Calculate feature importance from SHAP values
    feature_names = valid_dl.dataset.x_names
    importances = {name: np.abs(shap_values[:, i]).mean() 
                   for i, name in enumerate(feature_names)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances
```

## Visualizing Feature Importance

```python
def plot_feature_importance(importances, top_n=10):
    plt.figure(figsize=(10, 6))
    
    # Get top N features
    top_features = list(importances.keys())[:top_n]
    top_importances = [importances[feature] for feature in top_features]
    
    # Create horizontal bar chart
    plt.barh(range(len(top_features)), top_importances, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()
```

Remember that different methods may give different results, and it's often helpful to use multiple approaches to get a more complete understanding of your model's feature importance.
"""
    
    save_to_file(os.path.join(doc_dir, "feature_importance_guide.txt"), content)
    print("Created feature importance guide")

if __name__ == "__main__":
    scrape_fastai_tabular_docs()
    create_feature_importance_guide() 