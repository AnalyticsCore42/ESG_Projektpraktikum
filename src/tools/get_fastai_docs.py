"""
Script to download fastai documentation using direct HTML parsing.
"""
import requests
import os
import re
from bs4 import BeautifulSoup

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

def save_to_file(filename, content):
    """Save content to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved: {filename}")

def scrape_page(title, url_path):
    """Scrape a page and save its raw HTML content."""
    url = f"{base_url}{url_path}"
    print(f"Scraping: {title} - {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Save the raw HTML
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            filename = os.path.join(doc_dir, f"{safe_title}.html")
            save_to_file(filename, response.text)
            
            # Also save a simplified text version
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Save text version
            txt_filename = os.path.join(doc_dir, f"{safe_title}.txt")
            save_to_file(txt_filename, f"# {title}\n\nURL: {url}\n\n{text}")
            
            return True
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
    
    return False

def create_feature_importance_guide():
    """Create a guide for feature importance in fastai tabular learners."""
    content = """# Feature Importance in fastai TabularLearner

Feature importance is a crucial aspect of tabular models to understand which features have the most significant impact on predictions. Here are different approaches to calculate feature importance with fastai TabularLearner:

## Method 1: Using sklearn's RandomForestRegressor

Since TabularLearner doesn't have a built-in feature_importance attribute, a common approach is to train a RandomForest on the same data:

```python
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def get_rf_feature_importance(df, cat_names, cont_names, target_name):
    # Prepare data for Random Forest
    X = pd.get_dummies(df, columns=cat_names)
    X = X.drop(columns=[target_name])
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

## Method 2: Using SHAP values

SHAP (SHapley Additive exPlanations) provides a unified approach to explain model outputs:

```python
import shap
import numpy as np

def get_shap_importance(learn, df, cat_names, cont_names, target_name):
    # Prepare data
    X = pd.get_dummies(df, columns=cat_names)
    X = X.drop(columns=[target_name])
    
    # Train a surrogate model (Random Forest works well)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, df[target_name])
    
    # Sample rows for background data
    background_data = X.sample(min(100, len(X)))
    
    # Create explainer
    explainer = shap.Explainer(rf, background_data)
    
    # Calculate SHAP values
    shap_values = explainer(X.sample(min(200, len(X))))
    
    # Get global feature importance
    importances = {name: np.abs(shap_values.values[:, i]).mean() 
                  for i, name in enumerate(X.columns)}
    
    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances
```

## TabularModel Structure in fastai

fastai's TabularModel is implemented as a PyTorch nn.Module with this basic structure:

1. An embedding layer for categorical variables
2. A batch normalization layer for continuous variables 
3. Fully connected layers with dropout and ReLU activations
4. An output layer

Unlike some ML models, it doesn't have a built-in feature_importance attribute, which is why we use alternative methods.

## Common Errors

### 1. "TabularModel is not subscriptable"

This error occurs when trying to access model components with indexing:

```python
# Incorrect:
weights = learn.model[0]  # Error: 'TabularModel' is not subscriptable

# Correct:
# Access components by attribute name
embeds = learn.model.embeds
layers = learn.model.layers
```

### 2. CUDA tensor conversion errors

When working with GPU tensors, always move them to CPU before conversion:

```python
# Move model to CPU
learn.model.cpu()

# For tensors
tensor_cpu = cuda_tensor.cpu()
numpy_array = tensor_cpu.numpy()
```

### 3. Feature importance in TabularLearner

Since TabularModel doesn't have a built-in feature_importance attribute, use sklearn or SHAP as shown above.
"""
    
    guide_filename = os.path.join(doc_dir, "feature_importance_guide.txt")
    save_to_file(guide_filename, content)
    print("Created feature importance guide")

def scrape_fastai_docs():
    """Main function to scrape fastai documentation."""
    print("Starting to scrape fastai tabular documentation...")
    
    # Save index page
    index_content = "# fastai Tabular Documentation Index\n\n"
    for title, url_path in tabular_links:
        full_url = f"{base_url}{url_path}"
        index_content += f"* [{title}]({full_url})\n"
    
    save_to_file(os.path.join(doc_dir, "index.txt"), index_content)
    
    # Scrape each page
    success_count = 0
    for title, url_path in tabular_links:
        if scrape_page(title, url_path):
            success_count += 1
    
    print(f"Completed scraping. Successfully scraped {success_count} of {len(tabular_links)} pages.")
    print(f"Documentation saved to '{doc_dir}' directory.")

def download_official_docs():
    """Download documentation from GitHub fastai repo."""
    github_docs = [
        ("tabular.core", "https://raw.githubusercontent.com/fastai/fastai/master/nbs/43_tabular.core.ipynb"),
        ("tabular.data", "https://raw.githubusercontent.com/fastai/fastai/master/nbs/44_tabular.data.ipynb"),
        ("tabular.learner", "https://raw.githubusercontent.com/fastai/fastai/master/nbs/45_tabular.learner.ipynb"),
        ("tabular.model", "https://raw.githubusercontent.com/fastai/fastai/master/nbs/46_tabular.model.ipynb"),
        ("collab", "https://raw.githubusercontent.com/fastai/fastai/master/nbs/47_collab.ipynb")
    ]
    
    print("Downloading official documentation from GitHub...")
    
    for title, url in github_docs:
        print(f"Downloading: {title} - {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Save the raw notebook
                filename = os.path.join(doc_dir, f"{title}.ipynb")
                save_to_file(filename, response.text)
            else:
                print(f"Failed to download {url}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")

if __name__ == "__main__":
    # Scrape docs from the website
    scrape_fastai_docs()
    
    # Create feature importance guide
    create_feature_importance_guide()
    
    # Download official docs from GitHub
    download_official_docs() 