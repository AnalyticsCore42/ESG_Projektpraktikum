# ESG Emissions Prediction Model Summary

## Overview of Machine Learning Approaches
This document summarizes the various machine learning and deep learning approaches we've implemented to predict carbon emissions trends, along with their performance and insights.

## 1. FastAI TabularLearner Model (Neural Network)
**Approach:**
- Used FastAI's TabularLearner to create a neural network model
- Implemented in `14_fastai_emissions_predicitions.py`
- Target variable: `CARBON_SCOPE_12_INTEN_3Y_GIC_CAGR` (3-year CAGR of carbon intensity)

**Features:**
- Categorical features: Country, NACE codes, emission keys
- Continuous features: Market cap, sales, emission intensities, absolute emissions

**Preprocessing:**
- Winsorized outliers (1% on each tail)
- Applied Yeo-Johnson power transformation for skewed features
- Added interaction features (size_emissions_ratio)
- Created industry-specific and regional benchmark features

**Performance:**
- Moderate predictive power
- Validation R²: ~0.10-0.15
- Good at capturing general patterns but struggled with industry-specific nuances

**Insights:**
- Emission intensity metrics were the strongest predictors
- Industry classification showed high importance
- Interaction between size and emissions proved valuable

## 2. Industry-Segmented Neural Network Model
**Approach:**
- Extended the FastAI model with industry-specific segmentation
- Implemented in `15_industry_segmented_emissions_prediction.py`
- Trained separate models for major industry groups

**Features:**
- Same feature set as the general model
- Added industry-specific aggregations and benchmarks

**Performance:**
- Varied significantly by industry
- Some industries (Professional Services, Info & Comm) achieved R² > 0.20
- Other industries showed minimal improvement or worse performance

**Insights:**
- Industry-specific modeling improved predictions for certain sectors
- Different industries showed distinct feature importance patterns
- Smaller industries suffered from limited training data

## 3. Gradient Boosting Models (LightGBM)
**Approach:**
- Implemented gradient boosting using LightGBM
- Implemented in `16_gbm_industry_emissions_prediction.py`
- Compared general model vs. industry-specific models

**Features:**
- Similar feature set to previous models
- Careful handling of categorical features
- Enhanced feature engineering for interaction terms

**Performance:**
- General model: R² = 0.1252
- Top-performing industry models:
  - Professional Services: R² = 0.3184
  - Information & Communication: R² = 0.2259
  - Arts & Entertainment: R² = 0.2028
  - 3 other industries outperformed the general model

**Insights:**
- GBM outperformed neural networks for this prediction task
- Industry segmentation significantly improved performance for key sectors
- Emission intensity metrics remained the strongest predictors
- Company size indicators (sales, market cap) showed high importance
- Target and program features did not emerge among top predictors

## 4. Feature Importance Analysis

**Key predictive factors across models:**
1. **Existing emissions profile**:
   - Emissions intensity (CARBON_EMISSIONS_SCOPE_12_INTEN)
   - Historical trends (CARBON_SCOPE_12_INTEN_3Y_AVG)
   - Scope breakdown (Scope 1 vs. Scope 2 intensity)

2. **Company characteristics**:
   - Size (SALES_USD_RECENT, MarketCap_USD)
   - Industry classification (NACE_CLASS_CODE)

3. **Relatively less important factors**:
   - Country/region
   - Targets and programs
   - Governance indicators

## Future Directions

1. **Enhanced Program and Target Analysis**:
   - Create more nuanced derived features from program and target data
   - Develop program implementation quality scores
   - Add interaction terms between program types and industry

2. **Geographic Enhancement**:
   - Incorporate country-level policy stringency indicators
   - Add regional electricity grid carbon intensity metrics
   - Create country-industry interaction features

3. **Time Series Approaches**:
   - Explore temporal models to better capture emission trends
   - Implement LSTM or other sequence models for companies with sufficient historical data

4. **Ensemble Methods**:
   - Combine predictions from multiple model types
   - Build meta-models that leverage strengths of different approaches

5. **Advanced Interpretability**:
   - Apply SHAP values for deeper feature importance understanding
   - Develop counterfactual explanations for emissions predictions
   - Create scenario analysis tools based on the models

6. **Causal Inference**:
   - Explore causal modeling to better understand program impact
   - Implement quasi-experimental designs to isolate program effects

## Conclusion

Our modeling approaches have evolved from general neural networks to more sophisticated industry-segmented gradient boosting models. The results demonstrate that:

1. **Industry-specific modeling** significantly outperforms general approaches
2. **Gradient boosting models** appear more suitable than neural networks for this prediction task
3. **Existing emissions profiles** and **company characteristics** are stronger predictors than targets or programs
4. **Professional Services**, **Information & Communication**, and other knowledge-based industries show higher predictability

The current direction is focusing on better understanding the causal relationship between sustainability programs and emissions trends, with an emphasis on industry-specific patterns and more nuanced feature engineering for program and target data. 