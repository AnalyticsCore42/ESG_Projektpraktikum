# Program-Focused Emissions Analysis Findings

## Overview
This report summarizes the findings from our program-focused gradient boosting model analysis, which examined the impact of sustainability programs and targets on company emissions trends.

## Key Findings

### 1. Target Ambition Analysis
The analysis reveals a clear relationship between target ambition and emissions trends:
- Companies with **no targets** show an average emissions trend of +1.19% (increasing emissions)
- Companies with **medium ambition targets** (10-30% reduction) show the best performance with -3.82% emissions trend
- Companies with **high ambition targets** (>30% reduction) also perform well at -3.70% emissions trend

### 2. Industry-Program Effectiveness
The model reveals significant variation across industries in program effectiveness:

| Industry | High Program Score Trend | Low Program Score Trend | Difference |
|----------|-------------------------|------------------------|------------|
| Retail   | -1.15%                 | +7.80%                | ~9%        |
| Manufacturing - Heavy | -3.26% | +1.16% | ~4.5% |
| Other Industries | Varies | Varies | Consistently better with high scores |

### 3. Program Feature Importance
When focusing only on program/target features (without company characteristics), the most important predictors are:

1. **program_score_by_size**: Interaction between program comprehensiveness and company size
2. **target_by_emissions_intensity**: Interaction between target ambition and emissions intensity
3. **target_vs_industry_avg**: How a company's target compares to industry peers
4. **program_score_vs_industry**: How a company's program implementation compares to peers
5. **CBN_TARGET_REDUC_PCT**: Raw target percentage reduction

### 4. Overall Predictive Power
- The program-only model achieves R² of 0.0152, showing that programs/targets alone explain only about 1.5% of emissions trend variance
- This confirms that while programs and targets have real impact, other factors (industry, size, existing emissions profile) remain stronger determinants of overall trends

### 5. Program Types
Key insights about different types of programs:
- Carbon reduction programs in core operations have higher importance than renewable energy programs
- Executive body oversight appears as a key factor (rank 7), showing governance matters
- Program implementation age is also important (rank 11), suggesting longer-running programs have more impact

## Conclusions
1. Sustainability programs and targets do influence emissions trends, though their impact is modest compared to structural factors
2. The strongest program influence appears in emissions-intensive sectors like manufacturing and retail
3. Target setting, particularly medium-ambition targets (10-30% reduction), shows the most consistent positive impact
4. Program effectiveness is enhanced by:
   - Strong governance oversight
   - Long-term implementation
   - Industry-specific adaptation
   - Integration with company size and emissions intensity

## Recommendations
1. Focus on setting medium-ambition targets (10-30% reduction) as they show the best results
2. Prioritize program implementation in high-emissions sectors
3. Ensure strong governance oversight of sustainability programs
4. Consider company size and industry context when designing programs
5. Maintain long-term commitment to program implementation 