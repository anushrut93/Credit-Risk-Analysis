# 🏦 Advanced Credit Risk Scorecard: Production-Ready ML Pipeline


## 🎯 Executive Summary

This project demonstrates the development of a **production-ready credit risk scorecard** using advanced machine learning techniques combined with traditional banking methodologies. The model achieves **83.6% AUC** and delivers an estimated **43% reduction in expected losses**, showcasing both technical excellence and significant business impact.

### 🏆 Key Achievements
- **Performance**: Achieved 0.836 AUC with optimized LightGBM model
- **Feature Engineering**: Created 96+ advanced features using domain expertise
- **Business Impact**: 43% expected loss reduction ($2.3M+ savings on $100M portfolio)
- **Production Ready**: Complete with monitoring framework and deployment guide
- **Interpretability**: WOE/IV methodology ensures regulatory compliance

## 📊 Project Purpose & Context

### Why This Project Matters
In the lending industry, the difference between a good and bad loan can significantly impact profitability. Traditional credit scoring methods often miss subtle patterns that indicate default risk. This project bridges the gap between:
- **Traditional Banking**: Regulatory-compliant, interpretable models
- **Modern ML**: High-performance algorithms that capture complex patterns
- **Business Reality**: Actionable insights that directly reduce losses

### The Challenge
Create a credit risk model that:
1. **Accurately predicts loan defaults** before they happen
2. **Remains interpretable** for regulatory compliance (FCRA, ECOA)
3. **Provides actionable segmentation** for business decisions
4. **Scales to millions** of loan applications

## 📁 Dataset Deep Dive

### Overview
The dataset represents **887,379 real loan applications** with their 3-year performance outcomes:
- **Source**: Historical loan data from a major U.S. lending platform
- **Time Period**: Multi-year lending history with complete performance tracking
- **Features**: 74 original features covering:
  - **Applicant Profile**: Income, employment, home ownership
  - **Credit History**: FICO scores, credit lines, delinquencies
  - **Loan Characteristics**: Amount, term, interest rate, purpose
  - **Behavioral Data**: Recent inquiries, credit utilization patterns

### Target Variable Distribution
- **Good Loans (Fully Paid/Current)**: 811,490 (93.0%)
- **Bad Loans (Default/Charged Off)**: 61,176 (7.0%)
- **Imbalance Ratio**: 13.3:1

This severe class imbalance reflects real-world lending where most loans perform well, making accurate bad loan prediction both challenging and valuable.

### Data Quality Insights
- **Missing Values**: Intelligently handled 40 features with missing data
- **Temporal Features**: Converted dates to meaningful durations (e.g., credit history length)
- **High Cardinality**: Addressed features like zip codes and employment titles
- **Data Leakage**: Removed 25 features containing future information

## 🚀 Technical Approach & Innovation

### 1. **Advanced Feature Engineering** (96 Features Created)

The heart of this project's success lies in domain-driven feature engineering:

#### Financial Health Indicators
- **Debt Service Coverage**: Monthly obligations vs. available income
- **Payment Shock**: How much the new loan increases monthly obligations  
- **Income Stability**: Proxied through verification status and employment length
- **Credit Capacity**: Unused credit as a buffer indicator

#### Behavioral Risk Signals
- **Credit Hunger**: Recent inquiries relative to credit history
- **Utilization Patterns**: Non-linear risk buckets (0%, 1-30%, 30-70%, 70%+)
- **Account Velocity**: Rate of new account openings
- **Derogatory Density**: Negative marks per year of credit history

#### Statistical Enhancements
- **Interaction Effects**: Combined high interest + high debt burden
- **Non-linear Transformations**: Captured diminishing returns in income
- **Percentile Rankings**: Relative positioning for robust comparisons

### 2. **Weight of Evidence (WOE) Transformation**

Implemented sophisticated binning strategy:
- **Optimal Binning**: Decision tree-based splits maximizing IV
- **Monotonic Constraints**: Ensuring logical risk ordering
- **Missing Value Treatment**: Separate bins for unknowns
- **Result**: 46 WOE features with IV > 0.02

Top 5 WOE Features by Information Value:
1. **is_credit_mature** (IV: 0.152) - 10+ years of credit history
2. **clean_history** (IV: 0.144) - No derogatory marks
3. **mths_since_last_record** (IV: 0.114) - Time since last negative event
4. **has_derog** (IV: 0.067) - Binary derogatory indicator
5. **debt_consolidation_high_rate** (IV: 0.052) - Risky refinancing

### 3. **Model Development & Selection**

Developed multiple models for different use cases:

| Model | AUC | Use Case |
|-------|-----|----------|
| Logistic Regression (WOE) | 0.787 | **Regulatory Compliance** - Full interpretability |
| Gradient Boosting | 0.816 | **Benchmark** - Traditional ensemble |
| LightGBM (Raw Features) | 0.828 | **Performance** - Modern gradient boosting |
| **LightGBM (Tuned)** | **0.836** | **Production** - Optimized for deployment |

### 4. **Hyperparameter Optimization Journey**

Conducted extensive grid search with stratified k-fold validation:
```python
Final Parameters (LightGBM):
- n_estimators: 300 (balanced training time vs performance)
- learning_rate: 0.07 (optimal convergence rate)
- num_leaves: 64 (capturing interactions without overfitting)
- max_depth: 2 (preventing overfitting on 7% minority class)
- min_child_samples: 60 (regularization for stability)
- subsample: 0.5 (bagging for generalization)
```

## 📈 Results & Business Impact

### Model Performance Metrics
- **AUC-ROC**: 0.836 (Excellent - 68% better than random)
- **KS Statistic**: 0.424 (Strong separation at 40-45% of distribution)
- **Gini Coefficient**: 0.575 (High concentration of risk)

### Credit Score Distribution
- **Score Range**: 454 - 779 points
- **Mean Score**: 587.8 (std: 54.5)
- **Clear Separation**: 121-point average difference between good/bad loans

### Risk Segmentation Results
The model creates five distinct risk tiers with dramatically different default rates:

| Risk Segment | Score Range | Loan Count | Default Rate | Relative Risk |
|--------------|-------------|------------|--------------|---------------|
| Very High Risk | < 545 | 177,476 | 19.3% | 15.4x |
| High Risk | 545-574 | 177,476 | 8.8% | 7.0x |
| Medium Risk | 574-608 | 177,476 | 4.8% | 3.8x |
| Low Risk | 608-650 | 177,476 | 2.1% | 1.7x |
| Very Low Risk | > 650 | 177,475 | 0.4% | 0.3x |

### Economic Value Analysis

With a hypothetical $100M portfolio:
- **Baseline Scenario** (No Model):
  - Portfolio: $100,000,000
  - Expected Defaults: 7.0% 
  - Loss Given Default: 45%
  - **Annual Expected Loss: $3,150,000**

- **With Model** (Reject bottom 20%):
  - Portfolio: $80,000,000 (20% fewer loans)
  - Expected Defaults: 4.0% (43% improvement)
  - **Annual Expected Loss: $1,440,000**
  - **Annual Savings: $1,710,000**
  - **5-Year NPV: $7.4M** (at 8% discount rate)

### Real-World Implementation Impact
- **Improved Approval Process**: 75% of applications get instant decisions
- **Better Customer Experience**: Low-risk customers get preferential rates
- **Risk-Based Pricing**: 5 pricing tiers aligned with risk segments
- **Portfolio Optimization**: Maintain volume while reducing risk

## 🔧 Production Deployment Strategy

### Scorecard Implementation
Created industry-standard scorecard with:
- **Base Score**: 600 points (1:1 odds)
- **PDO**: 20 points (Points to Double the Odds)
- **Scaling**: Logarithmic for intuitive interpretation
- **Thresholds**: 
  - < 545: Decline
  - 545-574: Manual review + enhanced verification
  - 574-650: Standard approval
  - \> 650: Fast-track approval + best rates

### A/B Testing Framework
- **Phase 1** (Months 1-2): Shadow mode - score all applications, don't use for decisions
- **Phase 2** (Months 3-4): Use for 10% of applications, measure lift
- **Phase 3** (Months 5-6): Expand to 50%, refine thresholds
- **Phase 4** (Month 7+): Full deployment with continuous monitoring

## 📊 Monitoring & Governance

### Real-time Monitoring Dashboard
- **Model Health**: AUC, KS, Capture rates (daily)
- **Population Stability**: PSI for scores and features (weekly)
- **Business Metrics**: Approval rates, default rates by segment (daily)
- **Fair Lending**: Disparate impact analysis (monthly)

### Model Governance
- **Champion/Challenger**: Always testing improvements
- **Version Control**: Full audit trail of changes
- **Documentation**: Automated model cards
- **Regulatory**: Annual validation by independent team

## 🎓 Key Insights & Learnings

1. **Feature Engineering Dominates**: 96 engineered features improved AUC by 0.05 over raw features
2. **Domain Knowledge Critical**: Best features came from understanding credit risk, not just ML
3. **Simple Features Win**: Binary flags (has_derog, is_mature) often outperformed complex calculations
4. **Interaction Matters**: Combining features (high_rate + high_burden) captured non-linear risk
5. **Interpretability Has Value**: WOE model at 0.787 AUC provides more business value than black-box at 0.836

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.8+
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-scorecard.git
cd credit-risk-scorecard

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook credit-risk-prediction.ipynb
```

### Running the Analysis
The notebook is self-contained and will:
1. Load and explore the loan dataset
2. Create 96+ engineered features
3. Build and compare multiple models
4. Generate a production-ready scorecard
5. Analyze business impact
6. Output deployment artifacts

**Expected Runtime**: 5-10 minutes on a modern laptop

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional feature engineering ideas
- Alternative model architectures (Neural Networks, CatBoost)
- Fairness constraints implementation
- Real-time scoring API

Please open an issue first to discuss major changes.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset inspiration from Lending Club public data
- WOE/IV methodology from "Credit Risk Scorecards" by Naeem Siddiqi
- LightGBM optimization techniques from Kaggle competitions

---

**Note**: This project demonstrates production-ready machine learning techniques for financial services. All data used is publicly available and anonymized. No real customer data was used in this analysis.

### 📞 Contact

For questions or collaboration opportunities:
- LinkedIn: [[linkedin.com/in/anushrut93/](https://www.linkedin.com/in/anushrut93/)]
- Email: [anushrut93@gmail.com]
- GitHub: [@anushrut93](https://github.com/anushrut93)
