# Faster Than Expected: Causal Effect of Early Delivery on Customer Satisfaction

A causal inference study on the Olist Brazilian e-commerce dataset, using Propensity Score Matching and Double Machine Learning to estimate how much arriving ahead of schedule actually moves the needle on customer reviews.

---

## The Question

Every marketplace tells sellers to ship fast. But does getting there early genuinely cause customers to leave better reviews, or do fast sellers just happen to sell better products to more patient customers?

This project answers that using a proper causal framework rather than a simple correlation. The treatment is whether an order arrived at least 3 days before the estimated delivery date. The outcome is the customer review score (1 to 5). The goal is to estimate the causal effect after stripping out all the confounding reasons why early deliveries and good reviews might travel together.

The secondary output is a seller-level opportunity ranking: given a seller who does not currently deliver early, how much would their reviews improve if they did?

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset** on Kaggle: `olistbr/brazilian-ecommerce`

Olist is a Brazilian platform that connects small businesses to major marketplaces. The dataset covers roughly 100,000 orders from September 2016 to August 2018, across all Brazilian states and dozens of product categories.

**Download:**
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/ --unzip
```

Or download manually from Kaggle and unzip into a `data/` directory at the project root.

**Files used:**

| File | Contents |
|---|---|
| olist_orders_dataset.csv | Order status, purchase and delivery timestamps |
| olist_order_items_dataset.csv | Price, freight, seller and product per order item |
| olist_order_reviews_dataset.csv | Review scores and timestamps |
| olist_products_dataset.csv | Product category, weight, dimensions |
| olist_sellers_dataset.csv | Seller state |
| olist_customers_dataset.csv | Customer state |
| olist_order_payments_dataset.csv | Payment type and installment count |
| product_category_name_translation.csv | Portuguese to English category names |

---

## Methodology

The pipeline runs in three stages.

**Stage 1: Propensity Score Matching**

Early-delivering sellers are not a random sample. They tend to be more experienced, sell lighter products, and operate closer to major logistics hubs, all of which also independently predict better reviews. PSM corrects for this by matching each treated order to three similar control orders on product category, price tier, seller experience, seller historical review rate, product weight, and geography. A caliper of 0.2 standard deviations of the propensity score rejects matches that are too distant, enforcing real overlap between groups.

The imbalance here is moderate (roughly 22% treatment rate), so the extreme multi-seed subsampling used in very unbalanced industrial settings is not necessary. A 1:3 KNN match with deduplication handles it cleanly.

**Stage 2: Double Machine Learning**

After matching, `CausalForestDML` from Microsoft's `econml` library takes over. The procedure trains a model to predict the review score from all covariates (the Y model) and a separate model to predict treatment assignment (the T model). It then fits the causal forest on the residuals of both, removing any remaining confounding that PSM did not fully address.

Both nuisance models are selected via Optuna hyperparameter search, comparing LightGBM against XGBoost across 30 trials each, with out-of-fold MSE for the outcome model and AUC-ROC for the propensity model.

The `groups` parameter in the DML fit is set to `seller_id` because multiple orders from the same seller are correlated. This clusters standard errors at the seller level, making confidence intervals more honest.

**Stage 3: CATE Estimation and Ranking**

The causal forest produces an individual treatment effect (CATE) for each test order, plus a 95% confidence interval. These are aggregated to the seller level using a precision-weighted average (weight = 1 / CI width). The output ranking targets sellers with a low early-delivery rate, for whom the CATE represents the expected review improvement if their logistics improved.

**Validation (GATES)**

Test orders are split into quintiles by predicted CATE. The observed review gap between treated and control orders in each quintile should increase from Q1 to Q5 if the model correctly ranks heterogeneity. The Pearson correlation between predicted quintile means and observed gaps is the main validation metric.

---

## Project Structure

```
.
├── data/                              # Unzipped Olist dataset (not committed)
├── notebook.md                        # Notebook, one cell per #### Cell N block
├── olist_delivery_cate_ranking.csv    # Generated: seller opportunity ranking
└── README.md
└── requirements.txt
```

---

## Setup and Execution

Python 3.9 or higher. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm econml optuna plotly seaborn matplotlib
```

Set `DATA_DIR` in Cell 9 to the folder containing the Olist CSVs, then run all cells in order. Estimated runtimes on a standard laptop:

- Data loading and feature engineering: 1-2 minutes
- PSM: 2-4 minutes
- Optuna (4 model searches, 30 trials each): 15-25 minutes
- CausalForestDML training: 3-5 minutes
- Total: roughly 25-40 minutes

---

## Reading the Output

**ATE:** The average causal impact across all test orders. A positive value with a confidence interval that does not cross zero is evidence of a real effect.

**CATE histogram (Cell 13):** Shows how broadly or narrowly the effect is distributed. A wide distribution means some order types benefit much more than others, which is exactly what makes the ranking useful.

**GATES table (Cell 14):** If the correlation between predicted CATE quintiles and observed T-C gaps is above 0.5 the model discriminates well. Values between 0.3 and 0.5 are realistic for noisy review data and do not invalidate the causal estimates.

**Heatmap (Cell 16):** Mean CATE by product category and seller region. Identifies where faster logistics would have the largest payoff in satisfaction scores.

**Opportunity ranking (Cell 17):** `olist_delivery_cate_ranking.csv` lists sellers sorted by precision-weighted CATE. The `opportunity_score` column is a 0-100 percentile rank, ready to hand to a commercial or operations team prioritizing logistics investments.

---

## Limitations

**Seller features computed on full data.** `seller_avg_review` and `seller_early_rate` are computed across all orders before being assigned back to each order. The methodologically correct approach uses an expanding window to exclude the current order. For production use, this should be fixed; for a portfolio study the impact is minor since these features are controls, not outcomes.

**Unobserved confounders.** PSM and DML control for what is measured. Product fragility, delivery address accessibility, and seller packaging investments are not in the dataset and remain as potential sources of residual bias.

**Review score noise.** Customer ratings reflect product quality and seller communication as much as delivery timing. This structural noise limits model R-squared and puts a ceiling on GATES correlation, which is expected and does not mean the causal estimates are wrong.

---

## Extensions Worth Exploring

An RDD analysis around the 0-day and 3-day thresholds in `days_early` would complement the DML results. Sellers just barely above and just below the delivery estimate differ very little, making the discontinuity a credible natural experiment within the existing data.

Switching to a seller-level panel with quarterly aggregation would allow removing seller fixed effects via first-differencing, following the same structure as a proper DiD study. This would make the identification assumptions even more defensible.

Adding NLP-derived sentiment from the review comment text as a supplementary outcome could reduce noise and sharpen the GATES correlation, since written reviews tend to be more specific about delivery than the numeric score alone.

---

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1-C68.

Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.

Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.