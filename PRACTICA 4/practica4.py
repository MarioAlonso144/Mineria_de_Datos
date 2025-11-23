import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from itertools import combinations

data = pd.read_csv('ia_financial_market_dataset.csv')

print(data.head())


model = ols('prediction_next_day_return ~ C(ticker)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA result:")
print(anova_table)

groups = [data[data['ticker'] == t]['prediction_next_day_return'] for t in data['ticker'].unique()]

kruskal_stat, kruskal_p = stats.kruskal(*groups)
print("\nKruskal-Wallis test result:")
print(f"Statistic: {kruskal_stat}, p-value: {kruskal_p}")

tickers = data['ticker'].unique()
print("\nT-tests between tickers:")
for t1, t2 in combinations(tickers, 2):
    t_stat, p_val = stats.ttest_ind(
        data[data['ticker']==t1]['prediction_next_day_return'],
        data[data['ticker']==t2]['prediction_next_day_return']
    )
    print(f"{t1} vs {t2} -> t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")