import gradio as gr
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

housing = pd.read_csv(r'..\Datasets\Housing.csv')
intervals = [(0, 50000), (50000, 100000), (100000, 150000), (150000, 200000)]
bins = pd.IntervalIndex.from_tuples(intervals)
housing['price_slab'] = pd.cut(housing['price'], bins)
intervals = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]
bins = pd.IntervalIndex.from_tuples(intervals)
housing['area_slab'] = pd.cut(housing['lotsize'], bins)
fp_df = housing.drop(['price','lotsize'], axis=1)
fp_df = fp_df.astype(object)

def gen_rules(min_sup, min_conf):

    itemsets = apriori(fp_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(
        itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[['antecedents', 'consequents',
                   'support', 'confidence', 'lift']]
    rules['antecedents'] = [list(x) for x in rules['antecedents'].values]
    rules['consequents'] = [list(x) for x in rules['consequents'].values]
    rules.sort_values(by='lift', ascending=False)
    rules = rules[rules['lift'] > 1]
    return rules


demo = gr.Interface(fn=gen_rules,
                    inputs=[gr.Slider(value=0.01, step=0.01,
                                      label="Minimum Support",
                                      minimum=0.0001, maximum=1),
                            gr.Slider(value=0.01, step=0.01,
                                      label="Minimum Confidence",
                                      minimum=0.0001, maximum=1)],
                    outputs='dataframe')

if __name__ == "__main__":
    demo.launch()
