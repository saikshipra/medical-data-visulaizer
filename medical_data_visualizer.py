import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Step 1 - Import data
df = pd.read_csv('medical_examination.csv')

# Step 2 - Add 'overweight' column
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop('BMI', axis=1, inplace=True)

# Step 3 - Normalize cholesterol and glucose
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # Step 4 - Melt Data
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Step 5 - Group and format
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])\
                   .size().reset_index(name='total')

    # Step 6 - Draw catplot
    plot = sns.catplot(data=df_cat, 
                       kind='bar', 
                       x='variable', 
                       y='total', 
                       hue='value', 
                       col='cardio')

    fig = plot.fig
    return fig
def draw_heat_map():
    # Step 7 - Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Step 8 - Correlation matrix
    corr = df_heat.corr()

    # Step 9 - Generate mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 10 - Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Step 11 - Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    return fig
