# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Кейс А12, многофакторная линейная регрессия и прогнозирование

# %% [markdown]
# ## Знакомство с данными <a name='describe_data'/>

# %%
import pandas as pd
import seaborn as sns

mark_data = pd.read_csv('Data1/sales_prediction.csv.bz2')
mark_data = mark_data.set_index('month_number')
display(mark_data.head())
# Проверка, что колонки black_friday_days и days_of_sales дублируют друг друга
assert(not (mark_data.days_of_sales - mark_data.black_friday_days).any())

# %% [markdown]
# Видим, что в колонках days_of_sales и black_friday_days данные одинаковые.
# Убедимся, что в колонке `month_number` (которая индекс) все номера месяцев уникальны.

# %%
# print(len(mark_data.index), mark_data.index.min(), mark_data.index.max(), 
      mark_data.index.nunique())
assert(len(mark_data.index) == mark_data.index.nunique())
