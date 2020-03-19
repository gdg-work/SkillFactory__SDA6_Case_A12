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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# sns.set()

mark_data = pd.read_csv('Data1/sales_prediction.csv.bz2')
mark_data = mark_data.set_index('month_number')
display(mark_data.head())
# Проверка, что колонки black_friday_days и days_of_sales дублируют друг друга
assert(not (mark_data.days_of_sales - mark_data.black_friday_days).any())
mark_data = mark_data.drop('black_friday_days', axis=1)

# %% [markdown]
# Видим, что в колонках days_of_sales и black_friday_days данные одинаковые.
# Убедимся, что в колонке `month_number` (которая индекс) все номера месяцев уникальны.

# %%
# print(len(mark_data.index), mark_data.index.min(), mark_data.index.max(), mark_data.index.nunique())
assert(len(mark_data.index) == mark_data.index.nunique())

# %% [markdown]
# По условиям Гаусса-Маркова в многомерную линейную регрессию не нужно запускать зависимые
# или хотя бы коррелированные переменные. Поэтому нужно построить матрицу корреляций, которая
# показала бы нам такие переменные.
# <a name='corr_matrix_1'/>

# %%
display(mark_data.drop('Sales', axis=1).corr())

# %% [markdown]
# `marketing_total` у нас имеет высокую корреляцию с `TV`, `Internet` и `Blogs`.
# Проверяю, не является ли он суммой этих столбцов:

# %%
print("Сумма остатков от marketing_total после вычитания расходов на Интернет, блоги и ТВ: " +
      f"{(mark_data.TV + mark_data.Internet + mark_data.Blogs - mark_data.marketing_total).sum():.3e}")

# %% [markdown]
# Сумма очень небольшая, то есть предположение подтвердилось.  Исключу столбец `marketing_total` из рассмотрения.

# %%
mark_data = mark_data.drop('marketing_total', axis=1)
display(mark_data)

# %% [markdown]
# Построю тепловую карту, чтобы посмотреть, что из оставшихся параметров коррелирует друг с другом.
# Как и [ранее](#corr_matrix_1), исключу колонку Sales.

# %%
# plt.figure(figsize=(20,9))
# plt.title('Корреляции между параметрами')
# sns.heatmap(mark_data.drop('Sales', axis=1).corr(), annot=False, linewidth=0, cmap = sns.color_palette('GnBu', 20))
# plt.show()

# %% [markdown]
# На карте видим слабую корреляцию между количеством дней распродаж и расходами на рекламу в блогах и Интернете.
# Некоторые корреляции отрицательные (например, между расходами на Интернет и ТВ)

# %%
display(mark_data.drop('Sales', axis=1).corr())

# %% [markdown]
# ОК, теперь попробуем найти, что влияет на продажи. Построю матрицу корреляций с колонкой Sales, а для самых перспективных
# параметров сделаю отдельные линейные графики.

# %%
display(mark_data.drop('Sales', axis=1).corrwith(mark_data.Sales, axis=0).sort_values())

# %% [markdown]
# Судя по коэффициентам корреляции, наибольшее влияние на продажи оказывает количество дней распродаж,
# затем по убыванию идут количество дней выставок и реклама в Интернет.  Телевидение, блоги и количество
# мотивационных речей замыкают список, причём корреляция с количеством мотивационных спичей отрицательная.
# Могу предположить, что пока сотрудники слушают речи — они не работают.

# Но коэффициент корреляции не даёт всей картины, поэтому я построю графики пар "парамер - продажи".
# Начнём с наиболее сильно коррелированных.

# %% [markdown]
fig, axes = plt.subplots(1,3, figsize=(20,8))
plt.title('Корреляции различных параметров с продажами — высокие корреляции')
mark_data.loc[:,['days_of_sales',   'Sales']].plot(x='days_of_sales',   y='Sales', ax=axes[0], kind='scatter')
mark_data.loc[:,['exhibition_days', 'Sales']].plot(x='exhibition_days', y='Sales', ax=axes[1], kind='scatter')
mark_data.loc[:,['Internet',        'Sales']].plot(x='Internet',        y='Sales', ax=axes[2], kind='scatter')
plt.show()

# %% [markdown]
# Как видно из диаграмм, корреляция действительно присутствует, хотя разброс результатов (колонки Sales) очень велик.
# Для сравнения, построю диаграмму тех параметров, у которых минимальные коэффициенты корреляции с колонкой Sales

# %%
fig, axes = plt.subplots(1,3, figsize=(20,8))
plt.title('Корреляции различных параметров с продажами — низкие корреляции')
mark_data.loc[:,['TV',   'Sales']].plot(x='TV',   y='Sales', ax=axes[0], kind='scatter')
mark_data.loc[:,['Blogs', 'Sales']].plot(x='Blogs', y='Sales', ax=axes[1], kind='scatter')
mark_data.loc[:,['motivation_speeches_count', 'Sales']].plot(x='motivation_speeches_count', y='Sales', ax=axes[2], kind='scatter')
plt.show()

# %% [markdown]
# Если в диаграмме "TV" при некотором воображении можно различить относительно более плотную полосу, которая возрастает
# слева направо, то два оставшихся графика не показывают никакого влияния X на Y.
#
# Если выражать в числах то, насколько хорошо точки ложатся на линии, то полезен коэффициент детерминации ($R^2$).
# Можно рассчитать его из коэффициента корреляции Пирсона (как квадрат), но я воспользуюсь готовой функцией из
# пакета `sklearn`

# %%
def r_squared_by_name(df, column_from, column_to) -> float:
    """Расчёт коэффициента детерминации.
    Параметры:
        df - дата-фрейм с данными
        column_from - имя колонки X для регрессии, Y = aX + b
        column_to - имя колонки Y для регрессии
    Возвращает:
        коэффициент детерминации, R^2, как одно число.
    """
    model = LinearRegression()
    from_as_1col_df = df.loc[:,[column_from]]
    # хитрый синтаксис, чтобы выдался DF из одной колонки, а не Series
    model.fit(from_as_1col_df, df[column_to])
    # print(f"*DBG* Model fitted, intercept={model.intercept_}, slope={model.coef_}")
    # Функция расчёта R^2 требует двух массивов значений, реального и расчётного.
    r_squared = r2_score(df[column_to], model.predict(from_as_1col_df)) 
    return r_squared

R2_TMPL = """Коэффициент детерминации продаж (колонки Sales) параметром {:25s} составляет {:.4f}"""

r_squared_by_name(mark_data, 'days_of_sales', 'Sales')

for param in ('days_of_sales', 'exhibition_days', 'Internet', 'TV', 'Blogs', 'motivation_speeches_count'):
    print(R2_TMPL.format(param, r_squared_by_name(mark_data, param, 'Sales')))

# %% [markdown]
# Итак, при расчёте модели параметры `Blogs` и `motivation_speeches_count` можно исключить. Параметр `TV`
# тоже есть смысл исключить, хотя тут не всё однозначно и можно расчитать две модели: с учётом этого параметра
# и без.
