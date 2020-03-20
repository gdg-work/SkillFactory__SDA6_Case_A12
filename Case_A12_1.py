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
#
# ### Содержание
# - [Знакомство с данными](#describe_data)
# - [Подготовка данных к регрессии](#prepare)
#   + [Выводы по подготовительному этапу](#predv_conclusions)
# 
# - [Многомерная линейная регрессия](#mv_regression)
#   + [Проверка качества модели](#model_quality)
#   + [Итоговая модель](#final_model)
# 
# - [Общие выводы по исследованию](#conclusions)

# %% [markdown]
# ## Знакомство с данными <a name='describe_data'/>

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.graphics.gofplots import qqplot # Импорт части библиотеки для построения qq-plot
sns.set()
plt.ioff()

mark_data = pd.read_csv('Data1/sales_prediction.csv.bz2')
mark_data = mark_data.set_index('month_number')
display(mark_data.head())

# Проверка, что колонки black_friday_days и days_of_sales дублируют друг друга. Удаляю одну из них.
assert(not (mark_data.days_of_sales - mark_data.black_friday_days).any())
mark_data = mark_data.drop('black_friday_days', axis=1)

# %% [markdown]
# Построю пару графиков. Начну с самых простых, "продажи от времени" и "расходы на маркетинг от времени"

# %%
# 
ax = mark_data.plot(y='Sales', kind='line', figsize=(12,5))
ax.set_title("Изменение продаж по месяцам")
ax.set_xlabel("Номер месяца")
ax.set_ylabel('Sales')
plt.show()

## %%
ax = mark_data.plot(y='marketing_total', kind='line', figsize=(12,5))
ax.set_title("Расходы на маркетинг по месяцам")
ax.set_xlabel("Номер месяца")
ax.set_ylabel('Marketing_Total')
plt.show()

# %% [markdown]
# Структура маркетинговых расходов
# Покажу доли, которые занимают расходы на разные виды маркетинга в общем объёме расходов

# %%
mark_shares = mark_data.loc[:,['Internet', 'TV', 'Blogs']]
mark_shares['inet_share']  = mark_shares.Internet / (mark_shares.Internet + mark_shares.TV + mark_shares.Blogs)
mark_shares['TV_share']    = mark_shares.TV       / (mark_shares.Internet + mark_shares.TV + mark_shares.Blogs)
mark_shares['Blogs_share'] = mark_shares.Blogs    / (mark_shares.Internet + mark_shares.TV + mark_shares.Blogs)
print("Сумма средних долей расходов: {:.3f}".format(
    mark_shares.Blogs_share.mean() + mark_shares.TV_share.mean() + mark_shares.inet_share.mean()
    ))
print("""Средняя труктура маркетинговых расходов за всё время наблюдений:
  - TV      : {0:.2%}
  - Internet: {1:.2%}
  - Blogs   : {2:.2%}
""".format(
    mark_shares.TV_share.mean(), mark_shares.inet_share.mean(), mark_shares.Blogs_share.mean()
))

# %% [markdown]
# Открытые вопросы к данным:
# 
# - Первой колонкой идёт номер месяца, в интервале от 1 до 231.  При этом у меня вызывает большие сомнения
#   подобная длительность существования *интернет-магазина светильников*: в 2000 году интернет-магазины,
#   по крайней мере в России, не были особо популярны.  И Интернет был гораздо в большей степени «игрушкой для гиков»,
#   чем сейчас.
#
# - В каких единицах выражены данные в колонках расходов и в колонке Sales?  Учтена ли инфляция (в противном случае
#   особого смысла в сравнении и всех дальнейших построениях нет)
#
# - За почти 20 лет не замечено существенного роста продаж.  Торговля не развивается?
#
# - Продажи очень резко изменяются от месяца к месяцу, и что более удивительно — так же резко меняются и расходы
#   на маркетинг.
#
# Но будем считать, что это модельные данные и на все эти вопросы ответит кто-то другой в другом месте и в другое время.

# %% [markdown]
# ## Подготовка данных к регрессии <a name='prepare'/>

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
mark_total_ser = mark_data.marketing_total.copy()
mark_data = mark_data.drop('marketing_total', axis=1)
display(mark_data.head(3))

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

# %%
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
# ### Выводы по подготовительному этапу <a name='predv_conclusions'/>
# 
# Итак, при расчёте модели параметры `Blogs` и `motivation_speeches_count` можно исключить.
# C параметром `TV`не всё однозначно. Построю две модели: использующую для входных данных только
# `days_of_sales`, `exhibition_days` и `Internet` , и добавляющую к этому набору `TV`

# %% [markdown]
# ## Многомерная линейная регрессия <a name='mv_regression'/>

# %%
display(mark_data.head())
params_df      = mark_data.drop(['Blogs', 'motivation_speeches_count', 'TV', 'Sales'], axis=1)
params_df_w_tv = mark_data.drop(['Blogs', 'motivation_speeches_count', 'Sales'], axis=1)

mv_model         = smf.ols('Sales ~ days_of_sales + exhibition_days + Internet',      data=mark_data).fit()
mv_model_with_tv = smf.ols('Sales ~ days_of_sales + exhibition_days + Internet + TV', data=mark_data).fit()

# %% [markdown]
# Метод `.summary()` показывает информацию о модели.

# %% 
print("=====================================" + "\n" +
      "=         Model without TV          =" + "\n" +
      "=====================================")
display(mv_model.summary())

# %%
print("==================================" + "\n" +
      "=         Model WITH TV          =" + "\n" +
      "==================================")
display(mv_model_with_tv.summary())

# %% [markdown]
# Судя по числам в кратком описании модели, влияние расходов на ТВ рекламу в изменения продаж оказались
# достаточно мало (менее 5%), чтобы им можно было пренебречь.

# %% [markdown]
#
# Построим скаттерограммы:

# %%
sales_pred_df    = pd.DataFrame.from_dict({'sales': mark_data.Sales, 'pred': mv_model.fittedvalues})
sales_pred_tv_df = pd.DataFrame.from_dict({'sales': mark_data.Sales, 'pred': mv_model_with_tv.fittedvalues})

fig, axes = plt.subplots(1,2, figsize=(12,5))
fig.suptitle('Предсказание против исходных данных')
__ = sales_pred_df.plot(   x='pred', y='sales', kind='scatter', alpha=0.6, ax=axes[0])
__ = sales_pred_tv_df.plot(x='pred', y='sales', kind='scatter', alpha=0.6, ax=axes[1])
axes[0].set_title("Без TV")
axes[1].set_title('с TV')
for ax in axes:
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    ax.set_xlim(xmin=20)
    ax.set_ylim(ymin=20)
plt.show()

# %% [markdown]
# Видим, что в обоих случаях предсказание не идеально, но работает достаточно прилично.

#
# Модель с учётом телевидения получилась *хуже*, чем без него.  Это видно по параметрам.
# Кроме того, 95% доверительный интервал для Intercept в модели с телевидением развалился:
# он от 4.6 до 23.  В модели безе телевидения интервал поприличнее: 14.4 до 23.9.
#
# Параметр `Durbin-Watson` показывает отсутствие автокорреляции в данных для обеих моделей.

# %% [markdown]
# ### Проверка качества модели <a name='model_quality'/>

# %% [markdown]
#
# Качество нашей работы оценим двумя способами: распределением ошибок (QQ-Plot) и гистограммой
# остатков.
# 
# Идеальная ошибка должна быть невелика, независима от исходных данных и распределена нормально.
# На этом графике точки должны попасть на прямую:

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].set_title("Без TV")
axes[1].set_title("С TV")
qqplot(mv_model.resid, line = 's', alpha=0.5, ax=axes[0])
qqplot(mv_model_with_tv.resid, line = 's', alpha=0.5, ax=axes[1])
plt.show()

# %% [markdown]
#
# Независимость ошибки от исходных данных проверим корреляцией остатка с исходными параметрами.
#

# %%
print("\n" + "Модель «без телевизора»")
display(mark_data.drop('Sales', axis=1).corrwith(mv_model.resid, axis=0))
print("\n" + "Модель «с телевизором»»")
display(mark_data.drop('Sales', axis=1).corrwith(mv_model_with_tv.resid, axis=0))

# %% [markdown]
# В первой модели ошибки несколько коррелированы с расходами на телевизионную рекламу,
# но коэффициент корреляции достаточно мал, всего 8%. Остальные корреляции этой модели,
# а также все корреляции модели «с телевизором» менее 2%, то есть пренебрежимо малы.

# %% [markdown]
# Распределение ошибки можем оценить по гистограммам для обеих моделей.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].set_title("Без TV")
axes[1].set_title("С TV")
fig.suptitle('Отклонение от предсказанного значения')
mv_model.resid.plot(kind='hist',  ax=axes[0])
mv_model_with_tv.resid.plot(kind='hist',  ax=axes[1])
plt.show()

# %% [markdown]
# ### Итоговая модель <a name='final_model'/>

# %% [markdown]
# Из двух моделей, которые дают близкие результаты, возьму более простую, не учитывающую
# расходы на телевидение.  Коэффициент при этих расходах около 0.02, то есть зависимость
# продаж от расходов на телерекламу минимальна.
# 
# $$
# Sales = 0.54 \times days\_of\_sales + 0.5 \times exhibition\_days + 0.17 \times Internet + 19.13
# $$

# %% [markdown]
# ## Выводы по исследованию <a name='conclusions'/>

# %% [markdown]
# 1. 70% маркетингового бюджета тратится на методы, которые не коррелируют с продажами.
# 2. Судя по представленным данным, наиболее эффективны распродажи, участие в выставках и реклама в Интернет.
# 3. Есть смысл постепенно снижать долю маркетингового бюджета, которая расходуется на ТВ и блоги, эти методы не дают эффекта.
# 4. Мотивационные собрания и речи не отражаются на продажах.
# 5. Рекламу в Интернет стоит диверсифицировать, на одном Яндекс.Директ свет клином не сошёлся.  Есть специализированные
# сайты, есть другие поисковые системы.
# 6. Линейная прогностическая модель представлена в [отдельном  разделе выше](#final_model)
