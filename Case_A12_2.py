#!/usr/bin/env python
# —-
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
# —-

# %% [markdown]
# # Кейс А12, многофакторная линейная регрессия и прогнозирование
#
# ### Содержание
# 
# * [Знакомство с данными](#describe_data)
# 
# * [Построение математической модели продаж](#modelling)
#   - [Подготовка данных](#prepare_data)
#     + [Месяцы с неправильным количеством дней](#strange_months)
#     + [Избавление от корреляций в исходных данных](#remove_correlations)
#     + [Подготовка данных к моделированию](#prepare)
#     + [Корреляции целевого параметра с переменными](#partial_correlations)
# 
#   - [Моделирование](#model)
#   - [Оценка качества модели](#model_quality)
#     + [Оценка разброса остатков](#modchk_disp)
#     + [Проверка нормальности распределения остатков](#modchk_norm)
#     + [Проверка корреляций с исходными данными](#modchk_corr)
# 
#    - [Итоги моделирования](#model_results)
# 
# * [Выводы по исследованию](#conclusions)
# 

# %% [markdown]
# Код в этом документе по умолчанию скрыт (после первого запуска).
# Для того, чтобы показать код, нажмите кнопку ниже.

# %%
from IPython.display import HTML
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>
''')


# %% [markdown]
# ## Знакомство с данными <a name='describe_data'/>

# %%
# Imports (I see, too many :) )
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.graphics.gofplots import qqplot # Импорт части библиотеки для построения qq-plot
from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif
from math import sqrt
from matplotlib.axes._axes import _log as matplotlib_axes_logger

# Some initialization
sns.set()
matplotlib_axes_logger.setLevel('ERROR')  # disable color warnings in Matplotlib
sns.set_style('whitegrid')
plt.ioff()

# make the program work outside Jupyter and iPython
try:
    __ = get_ipython()
except NameError:
    def display(*args, **kwargs):
        print(*args, **kwargs)

# read data
mark_data = pd.read_csv('https://miscelanneous.s3-eu-west-1.amazonaws.com/SkillFactory/SDA_CaseA12/lights_sales_data.csv.bz2')
mark_data = mark_data.set_index('month_number')

# %% [markdown]
# Для начала посмотрим, какие есть колонки и сколько значений, потом поищем связи между колонками.

# %%
print(mark_data.info())
# Колонки
print(mark_data.head(3))


# %% [markdown]
# Видимо, наша целевая колонка — продажи.  Что с ними происходит со временем?

# %%
fig, (ax_sales, ax_sales_hist) = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios': (4,1)}) 
fig.suptitle('Продажи по месяцам и их гистограмма') 
mark_data.sales.plot(kind='line', ax=ax_sales) 
mark_data.sales.plot(kind='hist', ax=ax_sales_hist) 
ax_sales.set_ylabel('Продажи') 
ax_sales_hist.set_xlabel('Продажи') 
plt.show()

# %% [markdown]
# График продаж не показывает роста за 200 месяцев. Зачем нужен это магазин? :) Видимо, данные
# синтетические. А что происходит с маркетинговыми расходами?

# %%
fig, (ax_mkt, ax_mkt_hist) = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios': (4,1)}) 
fig.suptitle('Расходы на маркетинг и их гистограмма') 
mark_data.total_marketing_spendings.plot(kind='line', ax=ax_mkt) 
mark_data.total_marketing_spendings.plot(kind='hist', ax=ax_mkt_hist) 
ax_mkt.set_ylabel('Маркетинговые расходы') 
ax_mkt_hist.set_xlabel('Маркетинговые расходы') 
plt.show()

# %% [markdown]
# Распределение доходов от продаж колокообразное, а расходов на маркетинг более интересное — бимодальное.
# Скорее всего, это связано с тем, что общие расходы на маркетинг — это составной параметр,
# который образуется из других колонок (как будет показано ниже).

# %% [markdown]
# Какие у нас есть интересные колонки? Например, расходы на маркетинг представлены в двух валютах.
# Посмотрим, изменялся ли курс за 200 месяцев, то есть выведем график "расходы в долларах от расходов
# в рублях"
# Следующая интересная колонка — количество дней в месяце. Можно посмотреть, как меняются продажи
# в зависимости от этого показателя, заодно увидим и изменения количества дней.

# %%
fig, (ms_ax, sl_ax) = plt.subplots(1,2, figsize=(12,5))
mark_data.plot(x='total_marketing_spendings', y='total_marketing_spendings, $', kind='scatter', ax=ms_ax)
mark_data.plot(x='days_in_month', y='sales', kind='scatter', ax=sl_ax)
ms_ax.set_title('Маркетинговые расходы в USD и рублях')
sl_ax.set_title('Продажи от количества дней в месяце')
plt.show()

# %% [markdown]
# Судя по красивой прямой на графике "расходы в рублях и в долларах", всё приведено к одному курсу USD.
# Поэтому дальнейшие исследования можно делать в любой из валют.  Оставляю рубли.

# Правая диаграмма показывает, что наряду с обычными месяцами, в которых 31 день, 30 дней и 28/29 дней,
# есть странные "месяцы" с 10-25 днями. Причём не видно влияния количества дней в месяце на продажи.

# %% [markdown]
# ## Построение математической модели продаж <a name='modelling'/>

# %% [markdown]
# ### Подготовка данных <a name='prepare_data'/>

# #### Месяцы с неправильным количеством дней <a name='strange_months'/>

# Попробую понять, что за месяцы со странным количеством дней. Известно, что количество дней в месяцах одного
# года меняется как: [31, 28(29), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]. Что у нас в данных?

# %%
print("\n Нормальное количество дней в месяцах, начиная с февраля, 200 месяцев (без учёта високосных лет)")
display(pd.Series( ([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] * 17)[1:201] ).value_counts())
print("\n Количество дней из данных, цифра слева -- кол-во дней, справа - месяцев с этим количеством")
mark_data.days_in_month.value_counts()

# %% [markdown]
# В данных должно было встретиться 3-4 високосных февраля в зависимости от того, попадался ли год, номер
# которого кратен 100.
#
# Видим, что у нас недостаток по всем месяцам, но разный: не хватает одного месяца с 31 днём, 14 месяцев с 30 днями,
# одного стандартного февраля и двух високосных.
#
# А как ведут себя расходы по разным видам рекламных носителей, опросы и распродажи в зависимости от числа дней в месяце?

# %%
fig, ((ax_tv, ax_rd, ax_np), (ax_dos, ax_surv, ax_tot)) = plt.subplots(2, 3, figsize=(12, 9), sharex='col')
fig.suptitle('Зависимость маркетинговых расходов от количества дней в месяце')
mark_data.plot(x='days_in_month', y='TV',            kind='scatter', ax=ax_tv,   alpha=0.5)
mark_data.plot(x='days_in_month', y='Radio',         kind='scatter', ax=ax_rd,   alpha=0.5)
mark_data.plot(x='days_in_month', y='Newspapers',    kind='scatter', ax=ax_np,   alpha=0.5)
mark_data.plot(x='days_in_month', y='days_of_sales', kind='scatter', ax=ax_dos,  alpha=0.5)
mark_data.plot(x='days_in_month', y='Surveys',       kind='scatter', ax=ax_surv, alpha=0.5)
mark_data.plot(x='days_in_month', y='total_marketing_spendings', kind='scatter', ax=ax_tot, alpha=0.5)
plt.show()


# %% [markdown]
# Какая-то зависимость видна только для количества дней распродаж. Предполагаю, что «странные»
# значения в колонке `days_in_month` внесены по ошибке.
# 
# Принимаю волюнтаристское решение всем слишком коротким месяцам
# присвоить количество дней, равное 30.

# %%
mark_data['days_in_month'] = mark_data.days_in_month.apply(lambda x: 30 if x < 28 else x)

# %% [markdown]
# #### Избавление от корреляций в исходных данных <a name='remove_correlations'/>

# %% [markdown]
# Есть подозрение, что колонка `total_marketing_spendings` является суммой других колонок. Каких именно?
# Попробую включить все подозрительные колонки

# %%
display((mark_data.total_marketing_spendings - mark_data.TV - mark_data.Radio - mark_data.Newspapers - mark_data.Surveys).head(3))

# %% [markdown]
# получилось содержимое колонки Surveys с обратным знаком. Значит, она не входит в total_marketing_spendings.

# %%
display((mark_data.total_marketing_spendings - mark_data.TV - mark_data.Radio - mark_data.Newspapers).head(3))

# %% [markdown]
# Остатки очень малы, скорее всего, это погрешности округления. Посчитаем сумму и среднеквадратическое отклонение от нуля.

# %%
test = mark_data.total_marketing_spendings - mark_data.TV - mark_data.Radio - mark_data.Newspapers
print("""Остатки total_marketing_spendings после вычитания отдельных компонентов:
Сумма остатков: {0:.3e}
Среднеквадратическое (корень из суммы квадратов): {1:.3e}
""".format(
  test.sum(),
  sqrt((test**2).sum())
))
del(test)

# %% [markdown]
# Остатки малы как по отдельности, так и в сумме. Считаю их ошибками округления. Тогда колонка `total_marketing_spendings`
# зависит от колонок `TV`, `Radio`, `Newspapers` и не должна использоваться для моделирования.  Но пока мы её не удалили,
# можно использовать эту колонку для расчёта долей разных рекламных носителей в расходах.

# %%
# Отдельный датафрейм для долей (буду строить боксплот)
mkdata_shares = pd.DataFrame(index=mark_data.index)
mkdata_shares["TV_share"]  = mark_data.TV         / mark_data.total_marketing_spendings
mkdata_shares["Radio_share"] = mark_data.Radio      / mark_data.total_marketing_spendings
mkdata_shares["Newspapers_share"]  = mark_data.Newspapers / mark_data.total_marketing_spendings
# Посчитаю средние доли
print("Средние доли расходов на рекламные носители")
display(mkdata_shares.mean())
# Но средние доли мало что дают, когда речь идёт о разбросе, построю боксплот
sns.violinplot(data=mkdata_shares)
plt.show()
display(mkdata_shares.corr())
del(mkdata_shares)

# %% [markdown]
# Итак, больше всего расходов (в среднем две трети) на телевизионную рекламу, оставшаяся треть делится
# между газетами (расходы чуть больше) и радио (минимальные расходы). От месяца к месяцу расходы сильно
# изменяются.
 
# %% [markdown]
# #### Подготовка данных к моделированию <a name='prepare'/>

# %% [markdown]
# Для моделирования возьму колонки с расходами на рекламные носители, опросами,
# количеством дней распродаж  и результирующую колонку продаж.

# %%
# Делаю независимую от исходных данных таблицу
mkdata_fixed=pd.DataFrame(index=mark_data.index)
mkdata_fixed['TV']            = mark_data.TV.copy()
mkdata_fixed['Radio']         = mark_data.Radio.copy()
mkdata_fixed['Newspapers']    = mark_data.Newspapers.copy()
mkdata_fixed['Surveys']       = mark_data.Surveys.copy()
mkdata_fixed['days_of_sales'] = mark_data.days_of_sales.copy()
mkdata_fixed['sales']         = mark_data.sales.copy()
display(mkdata_fixed.head(n=4))

# %% [markdown]
# Проверим взаимные корреляции в таблице `mkdata_fixed`

# %%
display(mkdata_fixed.drop('sales', axis=1).corr())

# %% [markdown]
# Есть заметная (17%) корреляция между телевидением и количеством дней распродаж. Это
# логично, о распродажах есть смысл объявнять по телевидению, у него максимальный
# охват аудитории.


# %% [markdown]
# Ещё одна полезная характеристика мультиколлинеарности — VIF (variande inflation factors)
# Проверю и этот параметр.
 
# %%
def calculate_vifs(df: "Data frame to compute VIFs in", exclude_cols=[]) -> dict:
    """
    Расчёт факторов VIF в дата-фрейме.  Параметры:
        1) датафрейм для подсчёта,
        2) список имён колонок, которые нужно исключить.
    Подразумевается, что имена колонок уникальны.
    Возвращает: словарь {имя_колонки: значение_vif (float)}
    """
    cols = [c for c in list(df.columns) if not c in exclude_cols]
    vif_dict = {}
    for idx, col in enumerate(cols):
        vif_dict[col] = sm_vif(df.loc[:,cols].values, idx)
    return vif_dict

print("Расчёт VIF для таблицы 'mkdata_fixed': ", calculate_vifs(mkdata_fixed, 'sales'))


# %% [markdown]
# Не стал возиться с красивым оформлением этого вывода — данные технические.  VIF
# около критического значения, но не превосходят его.  Это говорит о том, что регрессию выполнять можно.

# %% [markdown]
# #### Корреляции целевого параметра с переменными <a name='partial_correlations'/>

# %% [markdown]
# Расчитаю корреляции целевого параметра `sales` с переменными: различными маркетинговыми
# расходами, днями распродаж и количеством опросов.

# %%
display(mkdata_fixed.drop('sales', axis=1).corrwith(mkdata_fixed.sales))
# И строю картинку 
fig, ((ax_tv, ax_rd, ax_np), (ax_sv, ax_dos, __)) = plt.subplots(2,3, sharey='row', figsize=(12,8))
mkdata_fixed.plot(x='TV',            y='sales', kind='scatter', ax=ax_tv, alpha=0.5)
mkdata_fixed.plot(x='Radio',         y='sales', kind='scatter', ax=ax_rd, alpha=0.5)
mkdata_fixed.plot(x='Newspapers',    y='sales', kind='scatter', ax=ax_np, alpha=0.5)
mkdata_fixed.plot(x='Surveys',       y='sales', kind='scatter', ax=ax_sv, alpha=0.5)
mkdata_fixed.plot(x='days_of_sales', y='sales', kind='scatter', ax=ax_dos, alpha=0.5)
plt.show()


# %% [markdown]
# На скаттерплотах видно, что количество опросов не оказывает влияния на продажи вовсе,
# расходы на рекламу по радио слабо коррелированы с продажами, неплохая корреляция у дней распродаж
# и телерекламы.  С газетами всё непросто, но похоже, что корреляция есть.  Оставил их в модели,
# а опросы убрал.


# %% [markdown]
# ### Моделирование <a name='model'/>

# %% [markdown]
# Использую многомерную линейную регрессию, которую предосталяет пакет `statmodels`, метод наименьших
# общих квадратов (OLS — Ordinary Least Squares).

# %%
model  = smf.ols('sales ~ TV + Radio + Newspapers + days_of_sales', data=mkdata_fixed).fit()
display(model.summary())

# %% [markdown]
# Что интересного говорят нам эти цифры? В первую очередь, посмотрим, насколько построенная регрессия
# объясняет разброс данных, это параметр R-squared (и Adjusted R-squared). Примерно на 70%, что неплохо.
# F-Statistic показывает, может ли изменение целевого параметра быть объяснено только интерсептом, то
# есть насколько вероятна гипотеза, что наши переменные вообще не влияют на результат. В данном
# случае видно, что это крайне маловероятно. Подробнее про F-Test можно прочесть
# [на medium.com](https://towardsdatascience.com/fisher-test-for-regression-analysis-1e1687867259)
#
# Далее следует блок коэффициентов. С каждой переменной связан коэффициент, который показывает её вклад
# в изменения целевой переменной. Специальная переменная `Intercept` показывает, какое _расчётное_ значение будет
# у `sales` в случае, когда все исходные переменные равны нулю.
#
# В данном случае видим, что наибольшее влияние на продажи оказывает количество дней распродаж, далее
# следует реклама на радио и телевидении. Влияние газетной рекламы мало, менее процента.
#
# Параметр `Durbin-Watson` показывает отсутствие автокорреляции в данных для модели.
#
# Перестрою модель, исключив газетную рекламу, и снова выведу её краткую характеристику.

# %%
model = smf.ols('sales ~ TV + Radio + days_of_sales', data=mkdata_fixed).fit()
display(model.summary())

# %% [markdown]
# Новая модель получилась лучше прежней, так как:
# - меньше переменных
# - Такой же коэффициент детерминации ($R^2$)
# - Увеличился параметр F-Statistic
# - Снизились стандартные отклонения коэффициентов, соответственно, сузились границы их доверительных интервалов
#

# %% [markdown]
# ### Оценка качества модели <a name='model_quality'/>

# %% [markdown]
# Качество нашей работы оценим по остаткам (residuals).
#
# Идеальные остатки должны быть невелики, независимы от исходных данных и распределены нормально.

# %% [markdown]
# #### Оценка разброса остатков <a name='modchk_disp'/>

# %% [markdown]
# Для этой оценки построю скаттерограмму предсказанного и реального значений `sales`.

# %%
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle("Предсказанные и реальные значения `sales'")
ax.set_xlabel("Расчётные значения")
ax.set_ylabel("Значения из данных")
ax.set_xlim(xmin=20,xmax=60)
ax.set_ylim(ymin=20,ymax=60)
plt.scatter(x=model.fittedvalues, y=mkdata_fixed.sales)
plt.show()

# %% [markdown]
# В идеале все точки должны быть на прямой, которая идёт под углом 45°.
# На практике видим группировку точек вокруг этой прямой. При больших
# значениях `sales` группа точек слегка отклоняется вверх (примерно на 5),
# т.&nbsp;е. на 10%.
#
# Также разброс можно оценить по гистограмме в [следующем параграфе](#modchk_norm)

# %% [markdown]
# #### Проверка нормальности распределения остатков <a name='modchk_norm'/>

# %% [markdown]
# На графике [QQ-Plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
# точки должны попасть на прямую, гистограмма остатков должна походить на «колокол»
# нормального распределения:

# %%
fig, (ax_qq, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))
ax_qq.set_title("QQ-Plot")
ax_hist.set_title("Гистограмма остатков")
qqplot(model.resid, line = 's', alpha=0.6, ax=ax_qq)
model.resid.plot(kind='hist',  ax=ax_hist)
plt.show()

# %% [markdown]
# Существует [web-приложение](https://xiongge.shinyapps.io/QQplots/), которое помогает интерпретировать
# QQ-Plot и строит его для различных искажений нормального распределения.

# %% [markdown]
# Распределение остатков от моделирования достаточно близко к нормальному,
# так что модель адекватна наблюдениям.
#
# Выполню ещё несколько дополнительных проверок. Я уже знаю, что ошибка респределена
# нормально, но есть ли её зависимость от исходных данных?
#

# %% [markdown]
# #### Проверка корреляций с исходными данными <a name='modchk_corr'/>

# %% [markdown]
# Рассчитаю корреляцию ошибки со всеми исходными данными в таблице `mkdata_fixed`.

# %%
display(mkdata_fixed.drop('sales', axis=1).corrwith(model.resid))


# %% [markdown]
# Все параметры, участвующие в модели, показывают полное отсутствие корреляции. Два
# оставшихся «за бортом» параметра, `Newspapers` и `Surveys`, показывают корреляцию от 2 до 5%,
# то есть пренебрежимо малую.


# %% [markdown]
# ### Итоги моделирования <a name='model_results'/>

# %% [markdown]
# Параметр `sales` в модели зависит от параметров `days_of_sales`, `Radio` и `TV` в
# соответствии с формулой:
#
# $$
# sales = 22 + 0.3 \times days\_of\_sales + 0.14 \times Radio + 0.055 \times TV.
# $$
#
# Влиянием остальных факторов из таблицы `mark_data` можно пренебречь.
# 
# Влияние маркетинговых расходов на продажи в процентах в соответствии с моделью:
#
# |             | Дни распродаж | Радио | Телевидение |
# |:------------|--------------:|------:|------------:|
# | Cреднее     | 30            | 14    |         5.5 |
# | От-До (95%) | 20-40         | 10-17 |         5-6 |

# %% [markdown]
# ## Выводы по исследованию <a name='conclusions'/>
# - Обнаружена линейная зависимость продаж от (по  порядку значимости)
# 
#   1. количества дней распродаж,
#   2. расходов на рекламу по радио,
#   3. расходов на рекламу по телевидению.
#   
# - Теоретически, если ничего не делать для повышения продаж, они будут на уровне 22.
#
# Наибольший вклад (в среднем 30%, доверительный интервал от 20% до 40%) вносят дни
# распродаж.  Приблизительно 14% вклад у рекламы по радио (интервал 10-17%) и замыкает
# тройку телевидение (около 5%).  Уравнение регрессии:
#
# $$
# sales = 22 + 0.3 \times days\_of\_sales + 0.14 \times Radio + 0.055 \times TV.
# $$
# 
# Эффекта рекламы в газетах обнаружить не удалось.  Исследования спроса (`Survey`)
# также не влияют на продажи видимым образом.
#
# Обращаю внимание, что в настоящий момент две трети рекламного бюджета
# уходит на телевидение, и из оставшейся трети больше половины тратится
# на рекламу в газетах.  Мне представляется, что это неэффективное расходование
# средств.
#
# ### Предложения:
#
# 1. Чаще устраивать распродажи, это самый эффективный способ поднятия продаж.
#
# 2. Перераспределить рекламный бюджет:
# 
#   - Уменьшить расходы на телевизионную рекламу и жёстко привязать её к дням перед
#     распродажами.
#
#   - Сократить расходы на газетную рекламу.
#
#   - Увеличить расходы на рекламу по радио.
#
#   - Начать рекламировать магазин в Интернет. Предшественник жил словно в каменном веке,
#     но мы не обязаны следовать его примеру.  Часть опросов тоже можно перенести в Сеть,
#     это снизит расходы на них.
#
# 3. Изменения пропорций рекламных носителей в бюджете выполнять постепенно (например, по 5% в месяц),
#    с контролем влияния изменений на продажи.  С учётом больших изменений продаж от месяца к месяцу,
#    эффект изменений может быть заметен не сразу.
#
# 4. Могу предположить, что поскольку радиостанций существенно больше, чем телевизионных программ,
#    увеличенный бюджет радио-рекламы нужно потратить на рост числа радиостанций, которые будут
#    транслировать рекламу, а не на увеличение насыщенности рекламой эфира тех радиостанций, где
#    она уже есть.
#    
