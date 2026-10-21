import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dcor
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

# 1. и 2. Открытие файла с данными и его очистка
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('dataset.csv', index_col=0)
df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')], errors='ignore')
print(df.head())
df = df.drop(index=11, errors='ignore')     # убираем атомарный углерод
numeric_cols = [
    'Molecular weight', 'Normal boiling point', 'Melting point',
    'Critical temperature', 'Critical pressure', 'Critical volume',
     'logP', 'Molecular refractivity'
]
df_num = df[numeric_cols].copy()
df_num.loc[54, 'Critical temperature'] = np.nan     # Обнуляем ошибочную Tc у нафталина

"""
# Построение любого графика (в виде точек) по двум колонкам
x_col = 0
y_col = 1

x_name = df_num.columns[x_col]
y_name = df_num.columns[y_col]

plt.figure(figsize=(10, 6))
plt.scatter(df_num[df_num.columns[x_col]], df_num[df_num.columns[y_col]],
            label=f'{y_name} от {x_name}',
            s=20,           # размер точек
            alpha=0.7,      # прозрачность (0–1)
            edgecolors='black',  # обводка точек
            linewidths=0.5)
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title(f'Зависимость {y_name} от {x_name}')
plt.grid(True)
plt.legend()
plt.show()
"""

"""
# 3. Построение гистограмм
df_num.hist(bins=30, figsize=(15, 12), edgecolor='black', alpha=0.7)
plt.suptitle('Распределение характеристик', y=1.02)
plt.tight_layout()
plt.show()
"""

"""
# 4. Корреляции
corr = df_num.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Корреляционная матрица по Пирсону')
plt.tight_layout()
plt.show()

corr_spearman = df_num.corr(method='spearman', numeric_only=True)
corr_kendall  = df_num.corr(method='kendall',  numeric_only=True)
# Сравнение трёх матриц
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, c) in zip(axes, [('Pearson', corr),
                                ('Spearman', corr_spearman),
                                ('Kendall', corr_kendall)]):
    sns.heatmap(c, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar=False, vmin=-1, vmax=1)
    ax.set_title(name)
plt.tight_layout()
plt.show()

def distance_corr_matrix(df):
    cols = df.columns
    n = len(cols)
    mat = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    for i in range(n):
        for j in range(i+1, n):
            x = df.iloc[:, i].values
            y = df.iloc[:, j].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() > 2:
                d = dcor.distance_correlation(x[mask], y[mask])
                mat.iloc[i, j] = mat.iloc[j, i] = d
    return mat

dcor_mat = distance_corr_matrix(df_num)

plt.figure(figsize=(10, 8))
sns.heatmap(dcor_mat, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, vmin=0, vmax=1)
plt.title('Distance correlation (ловит нелинейные связи)')
plt.tight_layout()
plt.show()
"""

"""
def plot_with_fit(df, x_col, y_col, expected_r=None):
    # Строит график в виде точек и делает линейную аппроксимацию.
    # expected_r — ожидаемый коэффициент корреляции Пирсона (для сравнения).
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].values
    y = data[y_col].values

    if len(x) < 2:
        print(f'Недостаточно данных для {x_col} vs {y_col}')
        return None

    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    y_pred = poly(x)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot # R²

    r_p = pearsonr(x, y)[0] # Корреляция Пирсона
    r_s = spearmanr(x, y)[0] # Корреляция Спирмена

    plt.figure(figsize=(8, 6)) # Сам график
    plt.scatter(x, y, alpha=0.7, s=30,
                edgecolors='black', linewidths=0.4, label='Данные')

    x_sorted = np.sort(x)
    plt.plot(x_sorted, poly(x_sorted), 'r--', linewidth=1.5,
             label=f'y = {coef[0]:.4g}·x + {coef[1]:.4g}\n'
                   f'R² = {r2:.3f}')

    plt.xlabel(x_col)
    plt.ylabel(y_col)

    title = f'{y_col} vs {x_col}'
    if expected_r is not None:
        title += f'  (ожид. Пирсон = {expected_r})'
    plt.title(title)

    # Врезка с корреляциями
    plt.text(0.05, 0.95,
             f'Пирсон  r = {r_p:.3f}\n'
             
             f'N = {len(x)}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {'X': x_col, 'Y': y_col,
            'Наклон': coef[0], 'Своб. член': coef[1],
            'R2': r2, 'Пирсон': r_p, 'Спирмен': r_s, 'N': len(x)}

# Список зависимостей (по Пирсону) с r > 0.7
pairs = [
    ('Molecular weight',        'logP',                   0.81),
    ('Normal boiling point',    'Melting point',          0.80),
    ('Normal boiling point',    'Critical temperature',   0.99),
    ('Normal boiling point',    'Critical volume',        0.85),
    ('Normal boiling point',    'Molecular refractivity', 0.90),
    ('Melting point',           'Critical temperature',   0.79),
    ('Melting point',           'Molecular refractivity', 0.71),
    ('Critical temperature',    'Critical volume',        0.80),
    ('Critical temperature',    'Molecular refractivity', 0.88),
    ('Critical volume',         'logP',                   0.76),
    ('Critical volume',         'Molecular refractivity', 0.97),
    ('logP',                    'Molecular refractivity', 0.75),
]

# Построение график, для которых ожидалась зависимость (линейная)
results = []
for x_col, y_col, expected in pairs:
    res = plot_with_fit(df_num, x_col, y_col, expected_r=expected)
    if res:
        results.append(res)
        
# Сводная таблица
summary = pd.DataFrame(results)
summary = summary[['X', 'Y', 'Наклон', 'Своб. член', 'R2', 'Пирсон', 'Спирмен', 'N']]
summary['Δ (Спирмен − Пирсон)'] = summary['Спирмен'] - summary['Пирсон']
print(summary.to_string(index=False))
"""

"""
# Была попытка выделить гиперболическую связь Tc и Pc, но безуспешно
x = df_num['Critical temperature'].values
y = df_num['Critical pressure'].values
mask = ~np.isnan(x) & ~np.isnan(y)
x, y = x[mask], y[mask]

print(f'Пирсон:  {pearsonr(x, y)[0]:.3f}')
print(f'Спирмен: {spearmanr(x, y)[0]:.3f}')
print(f'dCor:    {dcor.distance_correlation(x, y):.3f}')

def hyperbola(T, a, b): return a / T + b
def linear(T, a, b): return a * T + b

for name, f in [('Hyperbola', hyperbola), ('Linear', linear)]:
    popt, _ = curve_fit(f, x, y, maxfev=10000)
    y_pred = f(x, *popt)
    r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-y.mean())**2)
    print(f'{name}: R² = {r2:.3f}')

popt, _ = curve_fit(hyperbola, x, y, p0=[1e9, 0], maxfev=10000)
x_sorted = np.sort(x)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, s=25, edgecolors='black', linewidths=0.4, label='Данные')
plt.plot(x_sorted, hyperbola(x_sorted, *popt), 'r--',
         label=f'Pc = {popt[0]:.2e}/Tc + {popt[1]:.2e}')
plt.xlabel('Critical temperature, K')
plt.ylabel('Critical pressure, Pa')
plt.title('Гиперболическая связь Pc(Tc)')
plt.legend(); plt.grid(True); plt.show()
"""
