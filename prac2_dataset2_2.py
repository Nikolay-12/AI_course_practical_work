from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def calculate_descriptors(mol):
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        Lipinski.HeavyAtomCount(mol),
        rdMolDescriptors.CalcFractionCSP3(mol)
    ]
    return descriptors

########### Загрузка датасета и его очистка
mols = Chem.SDMolSupplier("logS_logP_dataset.sdf") # На выходе array-like объект с молекулами в виде Mol объектов из rdkit

counter = 0
solubilities = [] # Массив активностей
clean_mols = [] # Массив всех молекул, для которых удалось корректно распарсить молекулу
soluble_mols = [] # Массив молекул, которые активны
for mol in mols:
    counter += 1
    try:
        solubilities.append(mol.GetProp("Soluble"))
        clean_mols.append(mol)
        if mol.GetProp("Soluble") == "true":
            soluble_mols.append(mol)
    except Exception:
        print(f"Номер молекулы с ошибкой: {counter}")
        continue


########### Расчет дескрипторов для классификационной модели
X = []
y = []

for i, mol in enumerate(clean_mols):
    try:
        descriptors = calculate_descriptors(mol)

        if not np.all(np.isfinite(descriptors)):
            print(f"Некорректные дескрипторы у молекулы №{i}")
            continue

        X.append(descriptors)
        if mol.GetProp("Soluble") == "true":
            y.append(1)
        else:
            y.append(0)

    except Exception as e:
        print(f"Ошибка при обработке молекулы №{i}: {e}")
        continue

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)


############ Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42, # Фиксирует случайное распределение (чтоб переобучение не происходило)
    stratify=y # Сохраняет пропорции активность/не активность в выборках
)


############ Масштабирование данных
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train) # Подбор среднего значения и стандартного отклонения на обучающей выборке и
                                        # применение к обучающей выборке
X_test = scaler.transform(X_test) # Применение уже готовых средних значений и стандартного отклонения для тестовой
                                  # выборки (предполагается, что они должны для нее подходить)


############ Обучение классификационных моделей
from sklearn.naive_bayes import GaussianNB # Бейесовский классификатор (нормальное распределение признака внутри каждого класса)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

models = {
    "GaussianNB": make_pipeline(
        StandardScaler(),
        GaussianNB()
    ),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            C=1,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
}


############ Подбор гиперпараметров для каждого алгоритма
param_grids = {
    "GaussianNB": {
        "gaussiannb__var_smoothing": [
            1e-9, 1e-8, 1e-7, 1e-6
        ]
    },

    "Logistic Regression": {
        "logisticregression__C": [
            0.01, 0.1, 1, 10
        ]
    },

    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2]
    },

    "SVM": {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1]
    }
}


############ Подбор оптимальных параметров работы алгоритмов с использованием специального инструмента GridSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# 5 частей для кросс-валидации
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Дополнительная метрика для нерастворимых молекул (класс 0)
f1_insoluble = make_scorer(
    f1_score,
    pos_label=0
)

best_models = {} # лучшие модели
tuning_results = []

for name, model in models.items():

    print(f"\nПодбор параметров: {name}")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring={
            "f1_macro": "f1_macro",
            "f1_insoluble": f1_insoluble,
            "roc_auc": "roc_auc"
        },
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train) # Подбор параметров на обучающей выборке

    best_models[name] = grid_search.best_estimator_

    # Записываем результаты
    tuning_results.append({
        "Модель": name,
        "Лучшие параметры": grid_search.best_params_,
        "F1-macro (CV)": grid_search.best_score_,
        "F1 нерастворимых (CV)": grid_search.cv_results_[
            "mean_test_f1_insoluble"
        ][grid_search.best_index_],
        "ROC-AUC (CV)": grid_search.cv_results_[
            "mean_test_roc_auc"
        ][grid_search.best_index_]
    })

    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучший F1-macro:", round(grid_search.best_score_, 3))


import pandas as pd

tuning_df = pd.DataFrame(tuning_results)

print("\nРезультаты подбора гиперпараметров:")
print(tuning_df.round(3).to_string(index=False))


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

final_results = []

for name, model in best_models.items():

    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        prediction,
        labels=[0, 1],
        zero_division=0
    )

    final_results.append({
        "Модель": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "ROC-AUC": roc_auc_score(y_test, probabilities),

        "Precision (нераст.)": precision[0],
        "Recall (нераст.)": recall[0],
        "F1 (нераст.)": f1[0],

        "Precision (раст.)": precision[1],
        "Recall (раст.)": recall[1],
        "F1 (раст.)": f1[1],

        "F1-macro": (f1[0] + f1[1]) / 2
    })

final_df = pd.DataFrame(final_results)

print("\nИтоговое сравнение моделей на тестовой выборке:")
print(final_df.round(3).to_string(index=False))