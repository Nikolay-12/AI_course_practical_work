from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def calculate_descriptors(mol):
    descriptors = [
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
print(len(mols)) # Количество молекул

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

print(f"Количество отфильтрованных молекул (есть свойство Soluble): {len(clean_mols)}")
print(f"Количество активных молекул: {len(soluble_mols)}")


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
print("Количество дескрипторов:", X.shape[1])


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


############ Оценка успешности предсказаний разными метриками
from sklearn.metrics import (
    accuracy_score, # Долю правильных предсказаний
    precision_score, # Какая доля предсказанных активных действительно активна
    recall_score, # Какую долю всех активных молекул нашла модель
    f1_score, # Баланс Precision и Recall
    precision_recall_fscore_support,
    roc_auc_score, # Способность модели различать два класса
    classification_report
)

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

results = []

for name, model in models.items():
    model.fit(X_train, y_train) # Обучение
    prediction = model.predict(X_test) # Предсказания
    probabilities = model.predict_proba(X_test)[:, 1] # Вероятности классов

    # Метрики отдельно для каждого класса
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        prediction,
        labels=[0, 1],
        zero_division=0
    )

    # Метрики
    accuracy = accuracy_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, probabilities)

    # Добавляем результаты
    results.append({
        "Модель": name,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,

        "Precision (нераст.)": precision[0],
        "Recall (нераст.)": recall[0],
        "F1 (нераст.)": f1[0],

        "Precision (раст.)": precision[1],
        "Recall (раст.)": recall[1],
        "F1 (раст.)": f1[1]
    })

# Создаём таблицу
results_df = pd.DataFrame(results)

# Округляем до трёх знаков
results_df = results_df.round(3)

# Выводим
print(results_df.to_string(index=False))


############ Построение матрицы ошибок и ROC-кривая для наиболее успешной классификационной модели
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

prediction = models["Random Forest"].predict(X_test)

cm = confusion_matrix(y_test, prediction)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Растворима", "Нерастворима"],
    yticklabels=["Растворима", "Нерастворима"]
)

plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок — Random Forest")
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

probabilities = models["Random Forest"].predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

auc = roc_auc_score(y_test, probabilities)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.legend()
plt.grid()
plt.show()
