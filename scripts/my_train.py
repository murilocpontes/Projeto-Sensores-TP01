"""
Script de treinamento de modelo SVM personalizado com dados próprios coletados pelo smartphone.

Unifica os arquivos CSV gerados para cada classe de atividade, particiona os dados
em conjuntos de treino e teste, ajusta o pipeline (StandardScaler + SVC) e exporta
o modelo resultante.
"""

import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# glob.glob / pd.concat: localiza todos os arquivos CSV de classes coletadas e consolida em um único DataFrame
arquivos = glob.glob("meus_dados_classe_*.csv")
df_completo = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

# Separa a matriz de atributos (X) do vetor de rótulos de classe (y)
X = df_completo.iloc[:, :-1].values
y = df_completo.iloc[:, -1].values

# train_test_split: divide o conjunto de dados em 80% para treinamento e 20% para validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: define a sequência de padronização Z-score seguida pelo classificador SVM com kernel RBF
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel="rbf", C=2.0, gamma="scale"))
])

# pipeline.fit: treina o normalizador e o classificador SVM utilizando o subconjunto de treinamento
pipeline.fit(X_train, y_train)

# accuracy_score: avalia a taxa de acerto no subconjunto de teste
print("Acurácia no seu celular:", accuracy_score(y_test, pipeline.predict(X_test)))
# joblib.dump: persiste o pipeline treinado em arquivo binário para uso no servidor em tempo real
joblib.dump(pipeline, "pipeline_svm_meu_celular.pkl")