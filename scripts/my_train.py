import pandas as pd
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Junta todos os arquivos CSV criados
arquivos = glob.glob("meus_dados_classe_*.csv")
df_completo = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

# 2. Separa as Features (X) do Rótulo (y)
X = df_completo.iloc[:, :-1].values
y = df_completo.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treina o Pipeline Personalizado
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel="rbf", C=2.0, gamma="scale"))
])

pipeline.fit(X_train, y_train)

# 4. Avalia e Salva
print("Acurácia no seu celular:", accuracy_score(y_test, pipeline.predict(X_test)))
joblib.dump(pipeline, "pipeline_svm_meu_celular.pkl")