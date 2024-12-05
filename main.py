import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


# Dados simulados
np.random.seed(42)
num_samples = 1000

data = {
    'kilometragem': np.random.randint(1000, 200000, num_samples),
    'idade_veiculo': np.random.randint(1, 20, num_samples),
    'temperatura_motor': np.random.uniform(70, 120, num_samples),
    'vibracao': np.random.uniform(0.1, 5.0, num_samples),
    'nivel_oleo': np.random.uniform(0.5, 1.0, num_samples),
    'necessita_manutencao': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]), # Classificação
    'tempo_ate_manutencao': np.random.uniform(0, 12, num_samples) # Regressão
}

df = pd.DataFrame(data)

# Ajustar a regressão para casos onde não é necessária manutenção
df.loc[df['necessita_manutencao'] == 0, 'tempo_ate_manutencao'] = 0

df.head()



[ ]
# Separar features e labels
X = df[['kilometragem', 'idade_veiculo', 'temperatura_motor', 'vibracao', 'nivel_oleo']]
y_class = df['necessita_manutencao']
y_reg = df['tempo_ate_manutencao']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)



[ ]
# Classificação - prever se necessita manutenção
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_class_train)

# Prever no conjunto de teste
y_class_pred = clf.predict(X_test)

# Avaliar modelo de classificação
print("Relatório de Classificação:")
print(classification_report(y_class_test, y_class_pred))
print("Acurácia:", accuracy_score(y_class_test, y_class_pred))



[ ]
# Regressão - prever o tempo até a manutenção
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_reg_train)

# Prever no conjunto de teste
y_reg_pred = reg.predict(X_test)

# Avaliar modelo de regressão
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")


[ ]
# Importância das features (classificação)
feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(8, 5), title="Importância das Features")
plt.show()

# Comparação das previsões (regressão)
plt.figure(figsize=(10, 6))
plt.plot(y_reg_test.values, label="Valores Reais")
plt.plot(y_reg_pred, label="Previsões", alpha=0.7)
plt.title("Comparação de Previsões de Tempo até a Manutenção")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(y_class_test, y_class_pred)

# Visualizar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusão - Classificação")
plt.show()


