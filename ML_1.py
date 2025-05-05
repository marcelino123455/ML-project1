from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import h5py
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, kernel_type='linear', regularization_parameter=10000.0, max_iterations=500, polynomial_degree=3, gauss_gamma=1,sigmoid_alpha=1, sigmoid_c=0):

        self.kernel_type = kernel_type
        self.polynomial_degree = polynomial_degree
        self.gauss_gamma = gauss_gamma
        self.regularization_parameter = regularization_parameter
        self.max_iterations = max_iterations
        self.support_vector_indices = None
        #Kernell sgimoideo
        self.sigmoid_alpha = sigmoid_alpha
        self.sigmoid_c = sigmoid_c

    def _compute_kernel_matrix(self, X1, X2):

        # selección del kernel
        if self.kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel_type == 'poly':
            return np.dot(X1, X2.T)**self.polynomial_degree
        elif self.kernel_type == 'gauss':
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            dist_sq = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq
            return np.exp(-self.gauss_gamma * dist_sq)
        elif self.kernel_type == 'sigmoid':  # Caso para el kernel sigmoideo
          return np.tanh(self.sigmoid_alpha * np.dot(X1, X2.T) + self.sigmoid_c)
        else:
            raise ValueError(f"Kernel no soportado: {self.kernel_type}")

    def fit(self, feature_matrix, target_labels):
        self.features = feature_matrix.copy()
        self.labels = target_labels * 2 - 1  #etiquetas [0, 1] a [-1, 1]
        self.lagrange_multipliers = np.zeros_like(self.labels, dtype=float) # Estos son números que va a aprender para decidir qué puntos son vectores de soporte.

        kernel_values = self._compute_kernel_matrix(self.features, self.features) # k(x(i), x(j))


        # el "self.labels" y "self.labels[:, np.newaxis]" representa las clases reales de los datos de entrenamiento
        self.kernel_matrix = kernel_values * self.labels[:, np.newaxis] * self.labels #t(i)*t(j)*k(x(i)T, x(j))

        # optimización // para encontrar los multiplicadores de Lagrange
        for _ in range(self.max_iterations):
            for primary_index in range(len(self.lagrange_multipliers)):
                secondary_index = np.random.randint(0, len(self.lagrange_multipliers))

                quadratic_term = self.kernel_matrix[[[primary_index, primary_index], [secondary_index, secondary_index]],
                                                     [[primary_index, secondary_index], [primary_index, secondary_index]]]

                current_multipliers = self.lagrange_multipliers[[primary_index, secondary_index]]

                error_terms = 1 - np.sum(self.lagrange_multipliers * self.kernel_matrix[[primary_index, secondary_index]], axis=1)

                direction_vector = np.array([-self.labels[secondary_index], self.labels[primary_index]])

                optimal_step = np.dot(error_terms, direction_vector) / (np.dot(np.dot(quadratic_term, direction_vector), direction_vector) + 1E-15)

                constrained_step = self._restrict_to_valid_range(optimal_step, current_multipliers, direction_vector)

                self.lagrange_multipliers[[primary_index, secondary_index]] = current_multipliers + direction_vector * constrained_step #multiplicadores de lagrange optimizados

        # identificamos los vectores de soporte y calcular el sesgo

        self.support_vector_indices = np.nonzero(self.lagrange_multipliers > 1E-15)[0] # Identifica qué puntos de entrenamiento son vectores de soporte.

        if len(self.support_vector_indices) > 0: # calculo de sedsgo
            self.b = np.sum((1.0 - np.sum(self.kernel_matrix[self.support_vector_indices] *
                                            self.lagrange_multipliers, axis=1)) *
                               self.labels[self.support_vector_indices]) / len(self.support_vector_indices)
        else:
            self.b = 0.0

        if self.kernel_type == 'linear': # Cálculo del vector de pesos W (para kernel lineales)
            self.W = np.sum((self.lagrange_multipliers * self.labels)[:, np.newaxis] * self.features, axis=0)
        print("Sesgo (b):", self.b)
        print("Ejemplo de decision values:", np.sum(self._compute_kernel_matrix(X_val_scaled, self.features)[0] * self.labels * self.lagrange_multipliers) + self.b)
        return self


    def predict(self, X):
        #valores aplicando el kernel
        kernel_values = self._compute_kernel_matrix(X, self.features)

        #predicciones [1, -1]
        decision_values = np.sum(kernel_values * self.labels * self.lagrange_multipliers, axis=1) + self.b
        return (np.sign(decision_values) + 1) // 2  # conversion a [0,1]

    def _restrict_to_valid_range(self, step_size, current_values, direction_vector):
        #restringimos el paso de optimización para mantener los multiplicadores dentro de los límites permitidos

        step_size = (np.clip(current_values + step_size*direction_vector, 0, self.regularization_parameter) -
                    current_values)[1]/direction_vector[1]
        return (np.clip(current_values + step_size*direction_vector, 0, self.regularization_parameter) -
               current_values)[0]/direction_vector[0]

    def get_support_vectors(self):

        if self.support_vector_indices is None:
            raise ValueError("El modelo debe ser entrenado antes de obtener los vectores de soporte")

        return self.support_vector_indices
    



# Carga de datos
X = pd.read_pickle("all_features.pkl")
with h5py.File("train.h5", "r") as f:
    y_data = f['y'][:]
y = pd.Series(y_data, name="label")

# División del dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print("Distribución de clases en y_train:", Counter(y_train_bal))

plt.bar(['Clase 0', 'Clase 1'], Counter(y_train_bal).values())
plt.title('Distribución de clases tras SMOTE')
plt.show()
# Conversión a formato numpy (importante para compatibilidad)
X_train_bal = np.array(X_train_bal)
y_train_bal = np.array(y_train_bal)
X_val_scaled = np.array(X_val_scaled)
y_val = np.array(y_val)

# Inicialización y entrenamiento
svm = SVM(kernel_type='gauss', regularization_parameter=1.0, gauss_gamma=0.01)
svm.fit(X_train_bal, y_train_bal)
print("Multiplicadores de Lagrange (no cero):", np.sum(svm.lagrange_multipliers > 1e-5))
print("Soporte encontrados:", len(svm.get_support_vectors()))


decision_values = np.sum(svm._compute_kernel_matrix(X_val_scaled, svm.features) * svm.labels * svm.lagrange_multipliers, axis=1) + svm.b
print("Min:", decision_values.min(), "Max:", decision_values.max())
# Predicción y evaluación
y_pred = svm.predict(X_val_scaled)
print(" Resultados con SVM personalizado (Kernel Gaussiano/RBF):")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))
print("Predicciones:", y_pred[:20])
print("Reales:", y_val[:20])

for C in [0.1, 1, 10, 100]:
    for gamma in [0.01, 0.05, 0.1, 0.5, 1]:
        print(f"Probando C={C}, gamma={gamma}")
        svm = SVM(kernel_type='gauss', regularization_parameter=C, gauss_gamma=gamma)
        svm.fit(X_train_bal, y_train_bal)
        y_pred = svm.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_val, y_pred, zero_division=0))
        print("-" * 40)