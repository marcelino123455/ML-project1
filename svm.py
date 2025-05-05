import numpy as np
import pandas as pd

class SVM:
    def __init__(self, kernel_type='linear', regularization_parameter=10000.0, max_iterations=500, polynomial_degree=3, gauss_gamma=1,sigmoid_alpha=0.001, sigmoid_c=0):

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
            if X1 is X2:
                pairwise_distances = np.sum((X1[:, np.newaxis] - X2)**2, axis=2)
            else:
                pairwise_distances = np.sum((X2 - X1[:, np.newaxis])**2, axis=2)
            return np.exp(-self.gauss_gamma * pairwise_distances)
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

