import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mylib import stats

def grafico_deltaN(P_i, P_f, deltaN, d, lambda_, errori_deltaN, errori_P_i, errori_P_f, titolo="Indice di rifrazione dell'aria", xlabel="(P_i - P_f)", ylabel="DeltaN"):
    """
    Disegna un grafico con DeltaN in ordinata e (P_i - P_f) in ascissa.
    Calcola il valore di m con il suo errore, chi-quadro e p-value, e li stampa sul grafico.

    :param P_i: Lista dei valori iniziali di pressione.
    :param P_f: Lista dei valori finali di pressione.
    :param deltaN: Lista dei valori di DeltaN.
    :param d: Spessore noto (in unità coerenti con lambda_).
    :param lambda_: Lunghezza d'onda nota.
    :param errori_deltaN: Lista degli errori associati a DeltaN.
    :param errori_P_i: Lista degli errori associati a P_i.
    :param errori_P_f: Lista degli errori associati a P_f.
    :param titolo: Titolo del grafico (opzionale).
    :param xlabel: Etichetta dell'asse x (opzionale).
    :param ylabel: Etichetta dell'asse y (opzionale).
    """
    # Calcola (P_i - P_f) e i relativi errori
    x = np.array(P_i) - np.array(P_f)
    errore_x = np.sqrt(np.array(errori_P_i)**2 + np.array(errori_P_f)**2)

    # DeltaN e i relativi errori
    y = np.array(deltaN)
    errore_y = np.array(errori_deltaN)

    # Normalizza x e y per includere d e lambda_
    x_normalizzato = x * 2 * d / lambda_
    errore_x_normalizzato = errore_x * 2 * d / lambda_

    # Pesi per l'interpolazione (inversamente proporzionali agli errori al quadrato)
    pesi = 1 / errore_y**2

    # Interpolazione lineare per trovare m
    A = np.vstack([x_normalizzato, np.ones(len(x_normalizzato))]).T
    W = np.diag(pesi)  # Matrice dei pesi
    cov_matrix = np.linalg.inv(A.T @ W @ A)  # Matrice di covarianza calcolata con i pesi
    m, q = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)  # Risoluzione del sistema lineare
    errore_m, errore_q = np.sqrt(np.diag(cov_matrix))  # Errori su m e q

    # Calcolo del chi-quadro
    y_fit = m * x_normalizzato + q
    chi_quadro = np.sum(((y - y_fit) / errore_y)**2)
    dof = len(x) - 2  # Gradi di libertà
    p_value = 1 - chi2.cdf(chi_quadro, dof)

    # Disegna il grafico
    plt.errorbar(x, y, xerr=errore_x, yerr=errore_y, fmt='o', ecolor='red', capsize=5, label="Dati sperimentali")
    plt.plot(x, y_fit, label=f"Fit lineare: m={m:.2e} ± {errore_m:.2e}")
    plt.title(titolo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # Stampa i risultati sul grafico
    plt.text(0.05, 0.95, f"$\chi^2$={chi_quadro:.2f}\nDOF={dof}\np-value={p_value:.2f}", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()

    # Stampa i risultati in console
    print(f"m: {m:.2e} ± {errore_m:.2e}")
    print(f"Intercetta q: {q:.2e} ± {errore_q:.2e}")
    print(f"Chi-quadro: {chi_quadro:.2f}")
    print(f"Gradi di libertà: {dof}")
    print(f"P-value: {p_value:.2f}")

def main():
    # Dati
    P_i = []  # Pressione iniziale
    P_f = []  # Pressione finale
    P_i_1 = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione iniziale
    P_i.append(stats(P_i_1).mean())
    P_i_2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione iniziale
    P_i.append(stats(P_i_2).mean())
    P_i_3 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione iniziale
    P_i.append(stats(P_i_3).mean())
    P_i_4 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione iniziale
    P_i.append(stats(P_i_4).mean())
    P_i_5 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione iniziale
    P_i.append(stats(P_i_5).mean())
    P_f_1 = [0.6, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione finale
    P_f.append(stats(P_f_1).mean())
    P_f_2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione finale
    P_f.append(stats(P_f_2).mean())
    P_f_3 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione finale
    P_f.append(stats(P_f_3).mean())
    P_f_4 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione finale
    P_f.append(stats(P_f_4).mean())
    P_f_5 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pressione finale
    P_f.append(stats(P_f_5).mean())

    errori_P_i = []  # Errori su P_i
    errori_P_i.append(stats(P_i_1).sigma_mean())
    errori_P_i.append(stats(P_i_2).sigma_mean())
    errori_P_i.append(stats(P_i_3).sigma_mean())
    errori_P_i.append(stats(P_i_4).sigma_mean())
    errori_P_i.append(stats(P_i_5).sigma_mean())
    errori_P_f = []  # Errori su P_f
    errori_P_f.append(stats(P_f_1).sigma_mean())
    errori_P_f.append(stats(P_f_2).sigma_mean())
    errori_P_f.append(stats(P_f_3).sigma_mean())
    errori_P_f.append(stats(P_f_4).sigma_mean())
    errori_P_f.append(stats(P_f_5).sigma_mean())

    deltaN = []  # DeltaN
    deltaN_1 = [0.7, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltaN.append(stats(deltaN_1).mean())
    deltaN_2 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltaN.append(stats(deltaN_2).mean())
    deltaN_3 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltaN.append(stats(deltaN_3).mean())
    deltaN_4 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltaN.append(stats(deltaN_4).mean())
    deltaN_5 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    deltaN.append(stats(deltaN_5).mean())

    errori_deltaN = []  # Errori su DeltaN
    errori_deltaN.append(stats(deltaN_1).sigma_mean())
    errori_deltaN.append(stats(deltaN_2).sigma_mean())
    errori_deltaN.append(stats(deltaN_3).sigma_mean())
    errori_deltaN.append(stats(deltaN_4).sigma_mean())
    errori_deltaN.append(stats(deltaN_5).sigma_mean())

    
    
    d = 1  # Spessore noto
    lambda_ = 1  # Lunghezza d'onda nota

     # Grafico
    grafico_deltaN(P_i, P_f, deltaN, d, lambda_, errori_deltaN, errori_P_i, errori_P_f)

if __name__ == "__main__":
    main()
