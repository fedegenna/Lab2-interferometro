import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from mylib import stats

def grafico_deltaN(delta_P, deltaN, d, lambda_, errori_deltaN, errori_delta_P, titolo="Indice di rifrazione dell'aria", xlabel="delta_P", ylabel="DeltaN"):
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
    x = np.array(delta_P)
    errore_x = np.ones(len(x))

    # DeltaN e i relativi errori
    y = np.array(deltaN)
    errore_y = np.array(errori_deltaP)

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
    delta_P = [5,8.5,11.5,14,18]
    deltaN = [4,6,7,9,11]  # DeltaN
    

    errori_deltaN = np.ones(len(delta_P))  # Errori su DeltaN
    
    
    
    d = 3  # Spessore noto, in cm
    lambda_ =632.8*pow(10,-7)   # Lunghezza d'onda nota,in cm

     # Grafico
    grafico_deltaN(P_i, P_f, deltaN, d, lambda_, errori_deltaN, errori_P_i, errori_P_f)

if __name__ == "__main__":
    main()
