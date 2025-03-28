import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from mylib import stats

def determina_lambda(d, L, errore_L, N, d_inc, errore_d_inc, d_N, errori_d_N, titolo="Determinazione di lambda", xlabel="N", ylabel="cos(theta_inc) - cos(theta_N)"):
    """
    Determina il valore di lambda attraverso l'interpolazione:
    cos(theta_inc) - cos(theta_N) = lambda * N / d.
    Mostra il grafico con il valore di lambda, il suo errore, chi-quadro e p-value.

    :param d: Valore noto di d (in unità coerenti con L).
    :param L: Lunghezza nota (in unità coerenti con d_inc e d_N).
    :param errore_L: Errore associato a L.
    :param N: Lista dei valori di N.
    :param d_inc: Distanza incidente nota (in unità coerenti con L).
    :param errore_d_inc: Errore associato a d_inc.
    :param d_N: Lista delle distanze d_N (in unità coerenti con L).
    :param errori_d_N: Lista degli errori associati a d_N.
    :param titolo: Titolo del grafico (opzionale).
    :param xlabel: Etichetta dell'asse x (opzionale).
    :param ylabel: Etichetta dell'asse y (opzionale).
    """
    # Calcola cos(theta_inc) e il suo errore
    cos_theta_inc = L / np.sqrt(L**2 + d_inc**2)
    errore_cos_theta_inc = np.abs(cos_theta_inc) * np.sqrt(
        (errore_L / L)**2 + (d_inc * errore_d_inc / (L**2 + d_inc**2))**2
    )

    # Calcola cos(theta_N) e i relativi errori
    cos_theta_N = L / np.sqrt(L**2 + np.array(d_N)**2)
    errore_cos_theta_N = np.abs(cos_theta_N) * np.sqrt(
        (errore_L / L)**2 + (np.array(d_N) * np.array(errori_d_N) / (L**2 + np.array(d_N)**2))**2
    )

    # Calcola cos(theta_inc) - cos(theta_N) e i relativi errori
    y = cos_theta_inc - cos_theta_N
    errore_y = np.sqrt(errore_cos_theta_inc**2 + errore_cos_theta_N**2)

    # Funzione per il fit lineare
    def modello(N, lambda_):
        N = np.array(N)
        return lambda_ * N / d

    # Stima iniziale di lambda
    lambda_iniziale = 632.8e-9  # Lunghezza d'onda iniziale in metri

    # Fit non lineare con curve_fit
    popt, pcov = curve_fit(
        modello, N, y, p0=[lambda_iniziale], sigma=errore_y, absolute_sigma=True
    )
    lambda_ = popt[0]
    errore_lambda = np.sqrt(pcov[0, 0])

    # Calcolo del chi-quadro
    y_fit = modello(N, lambda_)
    chi_quadro = np.sum(((y - y_fit) / errore_y) ** 2)
    dof = len(N) - len(popt)  # Gradi di libertà
    p_value = 1 - chi2.cdf(chi_quadro, dof)

    # Disegna il grafico
    plt.errorbar(N, y, yerr=errore_y, fmt='o', ecolor='red', capsize=5, label="Dati sperimentali")
    N_fit = np.linspace(min(N), max(N), 500)
    y_fit_plot = modello(N_fit, lambda_)
    plt.plot(N_fit, y_fit_plot, label=f"Fit: lambda={lambda_:.2e} ± {errore_lambda:.2e}")
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
    print(f"lambda: {lambda_:.2e} ± {errore_lambda:.2e}")
    print(f"Chi-quadro: {chi_quadro:.2f}")
    print(f"Gradi di libertà: {dof}")
    print(f"P-value: {p_value:.2f}")

def main():
    # Dati
    d = 1e-3  # Spessore noto in metri
    L = 1.0  # Lunghezza nota in metri
    errore_L = 0.001  # Errore su L in metri
    N = [1, 2, 3, 4, 5]  # Valori di N
    d_inc = 0.01  # Distanza incidente in metri
    errore_d_inc = 0.0001  # Errore su d_inc in metri
    d_N = []  # Distanze d_N in metri
    errori_d_N = []  # Errori associati a d_N
    d_N_1 = [0.02, 0.03, 0.04] 
    d_N.append(stats(d_N_1).mean())
    errori_d_N.append(stats(d_N_1).sigma_mean())
    d_N_2 = [0.05, 0.06, 0.07]
    d_N.append(stats(d_N_2).mean())
    errori_d_N.append(stats(d_N_2).sigma_mean())
    d_N_3 = [0.08, 0.09, 0.1]
    d_N.append(stats(d_N_3).mean())
    errori_d_N.append(stats(d_N_3).sigma_mean())
    d_N_4 = [0.11, 0.12, 0.13]
    d_N.append(stats(d_N_4).mean())
    errori_d_N.append(stats(d_N_4).sigma_mean())
    d_N_5 = [0.14, 0.15, 0.16]
    d_N.append(stats(d_N_5).mean())
    errori_d_N.append(stats(d_N_5).sigma_mean())


    # Grafico
    determina_lambda(d, L, errore_L, N, d_inc, errore_d_inc, d_N, errori_d_N)

if __name__ == "__main__":
    main()
