import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def determina_n_v(theta_f, deltaN, theta_i, d, lambda_, errori_theta_f, errore_theta_i, errori_deltaN, titolo="Determinazione di n_v", xlabel="Theta_f", ylabel="DeltaN"):
    """
    Determina il valore di n_v attraverso l'interpolazione:
    deltaN = 2d(1-cos(theta))(n_v-1) / (lambda * (cos(theta) + n_v - 1)).
    Mostra il grafico con il valore di n_v, il suo errore, chi-quadro e p-value.

    :param theta_f: Lista dei valori di theta_f (in radianti).
    :param deltaN: Lista dei valori di DeltaN.
    :param theta_i: Valore noto di theta_i (in radianti).
    :param d: Spessore noto (in unità coerenti con lambda_).
    :param lambda_: Lunghezza d'onda nota.
    :param errori_theta_f: Lista degli errori associati a theta_f.
    :param errore_theta_i: Errore associato a theta_i.
    :param errori_deltaN: Lista degli errori associati a DeltaN.
    :param titolo: Titolo del grafico (opzionale).
    :param xlabel: Etichetta dell'asse x (opzionale).
    :param ylabel: Etichetta dell'asse y (opzionale).
    """
    # Calcola theta = theta_f - theta_i e i relativi errori
    theta = np.array(theta_f) - theta_i
    errore_theta = np.sqrt(np.array(errori_theta_f)**2 + errore_theta_i**2)

    # Funzione per il fit non lineare
    def modello(theta, n_v):
        cos_theta = np.cos(theta)
        return (2 * d * (1 - cos_theta) * (n_v - 1)) / (lambda_ * (cos_theta + n_v - 1))

    # Stima iniziale di n_v
    n_v_iniziale = 1.0003

    # Fit non lineare con curve_fit
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(
        modello, theta, deltaN, p0=[n_v_iniziale], sigma=errori_deltaN, absolute_sigma=True
    )
    n_v = popt[0]
    errore_n_v = np.sqrt(pcov[0, 0])

    # Calcolo del chi-quadro
    deltaN_fit = modello(theta, n_v)
    chi_quadro = np.sum(((deltaN - deltaN_fit) / errori_deltaN) ** 2)
    dof = len(theta) - len(popt)  # Gradi di libertà
    p_value = 1 - chi2.cdf(chi_quadro, dof)

    # Disegna il grafico
    plt.errorbar(theta_f, deltaN, xerr=errori_theta_f, yerr=errori_deltaN, fmt='o', ecolor='red', capsize=5, label="Dati sperimentali")
    theta_fit = np.linspace(min(theta_f), max(theta_f), 500)
    deltaN_fit_plot = modello(theta_fit - theta_i, n_v)
    plt.plot(theta_fit, deltaN_fit_plot, label=f"Fit: n_v={n_v:.6f} ± {errore_n_v:.6f}")
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
    print(f"n_v: {n_v:.6f} ± {errore_n_v:.6f}")
    print(f"Chi-quadro: {chi_quadro:.2f}")
    print(f"Gradi di libertà: {dof}")
    print(f"P-value: {p_value:.2f}")

def main():
    # Dati
    theta_f = [0.01, 0.02, 0.03, 0.04, 0.05]  # Angoli finali in radianti
    deltaN = [0.5, 0.6, 0.7, 0.8, 0.9]  # DeltaN
    theta_i = 0.005  # Angolo iniziale in radianti
    d = 1e-3  # Spessore noto in metri
    lambda_ = 632.8e-9  # Lunghezza d'onda in metri

    # Errori
    errori_theta_f = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]  # Errori su theta_f
    errore_theta_i = 0.0001  # Errore su theta_i
    errori_deltaN = [0.01, 0.01, 0.01, 0.01, 0.01]  # Errori su DeltaN

    # Grafico
    determina_n_v(theta_f, deltaN, theta_i, d, lambda_, errori_theta_f, errore_theta_i, errori_deltaN)

if __name__ == "__main__":
    main()
