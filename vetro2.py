import numpy as np
def indice_di_rifrazione(delta_N,lambda_,d,theta):
    return (((2*d-delta_N*lambda_)*(1-np.cos(theta)))/(2*d*(1-np.cos(theta))-delta_N*lambda_))


def errore_indice_di_rifrazione(delta_N,lambda_,d,theta,errore_delta_N,errore_theta,errore_lambda_ = 0,errore_d = 0.01*pow(10,-3)):
    return np.sqrt((lambda_/(2*d*(1-np.cos(theta))))**2*errore_delta_N**2+(delta_N*lambda_/(2*d*(1-np.cos(theta))**2))*errore_lambda_**2+(delta_N*lambda_/(2*d*(1-np.cos(theta))))**2*errore_d**2+(delta_N*lambda_/(2*d*(1-np.cos(theta))))**2*errore_theta**2)

def media_pesata(valori, errori):
    """
    Calcola la media pesata di una serie di valori con i relativi errori.

    Parametri:
    - valori: lista o array dei valori (es. indici di rifrazione)
    - errori: lista o array degli errori associati ai valori

    Ritorna:
    - media_pesata: la media pesata
    - errore_media_pesata: l'errore sulla media pesata
    """
    # Calcolo dei pesi (inverso del quadrato degli errori)
    pesi = 1 / np.array(errori)**2

    # Calcolo della media pesata
    media_pesata = np.sum(valori * pesi) / np.sum(pesi)

    # Calcolo dell'errore sulla media pesata
    errore_media_pesata = np.sqrt(1 / np.sum(pesi))

    return media_pesata, errore_media_pesata


def main():
    theta_f = [np.deg2rad(2.2), np.deg2rad(3.2),np.deg2rad(3.6), np.deg2rad(3.9),np.deg2rad(5.2),np.deg2rad(5.5)]  # Angoli finali radianti 
    deltaN = [4, 9, 7, 15, 24,21]  # DeltaN
    theta_i = 0  # Angolo iniziale in radianti
    d = 5.662*pow(10,-3)  # Spessore noto in metri,da vedere in Pasco
    lambda_ = 632.8e-9  # Lunghezza d'onda in metri

# Errori
    errori_theta_f = np.deg2rad(0.1)*np.ones(len(theta_f))  # Errori su theta_f
    errore_theta_i = 0  # Errore su theta_i
    errori_theta = errori_theta_f
    errori_deltaN = 0.2 * np.ones(len(deltaN)) # Errori su DeltaN

    indici_di_rifrazione = []
    errori_indici_di_rifrazione = []
    for i in range(len(theta_f)):
        indici_di_rifrazione.append(indice_di_rifrazione(deltaN[i],lambda_,d,theta_f[i]))
        errori_indici_di_rifrazione.append(errore_indice_di_rifrazione(deltaN[i],lambda_,d,theta_f[i],errori_deltaN[i],errori_theta[i]))
    print("Indici di rifrazione:", indici_di_rifrazione)
    print("Errori sugli indici di rifrazione:", errori_indici_di_rifrazione)
    #calcolo della media e dello scarto quadratico medio
    n_vetro = media_pesata(indici_di_rifrazione,errori_indici_di_rifrazione)
    print(n_vetro)
if __name__ == "__main__":
    main()
