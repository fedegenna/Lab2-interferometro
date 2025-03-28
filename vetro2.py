def indice_di_rifrazione(delta_N,lambda_,d,theta)
  return (((2*d-delta_N*lambda_)*(1-np.cos(theta)))/(2*d*(1-np.cos(theta))-delta_N*lambda_))


def main():
  theta_f = [np.deg2rad(2.2), np.deg2rad(3.2),np.deg2rad(3.6), np.deg2rad(3.9),np.deg2rad(5.2),np.deg2rad(5.5)]  # Angoli finali radianti 
  deltaN = [4, 9, 7, 15, 24,21]  # DeltaN
  theta_i = 0  # Angolo iniziale in radianti
  d = 5.662*pow(10,-3)  # Spessore noto in metri,da vedere in Pasco
  lambda_ = 632.8e-9  # Lunghezza d'onda in metri

  # Errori
    errori_theta_f = np.deg2rad(0.1)*np.ones(len(theta_f))  # Errori su theta_f
    errore_theta_i = 0  # Errore su theta_i
    errori_deltaN = [0.01, 0.01, 0.01, 0.01, 0.01]  # Errori su DeltaN, da capire 
