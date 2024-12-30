import numpy as np
import pandas as pd

# Pour les algorithmes d'optimisation minimale
from scipy.optimize import differential_evolution,minimize 
import pyfolio as pf  # Pour créer des feuilles de calcul de rendement et analyser les performances des stratégies de trading.
import quantstats as qs  # Pour les analyses de performance et de risque des séries temporelles financières.



def sharpe(rendements,rendement_taux_sans_risque):
    rendements_ajustes = np.where(rendements + 1 <= 0, 1e-10, rendements)
    #rendements_ajustes=rendements
    écart_type=rendements_ajustes.std()*np.sqrt(252)  
    rendement_annuel = (np.prod(rendements_ajustes + 1))**(252 / len(rendements_ajustes)) - 1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  

def sortino(rendements,rendement_taux_sans_risque):
    #rendements[rendements<0] crée un sous-ensemble du DataFrame qui ne contient que les valeurs négatives (les rendements à la baisse).
    val_négative=rendements[rendements<0]
    # Calculer l'écart-type des rendements négatifs, ce qui représente la volatilité du risque à la baisse.
    écart_type = val_négative.std() * np.sqrt(252)
    # np.prod(rendements+1) calcule le produit cumulatif des rendements (ajustés de 1 pour chaque période),ce qui donne la croissance totale du portefeuille sur la période.
    rendement_annuel = np.prod(rendements+1)**(252/len(rendements))-1
    return ( rendement_annuel - rendement_taux_sans_risque )/ écart_type  

def calmar(rendements,rendement_taux_sans_risque):
    val_portefeuille=np.cumprod(rendements+1) #Calcule la valeur cumulée du portefeuille en supposant que tous les rendements sont réinvestis.
    if np.any(rendements + 1 <= 0):
        raise ValueError("Un ou plusieurs rendements rendent les valeurs invalides (rendements + 1 <= 0).")
    peak=val_portefeuille.expanding().max()  #Détermine la valeur maximale du portefeuille jusqu'à chaque point dans le temps, ce qui aide à identifier les pics avant les drawdowns.
    rendement_annuel = np.prod(rendements+1)**(252/len(rendements))-1
    drawdown=(val_portefeuille-peak)/peak   #Calcule le drawdown comme la baisse relative depuis le dernier pic.
    return ( rendement_annuel - rendement_taux_sans_risque )/-np.min(drawdown)


def scipy_func(x,arguments):   # technique d'optimisation la plus rapide mais la moins fiable  
    Poids=x.reshape(-1,1)     # Reformate le vecteur des poids x en une matrice colonne pour faciliter les opérations matricielles.
    ratio, rendement_taux_sans_risque, rendements = arguments   # Décompose le tuple arguments pour extraire le ratio de performance, le taux de rendement sans risque et les rendements historiques des actifs.
    val_portefeuille_rendements=pd.Series(np.dot(rendements,Poids).reshape(-1,))   # calcul de la rentabilité du portefeuille
    résultat=-ratio(val_portefeuille_rendements,rendement_taux_sans_risque)  # Applique la fonction de ratio de performance (par exemple, Sharpe, Sortino, Calmar) aux rendements du portefeuille, et multiplie le résultat par -1 car l'optimiseur cherche à minimiser cette fonction.
    if np.isnan(résultat) or np.isinf(résultat): # pour enlever les problemes de calculs si denominateur de raproche de 0 ou inf
        return 10
    return résultat

def scipy_func_avec_pénalité(x,arguments):  #algo plus complexe mais plus lent  # Définition de la fonction avec deux arguments: x (les poids des actifs) et arguments (un tuple contenant le ratio de performance, le taux de rendement sans risque et les rendements historiques des actifs).
    Poids=x.reshape(-1,1)   # Reformate le vecteur des poids x en une matrice colonne pour faciliter les opérations matricielles.
    ratio, rendement_taux_sans_risque, rendements = arguments    
    val_portefeuille_rendements=pd.Series(np.dot(rendements,Poids).reshape(-1,))   # Calcule les rendements du portefeuille en multipliant la matrice des rendements historiques par les poids, puis transforme le résultat en une série pandas.
    résultat=-ratio(val_portefeuille_rendements,rendement_taux_sans_risque)
    pénalité=100* np.abs((np.sum(np.abs(x))-1)) # Calcule une pénalité proportionnelle à l'écart entre la somme des poids absolus des actifs et 1. Cette pénalité est ajoutée pour forcer les poids à s'additionner à 1.
    if np.isnan(résultat) or np.isinf(résultat): # Vérifie si le résultat est NaN (non défini) ou infini, ce qui peut arriver si les calculs ne sont pas valides (par exemple, division par zéro).
        return 1000 + pénalité  # Si le résultat est NaN ou infini, retourne une valeur très élevée (1000 dans ce cas) plus la pénalité, ce qui décourage l'optimiseur de choisir cette solution.
    return résultat + pénalité  # Retourne le résultat de la fonction de ratio de performance ajusté par la pénalité. Cela permet à l'optimiseur de prendre en compte à la fois la performance du portefeuille et le respect de la contrainte des poids.


def contrainte(x):  
    return np.sum(x) - 1  # La somme des poids des actifs doit être égale à 1. Cette ligne calcule cette somme et soustrait 1, visant à obtenir zéro 


def optimisation_Poids(x, arguments,VAD:bool = False):  # Fonction d'optimisation qui utilise l'algorithme 'SLSQP' pour trouver les poids optimaux des actifs.
    cons = {'type': 'eq', 'fun': contrainte}  # Crée un dictionnaire représentant une contrainte d'égalité ('eq') qui utilise la fonction 'contrainte' définie précédemment.
    nombre_actifs = x.size  
    if VAD == True :
        bounds_ = [(-1, 1) for _ in range(nombre_actifs)] 
    else:
        bounds_ = [(0, 1) for _ in range(nombre_actifs)]
    resultat = minimize(scipy_func, x, args=arguments, constraints=cons, method='SLSQP',bounds=bounds_) 
    # Appelle la fonction 'minimize' de SciPy avec la méthode 'SLSQP', en passant la fonction d'objectif 'scipy_func', le vecteur de poids initial 'x', les arguments supplémentaires, les contraintes et spécifie la méthode d'optimisation.
    return resultat  

def optimisation_Poids_efficace(x, arguments,VAD:bool = False):  # Fonction d'optimisation alternative utilisant l'algorithme 'differential_evolution'.
    nombre_actifs = x.size
    if VAD == True :
        bounds_ = [(-1, 1) for _ in range(nombre_actifs)] 
    else:
        bounds_ = [(0, 1) for _ in range(nombre_actifs)]
    def objective_function(Poids):  # Fonction d'objectif interne pour 'differential_evolution'.
        return scipy_func_avec_pénalité(Poids, arguments)  # Appelle la fonction 'scipy_func_avec_pénalité' avec les poids actuels et les arguments passés.
    résultat = differential_evolution(objective_function, bounds=bounds_)  # Appelle la fonction 'differential_evolution' de SciPy avec la fonction d'objectif et les bornes. Cette méthode est plus robuste mais potentiellement plus lente que 'SLSQP'.
    return résultat  
