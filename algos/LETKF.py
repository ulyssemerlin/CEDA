#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:39:51 2026

@author: ulyssemerlin

Implement LETKF data assimilation method tailored for Lorenz 96 model. 

Based on Hunt 2007 version of ETKF

"""

import numpy as np
from multiprocessing import Pool
from numpy.linalg import inv
from scipy.integrate import ode
import numpy as np
from algos.utils import RMSE, inv_svd, sqrt_svd, decompo_propre, localise_temp, select_ind
from tqdm import tqdm
from scipy.linalg import sqrtm # pour calculer racine matrice

def _LETKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, obs,rho,L, prng):
    sqB = sqrt_svd(B)
    
    Xa = np.zeros([Nx, Ne, T+1])
    Xf = np.zeros([Nx, Ne, T])
    
    # Initialize ensemble
    for i in range(Ne):
      Xa[:,i,0] = xb + sqB.dot(prng.normal(size=Nx))
      
    for t in range(T):
        if t%100== 0:
            print(f"{(t/T)*100:.3f} %")
        # FORCAST
        # on applique seulement le modèle de manière déterministe
        Xf[:,:,t] = f(Xa[:,:,t]) # on indexe à t mais en réalité c'est à t+1
        # on extrait les informations qui nous intéresse 
        Xf_mean = np.mean(Xf[:,:,t], axis = 1, keepdims=True)
        dXf = Xf[:,:,t]-Xf_mean
        Y = H.dot(Xf[:,:,t])
        Y_mean = np.mean(Y, axis = 1,keepdims=True)
        dY = Y-Y_mean
        # on crée un liste de variable pour facilité la localisation 
        list_var = np.arange(Nx)
        
        # si aucune obs disponible
        if not False in np.isnan(obs[:,t+1]):
            Xa[:,:,t+1] = Xf[:,:,t]
        else:
            for var in range(Nx): 
                # LOCALISATION
                # sélection des points d'interêt
                ind_loc = select_ind(var, L,Nx, list_var)
                # les indices des points à considérer
                # on garde seulement les lignes intéressantes dans xf_barre et deltaXf 
                # pareil pour y_barre et deltaY
                Xf_mean_loc, dXf_loc, Y_mean_loc , dY_loc, R_loc, obs_loc = localise_temp(ind_loc,Xf_mean, dXf, Y_mean, dY,R, obs[:,t+1].reshape(-1,1)) # uniquement si R est diag
                # si aucune obs disponible dans le rayon de localisation
                if not False in np.isnan(obs_loc):
                    Xa[:,:,t+1][var,:] = Xf[:,:,t][var,:]
                else:
                    # ANALYSE
                    # étape 1 : on calcul un xa qui minimise J(x)= (x-x_barre)P⁻1(x-x_barra) + (y-Hx)R⁻1(y-Hx)
                    # pour ça il faut passer dans un espace S et introduire wa qui dépend d'un Pa_tilde
                    Pa_tile_inv = (Ne-1)*np.eye(Ne)/rho + (dY_loc.T).dot(inv_svd(R_loc)).dot(dY_loc)
                    Pa_tilde = inv_svd(Pa_tile_inv)
                    
                    # on définit un pseudo gain juste pour calculer wa
                    K_gain = Pa_tilde.dot(dY_loc.T).dot(inv_svd(R_loc)) # de dimension Ne * Nx
                    #on supprimer les indices pour lesquel il n'y a pas d'obs
                    innov = obs_loc-Y_mean_loc
                    ind,_=np.where(np.isnan(innov) == True) # indice des lignes à supprimer
                    innov = np.delete(innov,ind,0)
                    K_gain = np.delete(K_gain,ind,1)
                    wa = K_gain.dot(innov) # de dimension Ne*1
                    
                    # on peut enfin calculer la moyenne de l'analyse suivante
                    xa_barre = Xf_mean_loc + dXf_loc.dot(wa)
                    
                    # étape 2
                    # on veut retrouver l'ensemble à partir de Pa et plus particulirement de Pa_tilde
                    # il faut calculer une racine carré
                    Transform = sqrtm(Pa_tilde) * np.sqrt(Ne - 1) 
                    # on calcul dXA on ajoute un coef "alpha " mutiplicatif pour l'augmenter artificiellement 
                    dXa = dXf_loc.dot(Transform)
                    Xa_loc = xa_barre + dXa
                    Xa[:,:,t+1][var,:] = Xa_loc[L//2,:]
    return Xa, Xf


def LETKF(params,prng):
    Nx = params['state_size']
    Ne = params['nb_particles']
    T  = params['temporal_window_size']
    H  = params['observation_matrix']
    R  = params['observation_noise_covariance']
    obs = params['observations']
    Xt = params['true_state']
    No = params['observation_size']
    xb = params['background_state']
    B  = params['background_covariance']
    Q  = params['model_noise_covariance']
    alpha = params['inflation_factor']
    f  = params['model_dynamics']
    rho = params['rho']
    L = params['L']
    
    Xa, Xf = _LETKF(Nx, T, No, xb, B, Q, R, Ne, alpha, f, H, obs,rho,L, prng)
    
    res = {
            'analysis_ensemble': Xa,
            'forecast_ensemble': Xf,
            'params'           : params
           }
    return res