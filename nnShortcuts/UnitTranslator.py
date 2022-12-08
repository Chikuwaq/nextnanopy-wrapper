# -*- coding: utf-8 -*-
"""
Created: 2022/03/25
Updated: 2022/03/25

@author: takuma.sato
"""
from math import pi

hbar = 1.054571628E-34   # Planck constant / 2Pi in [J.s]
electron_mass = 9.10938356E-31   # in [kg]
e = 1.602176E-19   # elementary charge

scale_Joules_to_eV = 1 / e

def wavenumber_to_energy(sound_velocity, k_in_inverseMeter):
    """
    for acoustic phonons
    E = hbar * omega = hbar * c * k
    """
    E_in_Joules = hbar * sound_velocity * k_in_inverseMeter
    return E_in_Joules * scale_Joules_to_eV