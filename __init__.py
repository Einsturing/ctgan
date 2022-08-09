# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.4.4.dev0'

# from ctgan.demo import load_demo,load_adult,load_titanic,load_flare,load_abalone,load_breast,load_diabetes,load_ecoli,load_heart1,load_heart2,load_lung,load_yeast0,load_yeast6
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'load_demo',
    'load_adult',
    'load_titanic',
    'load_flare',
    'load_ecoli',
    'load_yeast6',
    'load_yeast0',
    'load_lung',
    'load_heart2',
    'load_heart1',
    'load_diabetes',
    'load_breast',
    'load_abalone'
)
