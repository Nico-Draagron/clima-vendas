#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script de teste do sistema

import sys

print("\nTESTANDO SISTEMA...")
print("="*50)

# Testar imports
imports_ok = True

try:
    import pandas as pd
    print("OK: pandas")
except:
    print("ERRO: pandas")
    imports_ok = False

try:
    import numpy as np
    print("OK: numpy")
except:
    print("ERRO: numpy")
    imports_ok = False

try:
    import sklearn
    print("OK: scikit-learn")
except:
    print("ERRO: scikit-learn")
    imports_ok = False

try:
    import streamlit
    print("OK: streamlit")
except:
    print("ERRO: streamlit")
    imports_ok = False

try:
    import requests
    print("OK: requests")
except:
    print("ERRO: requests")
    imports_ok = False

try:
    import plotly
    print("OK: plotly")
except:
    print("ERRO: plotly")
    imports_ok = False

try:
    import schedule
    print("OK: schedule")
except:
    print("ERRO: schedule")
    imports_ok = False

try:
    import cfgrib
    print("OK: cfgrib")
except:
    print("AVISO: cfgrib (opcional para dados GRIB2)")

try:
    import xarray
    print("OK: xarray")
except:
    print("AVISO: xarray (opcional para dados NetCDF)")

print("="*50)
if imports_ok:
    print("SUCESSO: Sistema pronto para uso!")
else:
    print("ERRO: Instale as dependencias faltantes")
    print("Execute: pip install -r requirements_weather.txt")
