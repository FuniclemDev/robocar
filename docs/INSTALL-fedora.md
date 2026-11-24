# Installation sur Fedora

> Le simulateur Unity (`RacingSimulator`) était fourni par Epitech aux équipes Robocar et n'est pas public. Sans lui,
> on peut réentraîner les modèles et regarder les données, mais pas faire rouler la voiture.

## 1. Dépendances

```bash
sudo dnf install python3 python3-devel gcc-c++ SDL2
git clone git@github.com:FuniclemDev/MyRobocarIA.git
cd MyRobocarIA
python3 -m venv venv_mlagents028
source venv_mlagents028/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fige toutes les versions, dont `mlagents-envs==0.28.0`, `lightgbm==4.6.0` et
`tensorflow==2.19.0` (cette dernière tire les paquets CUDA de NVIDIA ; elle n'est utile que pour les essais en
réseau de neurones). Python 3.10 à 3.12.

## 2. Placer le simulateur

Copiez la version Linux du simulateur dans `BuildLinux/`, de sorte que `BuildLinux/RacingSimulator` soit
l'exécutable, puis :

```bash
chmod +x BuildLinux/RacingSimulator
```

## 3. Utiliser

```bash
# enregistrer des tours (manette branchée) : écrit data/driving_data_dg.csv
python robot_ai.py

# entraîner : écrit model/driving_model_lgbm.pkl, model/scaler_X.save et deux graphiques
python train_model_lightGBM.py

# laisser conduire
python autopilot.py
```

`run.sh` enchaîne l'entraînement et l'autopilote.

## Sans simulateur

L'entraînement fonctionne seul, à partir des données du dépôt :

```bash
cp data/clem_ok/driving_data_dg_bestTRACK3.csv data/driving_data_dg.csv
python train_model_lightGBM.py
```

## Manette

Testé avec une manette Xbox 360. Sur une autre manette, l'axe de la gâchette d'accélération peut changer :
ajustez `R2_AXIS` en haut de `robot_ai.py`.
