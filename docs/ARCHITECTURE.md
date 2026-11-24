# Fonctionnement de l'IA

```
  manette Xbox 360 ──► robot_ai.py ──────────────► data/driving_data_dg.csv
  (pygame)             │  ▲                         40 capteurs, vitesse, direction,
                       ▼  │ ML-Agents (port 5004)   position x / y / z
                 simulateur Unity (RacingSimulator)
                       ▲  │
                       │  ▼
  model/*.pkl ◄── train_model_lightGBM.py ◄── données nettoyées, virages sur-échantillonnés
       │
       └────────► autopilot.py : capteurs → modèle → accélération et direction, à chaque pas
```

## Les fichiers

| Fichier | Rôle |
| --- | --- |
| `config.json` | réglage des capteurs passé au simulateur : 40 rayons sur 180° |
| `robot_ai.py` | pilotage à la manette et enregistrement des mesures |
| `robot_split.py` | variante au clavier (Z Q S D) qui écrit un fichier par tour |
| `verif.py` | courbes de la direction et de la vitesse d'un enregistrement, pour repérer les données pauvres |
| `train_model_tensorflow.py`, `train_model.py` | essais avec un réseau de neurones dense (Keras) |
| `train_model_lightGBM.py` | entraînement retenu : LightGBM, quatre sorties |
| `autopilot.py` | conduite autonome dans le simulateur, compte des tours |
| `*_bonnepiste.py` | versions plus récentes : enregistrement de la vitesse réelle et du frein, autopilote qui freine |
| `data/clem_ok/` | les trois meilleurs enregistrements, un par piste |
| `model/track1..3/` | un modèle entraîné et sa normalisation par piste, avec leurs graphiques |

## Ce qu'échangent Python et le simulateur

Le simulateur est un exécutable Unity construit avec **ML-Agents**. `mlagents_envs.UnityEnvironment` le lance et
dialogue avec lui sur le port 5004 :

- **Observations**, à chaque pas : les 40 distances mesurées par les rayons, puis d'autres valeurs internes, dont la
  position de la voiture (indices 52 à 54).
- **Actions** : deux valeurs continues, l'accélération (0 à 1) et la direction (−1 à gauche, +1 à droite).

## Les données

Une ligne par pas de simulation :

```
ray_0 … ray_39, speed, steering, pos_x, pos_y, pos_z, real_velocity, break
```

- `ray_0` est le rayon le plus à droite, `ray_39` le plus à gauche ;
- `speed` et `steering` sont les commandes envoyées par la manette ;
- `real_velocity` est la vitesse mesurée, `break` vaut 1 quand je freinais.

Trois enregistrements ont été retenus, après avoir écarté les tours ratés (`data/erroné`) : la boucle extérieure
(2 180 mesures), et deux parcours par les routes intérieures (12 695 et 9 192 mesures).

## Le modèle

- **Entrées** : les 40 rayons et la position `x`, `z`, normalisés (`StandardScaler`).
- **Sorties** : accélération, direction, vitesse réelle, freinage.
- **Modèle** : `MultiOutputRegressor(LGBMRegressor(n_estimators=4000, learning_rate=0.02, max_depth=14,
  num_leaves=60))`, un ensemble d'arbres par sortie.
- **Rééquilibrage** : les virages moyens (|direction| entre 0,2 et 0,6) sont copiés 5 fois, les virages serrés 9 fois,
  les freinages 5 fois, puis tout est mélangé.
- **Évaluation** : 20 % des données mises de côté, erreur absolue moyenne par sortie, graphique prédit / réel et
  importance de chaque entrée.

## L'autopilote

À chaque pas : lecture des observations, normalisation, prédiction. Si le modèle prédit un freinage et que la voiture
roule, l'accélération passe à zéro ; sinon elle ne descend pas sous 0,5 pour éviter que la voiture cale. Un tour est
compté quand la voiture repasse près de son point de départ, au moins cinq secondes plus tard, et sa durée est
affichée.
