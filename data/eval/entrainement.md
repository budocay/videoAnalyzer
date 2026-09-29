# Entraînement du classifieur de coups

353 frappes annotées utilisées, validation croisée en 5 plis.
Vidéos : {'padel_paris_final_hl-a0c2b4f5baae': 184, 'train1-cf94befa8120': 79, 'train2-46d97af5ebfd': 90} · **non utilisées (pas de cache ni de features)** : ['padel_paris_final_hl-5eacad63229d']

| Méthode | Précision sur les mêmes frappes |
|---|---|
| classifieur_cv | 42 % (n=353) |
| vlm | — |
| pose | 15 % (n=353) |

Exemples par classe : {'bandeja_vibora': 6, 'coup_droit': 15, 'lob': 9, 'pas_une_frappe': 127, 'revers': 10, 'service': 30, 'smash': 54, 'sortie_vitre': 5, 'volee_cd': 48, 'volee_revers': 49}

## Matrice de confusion (classifieur, validation croisée)

| vrai \ prédit | bandeja_vibora | coup_droit | lob | pas_une_frappe | revers | service | smash | sortie_vitre | volee_cd | volee_revers |
|---|---|---|---|---|---|---|---|---|---|---|
| **bandeja_vibora** | 0 | 0 | 0 | 2 | 0 | 0 | 3 | 0 | 1 | 0 |
| **coup_droit** | 0 | 1 | 1 | 1 | 0 | 1 | 3 | 1 | 2 | 5 |
| **lob** | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |
| **pas_une_frappe** | 0 | 0 | 1 | 74 | 3 | 6 | 9 | 0 | 12 | 22 |
| **revers** | 0 | 0 | 0 | 5 | 0 | 1 | 2 | 0 | 1 | 1 |
| **service** | 0 | 0 | 0 | 6 | 0 | 15 | 4 | 0 | 3 | 2 |
| **smash** | 1 | 1 | 0 | 7 | 2 | 4 | 31 | 0 | 6 | 2 |
| **sortie_vitre** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| **volee_cd** | 1 | 2 | 1 | 15 | 0 | 2 | 7 | 1 | 12 | 7 |
| **volee_revers** | 1 | 1 | 0 | 19 | 3 | 2 | 6 | 0 | 7 | 10 |
