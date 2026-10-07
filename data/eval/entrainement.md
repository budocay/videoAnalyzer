# Entraînement du classifieur de coups

604 frappes annotées utilisées, validation croisée en 5 plis.
Vidéos : {'padel_paris_final_hl-5eacad63229d': 251, 'padel_paris_final_hl-a0c2b4f5baae': 184, 'train1-cf94befa8120': 79, 'train2-46d97af5ebfd': 90}

| Méthode | Précision sur les mêmes frappes |
|---|---|
| classifieur_cv | 46 % (n=604) |
| vlm | 18 % (n=239) |
| pose | 16 % (n=604) |
| classifieur_match_jamais_vu | 35 % (n=604) |

Exemples par classe : {'bandeja_vibora': 16, 'coup_droit': 38, 'lob': 30, 'pas_une_frappe': 181, 'revers': 45, 'service': 52, 'smash': 73, 'sortie_vitre': 8, 'volee_cd': 79, 'volee_revers': 82}

## Sur un match jamais vu (entraîné sur les autres matchs)

La validation croisée mélange les frappes d'un même match et surestime. Ici chaque match est testé par un classifieur qui ne l'a jamais vu : c'est la précision à attendre sur une nouvelle vidéo.

| Match testé | Frappes | Tout correct | Type de coup correct | Fausses frappes écartées | Vraies frappes perdues |
|---|---|---|---|---|---|
| padel_paris_final_hl-5eacad63229d | 251 | 34 % | 24 % | 38/54 | 48/197 |
| padel_paris_final_hl-a0c2b4f5baae | 184 | 33 % | 24 % | 34/71 | 9/113 |
| train1-cf94befa8120 | 79 | 44 % | 30 % | 21/32 | 12/47 |
| train2-46d97af5ebfd | 90 | 32 % | 18 % | 17/24 | 26/66 |

## Matrice de confusion (classifieur, validation croisée)

| vrai \ prédit | bandeja_vibora | coup_droit | lob | pas_une_frappe | revers | service | smash | sortie_vitre | volee_cd | volee_revers |
|---|---|---|---|---|---|---|---|---|---|---|
| **bandeja_vibora** | 3 | 0 | 0 | 2 | 0 | 1 | 9 | 0 | 1 | 0 |
| **coup_droit** | 0 | 9 | 2 | 9 | 3 | 1 | 3 | 0 | 7 | 4 |
| **lob** | 0 | 3 | 14 | 10 | 1 | 1 | 0 | 0 | 1 | 0 |
| **pas_une_frappe** | 0 | 4 | 6 | 114 | 9 | 7 | 6 | 1 | 13 | 21 |
| **revers** | 0 | 7 | 4 | 10 | 12 | 0 | 1 | 0 | 5 | 6 |
| **service** | 2 | 2 | 1 | 9 | 1 | 29 | 3 | 1 | 1 | 3 |
| **smash** | 3 | 2 | 0 | 10 | 0 | 3 | 42 | 0 | 7 | 6 |
| **sortie_vitre** | 0 | 0 | 0 | 2 | 0 | 1 | 1 | 0 | 1 | 3 |
| **volee_cd** | 0 | 4 | 3 | 20 | 4 | 3 | 7 | 0 | 27 | 11 |
| **volee_revers** | 1 | 3 | 1 | 20 | 7 | 2 | 2 | 3 | 15 | 28 |
