# Entraînement du classifieur de coups

182 frappes annotées utilisées, validation croisée en 5 plis.
Vidéos : {'padel_paris_final_hl-a0c2b4f5baae': 184} · **non utilisées (pas de cache ni de features)** : ['padel_paris_final_hl-5eacad63229d']

| Méthode | Précision sur les mêmes frappes |
|---|---|
| classifieur_cv | 45 % (n=182) |
| vlm | — |
| pose | 14 % (n=182) |

Exemples par classe : {'coup_droit': 10, 'lob': 7, 'pas_une_frappe': 71, 'revers': 9, 'service': 12, 'smash': 30, 'volee_cd': 24, 'volee_revers': 19}
Classes écartées (moins de 4 exemples) : {'bandeja_vibora': 2}

## Matrice de confusion (classifieur, validation croisée)

| vrai \ prédit | coup_droit | lob | pas_une_frappe | revers | service | smash | volee_cd | volee_revers |
|---|---|---|---|---|---|---|---|---|
| **coup_droit** | 0 | 0 | 2 | 0 | 1 | 2 | 2 | 3 |
| **lob** | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 0 |
| **pas_une_frappe** | 3 | 0 | 43 | 6 | 4 | 3 | 6 | 6 |
| **revers** | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 5 |
| **service** | 2 | 0 | 3 | 0 | 7 | 0 | 0 | 0 |
| **smash** | 0 | 0 | 6 | 0 | 2 | 19 | 2 | 1 |
| **volee_cd** | 1 | 0 | 12 | 0 | 1 | 2 | 5 | 3 |
| **volee_revers** | 0 | 0 | 9 | 3 | 0 | 2 | 2 | 3 |
