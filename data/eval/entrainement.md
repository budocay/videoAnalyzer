# Entraînement du classifieur de coups

248 frappes annotées utilisées, validation croisée en 5 plis.

| Méthode | Précision sur les mêmes frappes |
|---|---|
| classifieur_cv | 47 % (n=248) |
| vlm | 18 % (n=236) |
| pose | 18 % (n=248) |

Exemples par classe : {'bandeja_vibora': 10, 'coup_droit': 23, 'lob': 21, 'pas_une_frappe': 54, 'revers': 35, 'service': 22, 'smash': 19, 'volee_cd': 31, 'volee_revers': 33}
Classes écartées (moins de 4 exemples) : {'sortie_vitre': 3}

## Matrice de confusion (classifieur, validation croisée)

| vrai \ prédit | bandeja_vibora | coup_droit | lob | pas_une_frappe | revers | service | smash | volee_cd | volee_revers |
|---|---|---|---|---|---|---|---|---|---|
| **bandeja_vibora** | 5 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 2 |
| **coup_droit** | 0 | 4 | 3 | 5 | 4 | 1 | 0 | 1 | 5 |
| **lob** | 0 | 4 | 11 | 3 | 1 | 0 | 0 | 1 | 1 |
| **pas_une_frappe** | 1 | 1 | 3 | 31 | 5 | 3 | 2 | 7 | 1 |
| **revers** | 2 | 4 | 5 | 3 | 16 | 0 | 1 | 3 | 1 |
| **service** | 2 | 2 | 1 | 1 | 3 | 12 | 0 | 0 | 1 |
| **smash** | 0 | 0 | 0 | 3 | 1 | 1 | 9 | 2 | 3 |
| **volee_cd** | 1 | 1 | 1 | 8 | 3 | 1 | 0 | 12 | 4 |
| **volee_revers** | 0 | 1 | 1 | 4 | 4 | 0 | 2 | 4 | 17 |
