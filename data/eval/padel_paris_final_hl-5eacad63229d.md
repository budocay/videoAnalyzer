# Évaluation — padel_paris_final_hl-5eacad63229d

315 frappes annotées (user : 315).

Précision de la détection de frappes : 78 % (n=251) des frappes détectées sont de vraies frappes.

| Méthode | Type de coup correct |
|---|---|
| pose + VLM (pipeline sans classifieur) | 22 % (n=197) |
| vlm | 22 % (n=197) |
| pose | 23 % (n=197) |
| classifieur (validation croisée) | 44 % (n=194) |

Bandeja et víbora sont comptées comme une seule classe.

## Matrice de confusion (pose + VLM)

| vrai \ prédit | bandeja_vibora | coup_droit | lob | revers | service | smash | sortie_vitre | volee_cd | volee_revers |
|---|---|---|---|---|---|---|---|---|---|
| **bandeja_vibora** | 7 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| **coup_droit** | 12 | 1 | 0 | 5 | 0 | 1 | 0 | 4 | 0 |
| **lob** | 11 | 1 | 0 | 6 | 0 | 2 | 0 | 1 | 0 |
| **revers** | 16 | 0 | 0 | 11 | 0 | 1 | 0 | 2 | 5 |
| **service** | 15 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 1 |
| **smash** | 8 | 1 | 0 | 1 | 0 | 4 | 0 | 4 | 1 |
| **sortie_vitre** | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **volee_cd** | 5 | 0 | 0 | 1 | 0 | 6 | 0 | 12 | 7 |
| **volee_revers** | 8 | 0 | 0 | 1 | 0 | 3 | 0 | 13 | 8 |
