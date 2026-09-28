"""Padel rules and court geometry used by the analysis (FIP rules; Premier Padel scoring).

Sources: FIP "Reglamento de juego del pádel" (court dimensions, service, ball in play) and the
Premier Padel tour regulations (scoring format).
"""

# --- Court (metres) -------------------------------------------------------------------------
COURT_LENGTH = 20.0             # interior, wall to wall
COURT_WIDTH = 10.0
NET_Y = COURT_LENGTH / 2
NET_HEIGHT_CENTER = 0.88
NET_HEIGHT_POSTS = 0.92
SERVICE_LINE_FROM_NET = 6.95
SERVICE_LINE_FROM_WALL = NET_Y - SERVICE_LINE_FROM_NET  # 3.05
BACK_WALL_HEIGHT = 4.0          # glass 3 m + 1 m mesh
SIDE_GLASS_LENGTH = 4.0         # side glass panels from each back wall

# --- Zones used for tactical stats (distance to the net) --------------------------------------
NET_ZONE_MAX = 4.0        # "red": volley zone, up to ~4 m from the net
TRANSITION_ZONE_MAX = 6.95  # between volley zone and service line: transition
# beyond the service line (> 6.95 m from the net): defence / back of the court

# --- Scoring -----------------------------------------------------------------------------------
POINTS = ["0", "15", "30", "40"]
GAMES_TO_WIN_SET = 6      # by 2 games; tie-break at 6-6 (first to 7 points, by 2)
SETS_TO_WIN_MATCH = 2     # best of three sets
# Premier Padel (since 2025): after the second deuce of a game, the next point decides the game
# ("star point"). Before 2025 the tour used the golden point at the first deuce. The receivers
# choose the side for a deciding point. Scoreboard reading only needs to recognise the states.

# --- Stroke vocabulary (Spanish/French names as used in padel) -------------------------------
STROKES = {
    "service": "service : frappe sous la taille après un rebond, derrière la ligne de service",
    "coup_droit": "coup droit (drive) : frappe côté raquette après rebond, sous l'épaule",
    "revers": "revers : frappe côté opposé à la raquette après rebond, sous l'épaule",
    "volee_cd": "volée coup droit : frappe sans rebond près du filet, côté raquette, sous la tête",
    "volee_revers": "volée revers : frappe sans rebond près du filet, côté opposé, sous la tête",
    "bandeja": "bandeja : frappe au-dessus de l'épaule, coupée, contrôlée, souvent depuis le milieu du "
               "terrain, pour garder le filet ; geste court, raquette qui reste devant",
    "vibora": "víbora : frappe au-dessus de l'épaule avec effet latéral/coupé, plus agressive que la "
              "bandeja, contact un peu plus latéral",
    "smash": "smash : frappe au-dessus de la tête bras tendu, puissante, pour finir le point (à plat, "
             "par 3 ou par 4 pour sortir la balle du court)",
    "lob": "lob (globo) : balle haute qui passe par-dessus les adversaires au filet, frappée depuis le fond",
    "sortie_vitre": "sortie de vitre (bajada/contre-vitre) : frappe après que la balle a rebondi sur la vitre",
}

# Rules a rally analysis must respect
RULE_NOTES = [
    "Le service se fait à la cuillère : la balle est frappée au niveau ou sous la taille, après un "
    "rebond, et doit rebondir dans le carré de service opposé en diagonale.",
    "Après avoir franchi le filet, la balle doit d'abord rebondir au sol dans le camp adverse ; "
    "ensuite seulement elle peut toucher les vitres ou le grillage.",
    "Un joueur peut jouer la balle après son rebond sur ses propres vitres ; il peut aussi la "
    "renvoyer contre sa propre vitre pour qu'elle franchisse le filet (contre-vitre).",
    "Une équipe au filet attaque (volées, bandejas, víboras, smashs) ; l'équipe au fond défend "
    "(lobs, sorties de vitre, chiquitas).",
    "Le point se termine par un coup gagnant, une faute directe, ou une balle qui rebondit deux fois.",
]
