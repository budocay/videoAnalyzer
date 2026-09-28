"""Padel match analysis (phase 2): shots → court → players/pose → strokes → score → report.

Every stage caches its output under <cache>/<video_key>/padel/ and writes into the main
document's `extensions.padel`, so the generic timeline stays untouched.
"""
