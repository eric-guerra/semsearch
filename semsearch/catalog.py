"""Synthetic home-goods catalog with the vocabulary mismatch real shoppers create: they type "couch"
when titles say "sofa", "duvet" when titles say "comforter". Descriptions use the alternate words
some of the time (as merchandising copy does), which is what lets a learned embedding connect them.

Also builds a judged query set (graded relevance 0-3) for offline evaluation.
"""
from __future__ import annotations

import random

from .search import Product

# category -> (title noun, synonyms used in descriptions/queries)
CATS = {
    "sofas": ("Sofa", ["couch", "settee", "loveseat"]),
    "bedding": ("Comforter", ["duvet", "quilt", "bedspread"]),
    "lighting": ("Floor Lamp", ["standing light", "uplighter", "reading lamp"]),
    "rugs": ("Area Rug", ["carpet", "runner", "mat"]),
    "storage": ("Bookcase", ["shelving unit", "bookshelf", "étagère"]),
    "kitchen": ("Dutch Oven", ["casserole pot", "cocotte", "braiser"]),
    "outdoor": ("Patio Umbrella", ["parasol", "sunshade", "market umbrella"]),
    "bath": ("Bath Towel", ["bath sheet", "towel set", "terry towel"]),
}
BRANDS = ["Hearth & Co", "Norden", "Casa Verde", "Ironwood", "Lumen", "Tidewater"]
MATERIALS = ["linen", "velvet", "oak", "walnut", "cotton", "wool", "cast iron", "teak", "brass"]
COLORS = ["navy", "sage", "charcoal", "ivory", "terracotta", "mustard"]
FEATURES = {
    "sofas": ["deep seat", "removable cushions", "kiln-dried frame", "stain resistant fabric"],
    "bedding": ["all-season warmth", "hypoallergenic fill", "machine washable", "corner ties"],
    "lighting": ["dimmable", "adjustable head", "warm LED", "weighted base"],
    "rugs": ["low pile", "non-slip backing", "hand woven", "pet friendly"],
    "storage": ["five shelves", "wall anchor included", "adjustable shelves", "open back"],
    "kitchen": ["enameled interior", "oven safe to 500F", "self-basting lid", "induction ready"],
    "outdoor": ["UV resistant canopy", "tilt mechanism", "wind vent", "crank lift"],
    "bath": ["600 GSM", "quick dry", "zero twist", "oversized"],
}


def build_catalog(seed: int = 3, per_cat: int = 40) -> list[Product]:
    rng = random.Random(seed)
    out = []
    n = 0
    for cat, (noun, syns) in CATS.items():
        for _ in range(per_cat):
            n += 1
            brand, mat, color = rng.choice(BRANDS), rng.choice(MATERIALS), rng.choice(COLORS)
            feats = rng.sample(FEATURES[cat], 2)
            title = f"{brand} {color.title()} {mat.title()} {noun}"
            if rng.random() < 0.25:  # only some merchandising copy uses the shopper's word
                syn = rng.choice(syns)
                desc = rng.choice([f"A {color} {mat} {syn} with {feats[0]} and {feats[1]}.",
                                   f"This {noun.lower()} has {feats[0]}; customers call it the perfect {syn}."])
            else:
                desc = rng.choice([f"A {color} {mat} {noun.lower()} with {feats[0]} and {feats[1]}.",
                                   f"{feats[0].capitalize()}, {feats[1]}. Built to last."])
            out.append(Product(f"P{n:04d}", title, desc, cat, brand,
                               rng.randint(20, 1500) * 100, in_stock=rng.random() > 0.08))
    return out


def judged_queries(catalog: list[Product], seed: int = 11, n: int = 60):
    """(query, kind, {product_id: grade}). kind: lexical | semantic | attribute."""
    rng = random.Random(seed)
    qs = []
    cats = list(CATS)
    for i in range(n):
        cat = cats[i % len(cats)]
        noun, syns = CATS[cat]
        kind = ["semantic", "lexical", "attribute"][i % 3]
        if kind == "semantic":
            q = rng.choice(syns)  # shopper's word never appears in the title
            rel = {p.id: 2 for p in catalog if p.category == cat}
        elif kind == "lexical":
            brand = rng.choice(BRANDS)
            q = f"{brand} {noun.lower()}"
            rel = {p.id: (3 if p.brand == brand else 1) for p in catalog if p.category == cat}
        else:
            color = rng.choice(COLORS)
            q = f"{color} {rng.choice(syns)}"
            rel = {p.id: (3 if color in p.title.lower() else 1) for p in catalog if p.category == cat}
        qs.append((q, kind, rel))
    return qs
