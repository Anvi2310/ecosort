"""Estimate contamination, moisture, recyclability and hazard without lab measurements.

A user cannot measure "contamination = 37%" from a photo. Instead each input is derived
from evidence a person (or a model) can actually provide, and every value carries its
source so the decision engine's explanation stays honest:

* recyclability  <- material knowledge base (category + item type). It is a property of
                    the material stream, not of the photo.
* hazard         <- item type baseline + yes/no safety flags (battery, chemicals, sharp...).
                    Never inferred from pixels alone.
* contamination  <- questionnaire (residue, mixed materials), else visual model, else prior.
* moisture       <- questionnaire (dry/damp/wet), else visual model, else prior.

Numbers are on the 0-100 scale the fuzzy system expects and act as the centres of the
linguistic answers (e.g. "visible residue" ~ 60 = between MEDIUM and HIGH contamination).
They are expert defaults, documented here so they can be revised or tuned by the GA.
"""

# --- Material knowledge base --------------------------------------------------------
# recyclability: how readily the item is accepted and recovered by a typical municipal
# stream (for Organic it means compostability). hazard: baseline handling risk.
# contamination/moisture: typical condition when nothing else is known.
MATERIAL_PROFILES = {
    "Plastic": {
        "default": "pet_bottle",
        "items": {
            "pet_bottle": {"label": "PET bottle / clear container (#1)", "recyclability": 85, "hazard": 3},
            "hdpe": {"label": "HDPE jug / detergent bottle (#2)", "recyclability": 78, "hazard": 8},
            "pp_tub": {"label": "PP tub or lid (#5)", "recyclability": 55, "hazard": 3},
            "film": {"label": "Plastic bag / film / wrap", "recyclability": 20, "hazard": 3},
            "polystyrene": {"label": "Polystyrene foam (#6)", "recyclability": 10, "hazard": 5},
            "multilayer": {"label": "Chip packet / multilayer pouch", "recyclability": 5, "hazard": 3},
        },
        "prior": {"contamination": 30, "moisture": 15},
    },
    "Paper": {
        "default": "cardboard",
        "items": {
            "office": {"label": "Office paper / newspaper", "recyclability": 92, "hazard": 1},
            "cardboard": {"label": "Cardboard box", "recyclability": 88, "hazard": 1},
            "carton": {"label": "Drink carton (Tetra Pak)", "recyclability": 45, "hazard": 1},
            "coated": {"label": "Coated / waxed / paper cup", "recyclability": 25, "hazard": 1},
            "tissue": {"label": "Tissue / napkin / paper towel", "recyclability": 5, "hazard": 5},
        },
        "prior": {"contamination": 20, "moisture": 15},
    },
    "Metal": {
        "default": "aluminium_can",
        "items": {
            "aluminium_can": {"label": "Aluminium can", "recyclability": 96, "hazard": 3},
            "steel_can": {"label": "Steel / tin food can", "recyclability": 90, "hazard": 8},
            "foil": {"label": "Aluminium foil / tray", "recyclability": 65, "hazard": 2},
            "aerosol": {"label": "Aerosol can", "recyclability": 60, "hazard": 45},
            "scrap": {"label": "Scrap metal / utensils", "recyclability": 80, "hazard": 20},
        },
        "prior": {"contamination": 25, "moisture": 15},
    },
    "Glass": {
        "default": "bottle_jar",
        "items": {
            "bottle_jar": {"label": "Bottle or jar", "recyclability": 90, "hazard": 10},
            "broken": {"label": "Broken container glass", "recyclability": 70, "hazard": 55},
            "window_mirror": {"label": "Window / mirror / drinking glass", "recyclability": 15, "hazard": 40},
            "ceramic": {"label": "Ceramic / Pyrex (not container glass)", "recyclability": 0, "hazard": 30},
        },
        "prior": {"contamination": 20, "moisture": 10},
    },
    "Organic": {
        "default": "food_scraps",
        "items": {
            "food_scraps": {"label": "Fruit / vegetable scraps", "recyclability": 90, "hazard": 3},
            "cooked_food": {"label": "Cooked food / meat / dairy", "recyclability": 65, "hazard": 12},
            "yard": {"label": "Leaves / garden waste", "recyclability": 92, "hazard": 2},
            "compostable_pack": {"label": "Certified compostable packaging", "recyclability": 55, "hazard": 2},
        },
        "prior": {"contamination": 15, "moisture": 65},
    },
    "Electronic waste": {
        "default": "small_device",
        "items": {
            "small_device": {"label": "Small device / cables / charger", "recyclability": 60, "hazard": 45},
            "phone_laptop": {"label": "Phone / laptop / tablet", "recyclability": 70, "hazard": 65},
            "battery": {"label": "Loose battery", "recyclability": 50, "hazard": 90},
            "lamp": {"label": "CFL / fluorescent lamp", "recyclability": 40, "hazard": 85},
            "large_appliance": {"label": "Large appliance", "recyclability": 65, "hazard": 55},
        },
        "prior": {"contamination": 15, "moisture": 5},
    },
}

# --- Plain-language questionnaire ---------------------------------------------------
QUESTIONS = [
    {"id": "residue", "target": "contamination", "text": "Is there food, liquid or dirt on or inside it?",
     "options": [("none", "Clean", 5), ("light", "A few drops or crumbs", 30),
                 ("visible", "Visible residue or stains", 62), ("heavy", "Heavily soiled / food left inside", 90)]},
    {"id": "mixed", "target": "contamination", "text": "Is it made of only one material?",
     "options": [("single", "Yes, one material", 0), ("removable", "Has labels, caps or tape", 10),
                 ("fused", "Glued or fused to other materials", 25)]},
    {"id": "wetness", "target": "moisture", "text": "How wet is it?",
     "options": [("dry", "Dry", 5), ("damp", "Damp or condensation", 40), ("wet", "Wet or soaked", 85)]},
]
HAZARD_FLAGS = {
    "battery": ("Contains or might contain a battery", 85),
    "chemical": ("Held chemicals, paint, oil, pesticide or fuel", 75),
    "sharp": ("Broken, sharp or has needles", 60),
    "medical": ("Medical / sanitary waste", 80),
    "pressurised": ("Pressurised or not fully empty aerosol/gas", 70),
}


def catalog():
    """JSON-friendly description of items, questions and flags for the UI."""
    return {
        "materials": {cat: {"default": p["default"],
                            "items": [{"id": k, "label": v["label"]} for k, v in p["items"].items()]}
                      for cat, p in MATERIAL_PROFILES.items()},
        "questions": [{"id": q["id"], "text": q["text"], "target": q["target"],
                       "options": [{"id": o, "label": label, "value": v} for o, label, v in q["options"]]}
                      for q in QUESTIONS],
        "hazard_flags": [{"id": k, "label": v[0]} for k, v in HAZARD_FLAGS.items()],
    }


def _option(qid, answer):
    q = next(q for q in QUESTIONS if q["id"] == qid)
    for oid, label, value in q["options"]:
        if oid == answer:
            return label, value
    raise ValueError(f"unknown answer '{answer}' for question '{qid}'")


def _clip(v):
    return round(max(0.0, min(100.0, float(v))), 1)


def estimate(category, item=None, answers=None, flags=None, visual=None):
    """Return {values, details} where each detail names its source and confidence.

    ``visual`` is an optional {"contamination": x, "moisture": y} from the image model;
    explicit user answers always take precedence because the user can see/touch the item.
    """
    if category not in MATERIAL_PROFILES:
        raise ValueError("category must be one of: " + ", ".join(MATERIAL_PROFILES))
    profile = MATERIAL_PROFILES[category]
    item = item or profile["default"]
    if item not in profile["items"]:
        raise ValueError(f"unknown item type '{item}' for {category}")
    spec = profile["items"][item]
    answers = {k: v for k, v in (answers or {}).items() if v}
    flags = list(flags or [])
    unknown = [f for f in flags if f not in HAZARD_FLAGS]
    if unknown:
        raise ValueError("unknown hazard flag(s): " + ", ".join(unknown))
    visual = visual or {}
    details = {}

    details["recyclability"] = {"value": spec["recyclability"], "source": "knowledge base", "confidence": "high",
                                "reason": f"{spec['label']} in the {category} stream"}

    # Contamination: residue answer + mixed-material penalty.
    if "residue" in answers or "mixed" in answers:
        base_label, base = _option("residue", answers["residue"]) if "residue" in answers else (None, None)
        if base is None:
            base = visual.get("contamination", profile["prior"]["contamination"])
        extra_label, extra = _option("mixed", answers["mixed"]) if "mixed" in answers else (None, 0)
        reason = "; ".join(x for x in (base_label, extra_label) if x)
        details["contamination"] = {"value": _clip(base + extra), "source": "user answers",
                                    "confidence": "medium" if "residue" in answers else "low", "reason": reason}
    elif "contamination" in visual:
        details["contamination"] = {"value": _clip(visual["contamination"]), "source": "image model",
                                    "confidence": "low", "reason": "estimated from visual stains/residue cues"}
    else:
        details["contamination"] = {"value": profile["prior"]["contamination"], "source": "category prior",
                                    "confidence": "low", "reason": f"typical condition for {category}; answer the questions to refine"}

    if "wetness" in answers:
        label, value = _option("wetness", answers["wetness"])
        details["moisture"] = {"value": value, "source": "user answers", "confidence": "medium", "reason": label}
    elif "moisture" in visual:
        details["moisture"] = {"value": _clip(visual["moisture"]), "source": "image model",
                               "confidence": "low", "reason": "estimated from gloss/darkening cues"}
    else:
        details["moisture"] = {"value": profile["prior"]["moisture"], "source": "category prior",
                               "confidence": "low", "reason": f"typical condition for {category}"}

    # Hazard: the worst applicable evidence wins (a battery inside a toy is a battery hazard).
    hazard, reasons = spec["hazard"], [f"baseline for {spec['label']}"]
    for f in flags:
        label, value = HAZARD_FLAGS[f]
        reasons.append(label)
        hazard = max(hazard, value)
    details["hazard"] = {"value": _clip(hazard), "source": "knowledge base + safety flags" if flags else "knowledge base",
                         "confidence": "high" if flags else "medium", "reason": "; ".join(reasons)}

    values = {k: float(details[k]["value"]) for k in ("contamination", "moisture", "recyclability", "hazard")}
    return {"category": category, "item": item, "values": values, "details": details}
