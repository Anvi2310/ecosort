"""Explainable category-aware fuzzy rules and a Sugeno-style weighted output."""
from .membership_functions import default_partition

EXPERT_RULES = [
    "IF contamination LOW AND moisture LOW AND recyclability HIGH THEN priority HIGH",
    "IF contamination HIGH AND recyclability HIGH THEN priority MEDIUM",
    "IF contamination LOW AND recyclability HIGH AND hazard LOW THEN priority VERY HIGH",
    "IF category PAPER AND moisture HIGH THEN priority LOW",
    "IF category METAL AND contamination LOW THEN priority VERY HIGH",
    "IF category GLASS AND contamination MEDIUM THEN priority MEDIUM",
    "IF category ORGANIC AND moisture HIGH THEN composting priority HIGH",
    "IF category ELECTRONIC AND hazard HIGH THEN special disposal priority VERY HIGH",
    "IF contamination HIGH AND moisture HIGH THEN suitability LOW",
    "IF contamination LOW AND moisture LOW AND hazard LOW THEN suitability HIGH",
]

REC = {
 "Plastic": "Rinse and send to an accepted plastic recycling stream.",
 "Paper": "Keep dry and separate from food or liquid contamination.",
 "Metal": "Separate and send to a metal recycling facility.",
 "Glass": "Sort by local glass rules and remove non-glass parts.",
 "Organic": "Separate for composting or local organics collection.",
 "Electronic waste": "Use an authorized e-waste collection or take-back program.",
}

def _mf(v, k, p): return float(p[k](v))

def infer(values, category, partition=None):
    """Return suitability and priority independently, plus fired-rule explanations."""
    p = partition or default_partition()
    c, m, r, h = (float(values[k]) for k in ("contamination", "moisture", "recyclability", "hazard"))
    low_c, med_c, high_c = (_mf(c, k, p) for k in ("low", "medium", "high"))
    low_m, high_m = (_mf(m, "low", p), _mf(m, "high", p))
    low_r, high_r = (_mf(r, "low", p), _mf(r, "high", p))
    low_h, high_h = (_mf(h, "low", p), _mf(h, "high", p))
    # Each tuple is rule label, firing strength, suitability consequent, priority consequent.
    rules = [
      ("R1: low contamination, low moisture, high recyclability", min(low_c,low_m,high_r), 85, 78),
      ("R2: high contamination, high recyclability", min(high_c,high_r), 40, 48),
      ("R3: low contamination, high recyclability, low hazard", min(low_c,high_r,low_h), 92, 92),
      ("R4: paper with high moisture", min(high_m,1.0 if category=="Paper" else 0.0), 18, 20),
      ("R5: metal with low contamination", min(low_c,1.0 if category=="Metal" else 0.0), 90, 90),
      ("R6: glass with medium contamination", min(med_c,1.0 if category=="Glass" else 0.0), 58, 55),
      ("R7: organic with high moisture (composting)", min(high_m,1.0 if category=="Organic" else 0.0), 78, 82),
      ("R8: electronic waste with high hazard (special handling)", min(high_h,1.0 if category=="Electronic waste" else 0.0), 5, 95),
      ("R9: high contamination and high moisture", min(high_c,high_m), 12, 18),
      ("R10: low contamination, moisture, and hazard", min(low_c,low_m,low_h), 88, 75),
      ("Material recyclability and hazard baseline", max(0.05, high_r*(1-high_h)), 60, 30+0.55*r-0.35*h),
    ]
    def aggregate(col):
        total = sum(x[1] for x in rules)
        return sum(x[1]*x[col] for x in rules)/(total or 1)
    suitability, priority = aggregate(2), aggregate(3)
    active = [{"rule": name, "strength": round(w,4), "suitability": s, "priority": q}
              for name,w,s,q in sorted(rules,key=lambda x:x[1],reverse=True) if w >= .15]
    if category == "Electronic waste": action = "special disposal"
    elif category == "Organic": action = "composting"
    else: action = "recycling" if suitability >= 50 else "separation and appropriate disposal"
    return {"suitability": round(suitability,2), "priority": round(priority,2),
            "recommendation": REC[category], "decision": action,
            "active_rules": active[:6], "uncertainty": round(1-max((x[1] for x in rules), default=0),3)}
