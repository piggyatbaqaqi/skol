# Model comparison — outputs for review

*Generated 2026-09-15T00:23 from persisted benchmark outputs. No GPU used.*

## Read this first

The previous review (`data/r1_json_review_20260911.md`) was produced by a **mis-prompted** run: the benchmark sent a bare species description with no instruction and no chat template, so the model was never told what to do. That is fixed — every output below comes from each model's own chat template wrapping the pipeline's `DEFAULT_PROMPT`.

**Judge two things separately**, because they have different fixes:

1. **Did it extract the right facts?** A model-quality question. This is what decides whether the small, fast model is good enough.
2. **Did it use sensible keys?** This is the R5 problem, and constrained decoding fixes it *whichever* model wins — so do not penalise a model for it.

Conflating them makes a model with good extraction and idiosyncratic keys look worse than one that extracts poorly into tidy keys.

## Speed

| model | whole | whole_512 | clause_seq | clause_batch | cap-hits | failures |
|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B 4-bit | 64.25 | 42.50 | 79.46 | 141.04 | 0.20 | 0 |
| Mistral-7B bf16 | 56.00 | 38.92 | 69.77 | 35.65 | 0.20 | 0 |
| Qwen2.5-1.5B bf16 | 17.10 | 13.04 | 30.61 | 16.47 | 0.10 | 0 |
| Qwen2.5-7B 4-bit | 54.79 | 38.60 | 78.88 | 121.58 | 0.20 | 0 |
| Phi-3.5-mini bf16 | 38.85 | 27.51 | 55.02 | 29.48 | 0.10 | 0 |

Seconds per treatment. Lower is better. 81 527 treatments in the corpus, so 1 s/treatment ≈ 0.94 days.

## Key vocabulary — the R5 problem, measured

Top-level keys each model chose across the 20 whole-description outputs. Wide disagreement here is expected and is *not* a model-quality defect; it is what a decode-time grammar is for.

**Mistral-7B 4-bit** — 20/20 produced extractable JSON. Keys: `Species`×7, `Shape`×3, `Hyphae`×2, `Conidiophores`×2, `Conidiogenous cells`×2, `Conidia`×2, `Color`×2, `Size`×2, `Culture Characteristics`×1, `entity`×1, `subentity`×1, `value`×1

**Mistral-7B bf16** — 17/20 produced extractable JSON. Keys: `Species`×6, `Shape`×3, `Size`×3, `Color`×3, `Conidiophores`×2, `Hyphae`×2, `Type`×1, `Stroma`×1, `Perithecia`×1, `Perithecial Wall`×1, `Asci`×1, `Ascospores`×1

**Qwen2.5-1.5B bf16** — 18/20 produced extractable JSON. Keys: `species`×11, `culture_characteristics`×3, `conidiophores`×2, `ascospores`×2, `features`×2, `location`×2, `name`×2, `morphology`×1, `colonization`×1, `mycelium`×1, `conidiogenous_cells`×1, `conidia`×1

**Qwen2.5-7B 4-bit** — 20/20 produced extractable JSON. Keys: `Conidiogenous_cells`×3, `Conidiophores`×3, `Conidia`×3, `sexual_morph`×2, `culture_characteristics`×2, `Culture characteristics`×2, `Hyphae`×2, `Ascomata`×2, `saprobic`×1, `asexual_morph`×1, `Type species`×1, `Taxonomic Features`×1

**Phi-3.5-mini bf16** — 20/20 produced extractable JSON. Keys: `Species`×11, `TaxonomicFeatures`×5, `Conidia`×3, `Location`×2, `Conidiophores`×2, `Conidiogenous Cells`×2, `YearOfDescription`×1, `Type`×1, `Proliferation`×1, `Author`×1, `Publication`×1, `Year`×1


## Side by side — 10 treatments

Each model saw exactly the same description. Outputs are the *extracted* JSON (what the pipeline would store), truncated to 700 characters.

### 1. `taxon_2dbd3cd444c977…`

**Description** (732 chars) — Saprobic on dead branches of bamboo. Sexual morph: Undetermined. Asexual morph: Colonies on natural substrate effuse, black. Mycelium superficial. Conidiophores absent. Conidiogenous cells monoblastic, integrated, usually indistinguishable from superficial hyphae. Conidia 45–85 × 20–30 µm ( = 60 × 25 µm, n = 50), solitary, subglobose, amygdaliform to ovoid, dark brown to black, distoseptate, consisting of 2–6 columns, with 0–6 setose appendages. Appendages 20–40 × 3–5 µm ( = 25 × 4 µm, n = 50), 0–2-septate, dark brown, smooth, unbranched, straight or curved. Culture characteristics – Colonies on PDA reaching 40–50 mm after 2 weeks at 25 ℃, circular, dry, with dense mycelium, raised, entire a

**Mistral-7B 4-bit**
*poor*
```json
{"Species": [{"Description": "Saprobic on dead branches of bamboo.", "Saprobic_on": ["dead branches", "bamboo"]}, {"Sexual_morph": "Undetermined.", "Sexual_morph_type": ["Undetermined"]}, {"Asexual_morph": "Colonies on natural substrate effuse, black.", "Colony_characteristics": {"Shape": ["effuse"], "Color": ["black"]}}, {"Mycelium": "Mycelium superficial.", "Mycelium_location": ["superficial"]}, {"Conidiophores": "Conidiophores absent.", "Conidiophores_present": ["Absent"]}, {"Conidiogenous_cells": "Conidiogenous cells monoblastic, integrated, usually indistinguishable from superficial hyphae.", "Conidiogenous_cell_characteristics": {"Type": ["monoblastic"], "Integration": ["integrated"], 
```

**Mistral-7B bf16**
```json
*good*
{"Species": [{"Saprobic Habitat": ["dead branches of bamboo"], "Sexual Morph": ["Undetermined"], "Asexual Morph": {"Colony": ["effuse", "black"], "Mycelium": ["superficial"], "Conidiophores": ["absent"]}, "Asexual Morph - Conidiogenous Cells": ["monoblastic", "integrated", "usually indistinguishable from superficial hyphae"], "Conidia": [{"Size": ["45–85 × 20–30 µm", "60 × 25 µm"], "Shape": ["subglobose", "amygdaliform to ovoid"], "Color": ["dark brown to black"], "Septation": ["distoseptate", "consisting of 2–6 columns"], "Appendages": [{"Size": ["20–40 × 3–5 µm", "25 × 4 µm"], "Septation": ["0–2-septate"], "Color": ["dark brown"], "Surface": ["smooth"], "Branching": ["unbranched"], "Shape"
```

**Qwen2.5-1.5B bf16**
*ok*
```json
{"species": {"name": "Undetermined", "description": "Saprobic on dead branches of bamboo. "}, "morphology": [{"type": "Sexual Morph", "value": "Undetermined"}], "colonization": [{"type": "Asexual Morph", "value": "Colonies on natural substrate effuse, black."}], "mycelium": [{"type": "Mycelium", "value": "Superficial"}], "conidiophores": [], "conidiogenous_cells": [{"type": "Monoblastic", "value": "Integrated, usually indistinguishable from superficial hyphae."}], "conidia": [{"type": "Conidia", "value": [{"size": ["45–85 µm", "20–30 µm"]}, {"shape": ["subglobose", "amygdaliform to ovoid"]}, {"color": ["dark brown to black"]}, {"setae": ["distoseptate"]}, {"appendages": [{"length": ["20–40 µ
```

**Qwen2.5-7B 4-bit**
*ok*
```json
{"saprobic": {"substrate": ["dead branches of bamboo"]}, "sexual_morph": {"status": ["Undetermined"]}, "asexual_morph": {"colonies": {"natural_substrate": {"growth": ["effuse"], "color": ["black"]}, "mycelium": {"location": ["superficial"]}, "conidiophores": {"presence": ["absent"]}, "conidiogenous_cells": {"type": ["monoblastic", "integrated"], "description": ["usually indistinguishable from superficial hyphae"]}, "conidia": {"size": ["45–85 × 20–30 µm ( = 60 × 25 µm, n = 50)"], "shape": ["solitary", "subglobose", "amygdaliform to ovoid"], "color": ["dark brown to black"], "septation": ["distoseptate"], "structure": ["consisting of 2–6 columns, with 0–6 setose appendages"]}, "appendages": {
```

**Phi-3.5-mini bf16**
*ok*
```json
{"Species": {"Taxonomic_Features": {"Habitat": ["Saprobic on dead branches of bamboo"], "Sexual_Morph": ["Undetermined"], "Asexual_Morph": ["Colonies on natural substrate effuse", "Black", "Mycelium superficial", "Conidiophores absent"], "Conidiogenous_Cells": ["Monoblastic", "Integrated", "Usually indistinguishable from superficial hyphae"], "Conidia": ["45–85 × 20–30 µm", "Solitary", "Subglobose", "Amygdaliform to ovoid", "Dark brown to black", "Distoseptate", "Consisting of 2–6 columns", "With 0–6 setose appendages"], "Appendages": ["20–40 × 3–5 µm", "0–2-septate", "Dark brown", "Smooth", "Unbranched", "Straight or curved"], "Culture_Characteristics": ["Colonies on PDA reaching 40–50 mm a
```

### 2. `taxon_2d5fa193e5ccf6…`

**Description** (686 chars) — Culture characteristics.Colonies after 7 d at 25 °C, on OA reaching 50–55 mm diam after 7 d, covered by floccose aerial mycelium, pink to grey, darker grey near the centre, margin regular; reverse black near the centre, yellow towards the periphery. On MEA reaching 50–55 mm diam, aerial mycelium floccose to cottony, buff with some mouse grey zones, margin regular; reverse orange with some radial yellow lines and some black zones. On PDA, reaching 45–50 mm diam, covered by floccose aerial mycelium, vinaceous grey to pale olivaceous, olivaceous grey near the centre, margin irregular; reverse buff to orange, black near the centre. NaOH spot test on OA: pale reddish discolouration.

**Mistral-7B 4-bit**
*ok*
```json
{"Culture Characteristics": {"Colonies": {"Diameter (after 7 d at 25 °C)": ["50–55 mm"], "Aerial Mycelium (OA)": ["floccose", "pink to grey", "darker grey near the centre", "regular"], "Aerial Mycelium (MEA)": ["floccose to cottony", "buff", "mouse grey zones", "regular"], "Aerial Mycelium (PDA)": ["floccose", "vinaceous grey to pale olivaceous", "olivaceous grey near the centre", "irregular"], "Reverse (OA)": ["black near the centre", "yellow towards the periphery"], "Reverse (MEA)": ["orange", "radial yellow lines", "some black zones"], "Reverse (PDA)": ["buff to orange", "black near the centre"]}, "NaOH Spot Test (OA)": ["pale reddish discolouration"]}}
```

**Mistral-7B bf16**
*good*
```json
{"Species": ["Culture characteristics", {"Colonies": [{"After": ["7 d", "25 °C"], "On": ["OA"], "Reaching": ["50–55 mm diam"], "Covered by": ["floccose aerial mycelium"], "Color": ["pink", "grey", "darker grey", "margin regular"], "Reverse": ["black near the centre", "yellow towards the periphery"]}, {"On": ["MEA"], "Reaching": ["50–55 mm diam"], "Aerial mycelium": ["floccose to cottony", "buff", "some mouse grey zones"], "Margin": ["regular"], "Reverse": ["orange", "some radial yellow lines", "some black zones"]}, {"On": ["PDA"], "Reaching": ["45–50 mm diam"], "Covered by": ["floccose aerial mycelium"], "Color": ["vinaceous grey to pale olivaceous", "olivaceous grey near the centre"], "Marg
```

**Qwen2.5-1.5B bf16**
*good*
```json
{"culture_characteristics": {"colonies_after_7_d_at_25_C": ["reaching 50-55 mm diam"], "reverse_color_change": {"centre": {"color": "dark grey", "near_center": "grey"}, "periphery": {"color": "yellow"}}, "aerial_mycelium": {"OA": {"texture": "floccose", "color": "pink to grey", "center": "darker grey", "margin": "regular"}, "MEA": {"texture": "cottony", "color": "buff with some mouse grey zones", "center": "buff", "margin": "orange with radial yellow lines and some black zones"}, "PDA": {"texture": "floccose", "color": "vinaceous grey to pale olivaceous", "center": "olivaceous grey", "margin": "irregular"}}, "naoh_spot_test": {"result": "pale reddish discolouration"}}}
```

**Qwen2.5-7B 4-bit**
*ok*
```json
{"Culture characteristics": {"Colonies": {"On OA": {"after 7 d at 25 °C": ["reaching 50–55 mm diam"], "covered by": {"floccose aerial mycelium": ["pink to grey", "darker grey near the centre"]}, "margin": ["regular"], "reverse": {"black near the centre": ["yellow towards the periphery"]}}, "On MEA": {"after 7 d": {"reaching 50–55 mm diam": ["aerial mycelium floccose to cottony", "buff with some mouse grey zones", "margin regular"]}, "reverse": {"orange with some radial yellow lines": ["some black zones"]}}, "On PDA": {"reaching 45–50 mm diam": ["covered by floccose aerial mycelium", "vinaceous grey to pale olivaceous", "olivaceous grey near the centre"], "margin": ["irregular"], "reverse": {
```

**Phi-3.5-mini bf16**
*excellent*
```json
{"Species": {"Colonies": {"Growth": ["7 days", "25 °C"], "Size": ["50-55 mm diam"], "Appearance": {"OA": {"Surface": ["floccose aerial mycelium", "pink", "darker grey near the centre", "margin regular", "reverse black near the centre", "yellow towards the periphery"], "NaOH_Spot_Test": ["pale reddish discolouration"]}, "MEA": {"Surface": ["floccose to cottony aerial mycelium", "buff", "some mouse grey zones", "margin regular", "reverse orange", "some radial yellow lines", "some black zones"]}, "PDA": {"Surface": ["floccose aerial mycelium", "vinaceous grey to pale olivaceous", "olivaceous grey near the centre", "margin irregular", "reverse buff to orange", "black near the centre"]}}}}}
```

### 3. `taxon_4b89d160bd68b7…`

**Description** (1876 chars) — Type species: Stylonectria applanata Höhn. 1915. Stroma thin, whitish or yellow, hyphal or subiculum-like. Perithecia gregarious in groups of up to 20, subglobose, pyriform to subcylindrical, with a rounded or broad, circular, flat disc on a venterlike neck, sometimes laterally collapsing when dry, pale yellow, orange-red, orange-brown, or pale to dark red, KOH+ dark red to purple, yellow in lactic acid, smooth, usually shiny, slightly iridescent, 150–250(–350) μm high. Perithecial wall consisting of two regions: inner region of hyaline, thin-walled, compressed, elongate cells; outer region of distinct, isodiametric to oblong, angular or globose, thick-walled cells. Asci cylindrical to clava

**Mistral-7B 4-bit**
```json
{"entity": "Type", "subentity": "Species", "value": ["Stylonectria applanata", "Höhn. 1915."]}
```

**Mistral-7B bf16**
*ok*
```json
{"Type": ["Stylonectria applanata", "Höhn. 1915."], "Stroma": ["Thin", "Whitish or yellow", "Hyphal or subiculum-like"], "Perithecia": ["Gregarious in groups", "Up to 20", "Subglobose", "Pyriform to subcylindrical", "With a rounded or broad, circular, flat disc on a venterlike neck", "Sometimes laterally collapsing when dry", "Pale yellow", "Orange-red", "Orange-brown", "Pale to dark red", "KOH+ dark red to purple", "Yellow in lactic acid", "Smooth", "Usually shiny", "Slightly iridescent", "150–250(–350) μm high"], "Perithecial Wall": ["Consisting of two regions", "Inner region of hyaline, thin-walled, compressed, elongate cells", "Outer region of distinct, isodiametric to oblong, angular or
```

**Qwen2.5-1.5B bf16**
```json
{"species": {"name": "Stylonectria applanata", "author": "Höhn. 1915"}, "stroma": ["thin", "whitish or yellow", "hyphal or subiculum-like"], "perithecia": ["gregarious in groups of up to 20", "subglobose", "pyriform to subcylindrical", "with a rounded or broad, circular, flat disc on a venterlike neck", "sometimes laterally collapsing when dry", "pale yellow", "orange-red", "orange-brown", "or pale to dark red", "KOH+ dark red to purple", "yellow in lactic acid", "smooth", "usually shiny", "slightly iridescent", "150–250(–350) μm high"], "perithecial_wall": ["two regions", "inner region of hyaline, thin-walled, compressed, elongate cells", "outer region of distinct, isodiametric to oblong, a
```

**Qwen2.5-7B 4-bit**
```json
{"Type species": {"Stylonectria applanata": {"Höhn.": ["1915"], "Stroma": {"description": ["thin, whitish or yellow, hyphal or subiculum-like"]}, "Perithecia": {"description": ["gregarious in groups of up to 20, subglobose, pyriform to subcylindrical, with a rounded or broad, circular, flat disc on a venterlike neck, sometimes laterally collapsing when dry", "pale yellow, orange-red, orange-brown, or pale to dark red", "KOH+ dark red to purple, yellow in lactic acid", "smooth, usually shiny, slightly iridescent", "150–250(–350) μm high"], "wall": {"regions": [{"inner": ["hyaline, thin-walled, compressed, elongate cells"], "outer": ["distinct, isodiametric to oblong, angular or globose, thick
```

**Phi-3.5-mini bf16**
```json
{"Species": "Stylonectria applanata", "YearOfDescription": "1915", "TaxonomicFeatures": {"Stroma": {"Color": ["thin", "whitish", "yellow"], "Texture": ["hyphal", "subiculum-like"]}, "Perithecia": {"Grouping": ["gregarious"], "Shape": ["subglobose", "pyriform", "subcylindrical"], "Venter": ["rounded", "circular", "flat"], "DryState": ["laterally collapsing"], "Color": ["pale yellow", "orange-red", "orange-brown", "pale to dark red", "KOH+ dark red to purple", "yellow in lactic acid"]}, "PerithecialWall": {"InnerRegion": ["hyaline", "thin-walled", "compressed", "elongate"], "OuterRegion": ["distinct", "isodiametric", "oblong", "angular", "globose", "thick-walled"]}, "Asci": {"Shape": ["cylindr
```

### 4. `taxon_bd737ea38aff53…`

**Description** (1294 chars) — Saprobic on submerged decaying wood. Asexual morph: Colonies superficial, effuse, hairy, scattered, dark brown, with subhyaline, fusiform conidia on the apex of conidiophores. Mycelium mostly immersed, composed of septate, pale brown, smooth hyphae. Conidiophores (70–) 130–230× 3.2–8.4 µm (x̅ = 153.4 × 4.9 µm, n = 15), mononematous, macronematous, erect, unbranched, cylindrical, straight or slightly flexuous, septate, smooth, thick-walled, dark brown at the base, and gradually paler towards the apex. Conidiogenous cells polyblastic, integrated, terminal, sympodially proliferating, subhyaline to pale brown. Conidia (10–) 25–50 × 3–8.5 µm (x̅ = 38.5 × 5.6 µm, n = 40), acrogenous, mostly fusifo

**Mistral-7B 4-bit**
```json
{"Species": [{"Saprobic_on": ["submerged decaying wood"], "Asexual_morph": {"Colonies": ["superficial", "effuse", "hairy", "scattered", "dark brown"], "Conidia": ["10–25–50 × 3–8.5 µm (x̅ = 38.5 × 5.6 µm, n = 40)", "acrogenous", "fusiform", "clavate", "solitary", "straight or slightly curved", "7–8-septate when mature", "septum darker", "smooth", "subhyaline to pale brown", "with a hilum at the base"], "Mycelium": {"Composition": ["septate", "pale brown", "smooth hyphae"], "Immersed": true}, "Conidiophores": ["70–130–230×3.2–8.4 µm (x̅ = 153.4 × 4.9 µm, n = 15)", "mononematous", "macronematous", "erect", "unbranched", "cylindrical", "straight or slightly flexuous", "septate", "smooth", "thic
```

**Mistral-7B bf16**
```json
{"Species": [{"Saprobic Habitat": ["Saprobic on submerged decaying wood"], "Asexual Morph": {"Colony": ["superficial", "effuse", "hairy", "scattered", "dark brown"], "Mycelium": ["immersed", "septate", "pale brown", "smooth"], "Conidiophore": ["mononematous", "macronematous", "erect", "unbranched", "cylindrical", "straight or slightly flexuous", "septate", "smooth", "thick-walled", "dark brown at the base", "gradually paler towards the apex", "70–130–230µm", "3.2–8.4µm"], "Conidiogenous Cell": ["polyblastic", "integrated", "terminal", "sympodially proliferating", "subhyaline to pale brown"], "Conidium": ["acrogenous", "fusiform", "clavate", "solitary", "straight or slightly curved", "7–8-sep
```

**Qwen2.5-1.5B bf16**
```json
{"type": "Colonies superficial, effuse, hairy, scattered, dark brown, with subhyaline, fusiform conidia on the apex of conidiophores."}
```

**Qwen2.5-7B 4-bit**
```json
{"Taxonomic Features": {"Morphology": {"Colonies": {"Description": ["superficial", "effuse", "hairy", "scattered", "dark brown"]}, "Conidia": {"Description": ["subhyaline", "fusiform"], "Dimensions": ["(10–) 25–50 × 3–8.5 µm (x̅ = 38.5 × 5.6 µm, n = 40)"], "Shape": ["acrogenous", "mostly fusiform", "sometimes clavate"], "Position": ["solitary", "straight or slightly curved"], "Septation": ["7–8-septate when mature", "septum darker, sometimes slightly constricted at the septum"], "Surface": ["smooth", "subhyaline to pale brown"], "Hilum": ["with a hilum at the base"]}, "Conidiophores": {"Dimensions": ["(70–) 130–230×3.2–8.4 µm (x̅ = 153.4 × 4.9 µm, n = 15)"], "Structure": ["mononematous", "ma
```

**Phi-3.5-mini bf16**
```json
{"Type": "Polyblastic", "Location": "Terminal", "Proliferation": "Sympodially"}
```

### 5. `taxon_21ea9b1fe58469…`

**Description** (553 chars) — Type. Geastrum hungaricum Hollós. Diagnosis — Exoperidium more or less saccate to arched when hydrated, strongly hygrometric (pseudoparenchymatous layer with thick-walled cells). Mycelial layer encrusting debris, doublelayered. Endoperidial body sessile. Endoperidial surface gla- brous and covered by a well-developed mesoperidium, composed by generative hyphae and abundant crystalline aggregates of COM, with scarce bipyramidal crystals of COD. Peristome fibrillose and distinctly delimited. Geastrum hungaricum Hollós, Math. Termeszettud. Ertes. 19:

**Mistral-7B 4-bit**
```json
{"Type": {"ScientificName": ["Geastrum hungaricum", "Hollós"], "CommonName": []}, "Diagnosis": {"Exoperidium": {"Description": ["more or less saccate to arched when hydrated", "strongly hygrometric (pseudoparenchymatous layer with thick-walled cells)"]}, "MycelialLayer": {"Description": ["Mycelial layer encrusting debris", "double-layered"]}, "EndoperidialBody": {"Position": ["sessile"]}, "EndoperidialSurface": {"Texture": ["glabrous"], "Coverage": ["covered by a well-developed mesoperidium"], "Components": {"GenerativeHyphae": [], "CrystallineAggregates": ["COM"], "BipyramidalCrystals": ["COD"], "ScarceCrystals": []}}, "Peristome": {"Texture": ["fibrillose"], "Delimitation": ["distinctly de
```

**Mistral-7B bf16**
```json
(no JSON extracted)
```

**Qwen2.5-1.5B bf16**
```json
{"species": {"name": "Geastrum hungaricum", "author": "Hollós"}, "features": [{"type": "exoperidium", "description": "more or less saccate to arched when hydrated, strongly hygrometric (pseudoparenchymatous layer with thick-walled cells)"}, {"type": "mucoraceous mycelial layer", "description": "encrusting debris, double-layered"}, {"type": "endoperidial body", "description": "sessile"}, {"type": "endoperidial surface", "description": "glabrous and covered by a well-developed mesoperidium, composed by generative hyphae and abundant crystalline aggregates of COM, with scarce bipyramidal crystals of COD."}, {"type": "peristome", "description": "fibrillose and distinctly delimited"}]}
```

**Qwen2.5-7B 4-bit**
```json
{"Type": ["Geastrum hungaricum Hollós"], "Diagnosis": {"Exoperidium": ["more or less saccate to arched when hydrated, strongly hygrometric (pseudoparenchymatous layer with thick-walled cells)"], "Mycelial layer": ["encrusting debris, double-layered"], "Endoperidial body": ["sessile"], "Endoperidial surface": ["gla-brous and covered by a well-developed mesoperidium, composed by generative hyphae and abundant crystalline aggregates of COM, with scarce bipyramidal crystals of COD"]}, "Peristome": ["fibrillose and distinctly delimited"]}
```

**Phi-3.5-mini bf16**
```json
{"Species": "Geastrum hungaricum", "Author": "Hollós", "Publication": "Termeszettud. Ertes.", "Year": "19", "TaxonomicFeatures": {"Exoperidium": {"Description": "more or less saccate to arched when hydrated", "Characteristics": {"Hygrometric": "strongly", "CellStructure": "pseudoparenchymatous layer with thick-walled cells"}}, "MycelialLayer": {"Description": "encrusting debris, doublelayered"}, "EndoperidialBody": {"Description": "sessile"}, "EndoperidialSurface": {"Description": "gla-grous"}, "Mesoperidium": {"Description": "composed by generative hyphae and abundant crystalline aggregates of COM, with scarce bipyramidal crystals of COD", "Components": {"GenerativeHyphae": "abundant", "COM
```

### 6. `taxon_b5b9b53db86d27…`

**Description** (347 chars) — is a trademark of AFRS, macroscopically defined as a thick, tenacious mucus, highly viscous in consistency and light tan to brown (peanut-butter-like appearance) or dark green. Microscopically, it is characterized by lamellated aggregates of dense inflammatory cells, mainly eosinophils and Charcot–Leyden crystals, and fungal hyphae (Dematiaceous

**Mistral-7B 4-bit**
```json
{"Species": [{"Description": "Trademark of AFRS", "Taxonomic_Classification": [{"Classification_Level": "Species", "Classification_Name": "AFRS"}]}, {"Description": "macroscopically defined as a thick, tenacious mucus", "Macroscopic_Characteristics": [{"Characteristic_Name": "Consistency", "Characteristic_Value": ["thick", "tenacious"]}, {"Characteristic_Name": "Color", "Characteristic_Value": ["light tan", "brown", "dark green", "peanut-butter-like appearance"]}]}, {"Description": "Microscopically, it is characterized by lamellated aggregates of dense inflammatory cells, mainly eosinophils and Charcot–Leyden crystals, and fungal hyphae (Dematiaceous)", "Microscopic_Characteristics": [{"Char
```

**Mistral-7B bf16**
```json
{"Species": ["AFRS"], "Macroscopic_Characteristics": [{"Consistency": ["Thick"], "Color": ["Light tan", "Brown", "Dark green"], "Appearance": ["Peanut-butter-like"]}], "Microscopic_Characteristics": [{"Cell_Aggregates": [{"Type": "Lamellated", "Cell_Type": ["Inflammatory cells", "Eosinophils", "Charcot–Leyden crystals"]}], "Fungal_Hyphae": ["Dematiaceous"]}]}
```

**Qwen2.5-1.5B bf16**
```json
(no JSON extracted)
```

**Qwen2.5-7B 4-bit**
```json
{"description": ["is a trademark of AFRS, macroscopically defined as a thick, tenacious mucus, highly viscous in consistency and light tan to brown (peanut-butter-like appearance) or dark green"], "macroscopic": {"appearance": ["thick, tenacious mucus", "highly viscous in consistency", "light tan to brown (peanut-butter-like appearance)", "dark green"]}, "microscopic": {"characteristics": ["lamellated aggregates of dense inflammatory cells", "mainly eosinophils and Charcot–Leyden crystals", "and fungal hyphae (Dematiaceous)"]}}
```

**Phi-3.5-mini bf16**
```json
{"Species": "Unknown Species", "Macroscopic Features": {"Mucus Consistency": ["thick", "tenacious", "highly viscous"], "Color": ["light tan", "brown (peanut-butter-like appearance)"], "Microscopic Features": {"Cell Aggregates": ["dense inflammatory cells", "mainly eosinophils"], "Crystals": ["Charcot–Leyden crystals"], "Fungal Hyphae": ["present"]}}, "Trademark": "AFRS", "Description Source": "Dematiaceous"}
```

### 7. `taxon_1416e7dffbe81f…`

**Description** (1004 chars) — Description. Sexual morph: Unknown. Asexual morph: Conidiomata pycnidial, aggregated or solitary, immersed under the host bark, subglobose to pulvinate, yellow to orange, 500–1200 μm wide, 150–450 μm high, multilocular, single ostiolate, forming long orange tendrils. Conidiophores cylindrical, aseptate, hyaline, sometimes reduced to conidiogenous cells. Conidiogenous cells lining inner cavity of conidiomata, phialidic, ampulliform, with attenuated or truncate apices, hyaline, smooth, 12.8–25.7 × 1.7–3.2 μm (n = 50). Conidia dimorphic. Microconidia minute, aseptate, hyaline, smooth, cylindrical, straight, (3.1–)3.3–4(–4.4) × (1.5–)1.6–2(–2.1) μm (n = 50), L/W = 1.6–2.7. Macroconidia aseptate,

**Mistral-7B 4-bit**
```json
{"Species": [{"Conidiomata": [{"Shape": ["pycnidial"], "Location": ["under the host bark"], "Color": ["yellow", "orange"], "Size": ["500–1200 μm wide", "150–450 μm high"]}]}, {"Conidiomata_Characteristics": [{"Shape": ["subglobose", "pulvinate"], "Color": ["yellow", "orange"], "Formation": ["multilocular", "single ostiolate", "forming long orange tendrils"]}]}, {"Conidiophore": [{"Shape": ["cylindrical"], "Septation": ["aseptate"], "Color": ["hyaline"], "Additional_Characteristics": ["sometimes reduced to conidiogenous cells"]}]}, {"Conidiogenous_Cells": [{"Shape": ["phialidic", "ampulliform"], "Apex": ["attenuated", "truncate"], "Color": ["hyaline"], "Surface": ["smooth"], "Size": ["12.8–25
```

**Mistral-7B bf16**
```json
{"Species": [{"Conidiomata": {"Type": "pycnidial", "Location": ["host bark"], "Shape": ["subglobose", "pulvinate"], "Color": ["yellow", "orange"], "Size": ["500–1200 μm wide", "150–450 μm high"]}, "Conidiomata_Characteristics": {"Locations": ["inner cavity"], "Shape": ["multilocular"], "Ostioles": ["single"], "Tendrils": ["forming long orange tendrils"]}, "Conidiophores": {"Shape": ["cylindrical"], "Septation": ["aseptate"], "Color": ["hyaline"], "Additional_Forms": ["reduced to conidiogenous cells"]}, "Conidiogenous_Cells": {"Shape": ["phialidic", "ampulliform"], "Apex": ["attenuated", "truncate"], "Color": ["hyaline"], "Smoothness": ["smooth"], "Size": ["12.8–25.7 × 1.7–3.2 μm (n = 50)"]},
```

**Qwen2.5-1.5B bf16**
```json
(no JSON extracted)
```

**Qwen2.5-7B 4-bit**
```json
{"Status": {"Unknown": ["Unknown"]}}
```

**Phi-3.5-mini bf16**
```json
{"Species": {"SexualMorph": "Unknown", "AsexualMorph": {"Conidiomata": {"Type": "pycnidial", "Arrangement": ["aggregated", "solitary"], "Location": "immersed under host bark", "Shape": "subglobose to pulvinate", "Color": ["yellow", "orange"], "Dimensions": {"Width": "500–1200 μm", "Height": "150–450 μm"}, "Structure": "multilocular", "Ostiolation": "single", "Tendrils": "long orange"}, "Conidiophores": {"Shape": "cylindrical", "Division": "aseptate", "Color": "hyaline", "Reduction": "to conidiogenous cells"}, "ConidiogenousCells": {"Location": "lining inner cavity of conidiomata", "Shape": "phialidic", "Apices": ["attenuated", "truncate"], "Texture": "smooth", "Size": {"Length": "12.8–25.7 μ
```

### 8. `taxon_b461a1b2e0fb8b…`

**Description** (1418 chars) — Description. Sexual morph not observed. Asexual morph on SNA. Vegetative mycelium hyaline, smooth-walled, septate, branched, 1–5 µm wide, chlamydospores absent. Sporulation abundant, conidia formed on hyphal cells. Conidiophores hyaline, smooth-walled, septate, often densely branched, up to 50 µm long. Conidiogenous cells enteroblastic, hyaline, smooth-walled, flask-shaped, 4.5–11.5 × 2.5–3.5 µm µm, collarettes narrowly funnel-shaped, 1.5–2 µm long, 1–1.5 µm wide at the upper edge, opening 0.5–1 µm, periclinal thickening sometimes observed. Conidia aggregated in heads, hyaline, smooth-walled, aseptate, subglobose, ovoidal, ellipsoidal to elongate-ellipsoidal, straight, with both ends rounded

**Mistral-7B 4-bit**
```json
{"Species": [{"Vegetative Mycelium": {"Description": ["hyaline", "smooth-walled", "septate", "branched", "1–5 µm wide", "chlamydospores absent"]}, "Asexual Morph": {"Location": ["SNA"]}, "Conidia": {"Shape": ["subglobose", "ovoidal", "ellipsoidal", "elongate-ellipsoidal"], "Size": {"Length": ["(3.5–)4–6(–9)"], "Width": ["2–2.5(–3) µm"]}, "Additional Features": ["straight", "both ends rounded"], "Sporulation Location": ["inside the medium"]}, "Conidiophores": {"Description": ["hyaline", "smooth-walled", "septate", "often densely branched", "up to 50 µm long"]}, "Conidiogenous Cells": {"Shape": ["flask-shaped"], "Size": {"Length": ["4.5–11.5"], "Width": ["2.5–3.5 µm"]}, "Collarettes": {"Shape"
```

**Mistral-7B bf16**
```json
{"Species": [{"Asexual Morph": [{"Location": "SNA", "Morphology": "present"}], "Vegetative Mycelium": [{"Color": ["hyaline"], "Wall": ["smooth-walled"], "Septate": ["present"], "Branched": ["present"], "Width": ["1–5 µm"]}, {"Chlamydospores": ["absent"]}], "Sporulation": [{"Abundance": ["abundant"]}], "Conidia": [{"Formation": ["on hyphal cells"], "Shape": ["subglobose", "ovoidal", "ellipsoidal", "elongate-ellipsoidal"], "Straightness": ["straight"], "Ends": ["both ends rounded"], "Size": [{"Width": ["2–2.5(–3) µm"], "Length": ["(3.5–)4–6(–9) µm"]}, {"Maximum": ["15 × 2.5 µm"]}], "Mean Size": {"Width": "2.2 ± 0.3 µm", "Length": "4.9 ± 1.2 µm", "L/W ratio": "2.2"}}], "Conidiophores": [{"Color
```

**Qwen2.5-1.5B bf16**
```json
{"species": {"name": "Not specified", "description": "Sexual morph not observed. Asexual morph on SNA. Vegetative mycelium hyaline, smooth-walled, septate, branched, 1–5 µm wide, chlamydospores absent. Sporulation abundant, conidia formed on hyphal cells. Conidiophores hyaline, smooth-walled, septate, often densely branched, up to 50 µm long. Conidiogenous cells enteroblastic, hyaline, smooth-walled, flask-shaped, 4.5–11.5 × 2.5–3.5 µm, collarettes narrowly funnel-shaped, 1.5–2 µm long, 1–1.5 µm wide at the upper edge, opening 0.5–1 µm, periclinal thickening sometimes observed. Conidia aggregated in heads, hyaline, smooth-walled, aseptate, subglobose, ovoidal, ellipsoidal to elongate-ellipso
```

**Qwen2.5-7B 4-bit**
```json
{"sexual_morph": {"observed": ["not observed"]}, "asequential_morph": {"sporulation": {"abundance": ["abundant"]}, "conidia": {"formed_on": ["hyphal cells"]}}, "vegetative_mycelium": {"characteristics": {"colour": ["hyaline"], "wall": ["smooth-walled", "septate"], "branching": ["branched"], "width": ["1–5 µm"]}}, "chlamydospores": {"presence": ["absent"]}, "conidiophores": {"characteristics": {"colour": ["hyaline"], "wall": ["smooth-walled", "septate"], "branching": ["densely branched"], "length": ["up to 50 µm"]}}, "conidiogenous_cells": {"shape": ["enteroblastic", "hyaline", "smooth-walled", "flask-shaped"], "dimensions": ["4.5–11.5 × 2.5–3.5 µm"], "collarette": {"shape": ["narrowly funnel
```

**Phi-3.5-mini bf16**
```json
{"Observation": "Not observed"}
```

### 9. `taxon_d09e1082069647…`

**Description** (507 chars) — Description – Hyphae branched, septate; fertile parts of hyphae aerial, lateral or terminal, with short lateral secondary branches consisting of subglobose cells. Sexual morph: Not observed. Asexual morph: Conidiophores are usually reduced to conidiogenous cells. Conidiogenous cells arise in clusters on intercalary or terminal, swollen, subglobose to inequilaterally ampulliform, cupshaped, like tuberculiform. Conidia solitary or in chains, semicircular or ellipsoidal to subglobose, with truncate bases.

**Mistral-7B 4-bit**
```json
{"Hyphae": {"Branched": ["true"], "Septate": ["true"]}, "Fertile parts of hyphae": {"Type": ["aerial"], "Location": ["lateral", "terminal"], "Secondary branches": {"Type": ["short lateral"], "Cells": ["subglobose"]}}, "Conidiophores": {"Form": ["reduced"], "Function": ["conidiogenous cells"]}, "Conidiogenous cells": {"Arise": ["in clusters"], "Location": ["intercalary", "terminal"], "Shape": ["swollen", "subglobose", "inequilaterally ampulliform", "cupshaped", "tuberculiform"]}, "Conidia": {"Number": ["solitary", "chains"], "Shape": ["semicircular", "ellipsoidal", "subglobose"], "Base": ["truncate"]}}
```

**Mistral-7B bf16**
```json
{"Hyphae": {"Branched": ["true"], "Septate": ["true"]}, "Fertile parts of hyphae": {"Location": ["aerial", "lateral", "terminal"]}, "Fertile parts of hyphae - Lateral secondary branches": {"Cells": ["subglobose"]}, "Sexual morph": {"Observed": ["false"]}, "Asexual morph": {"Conidiophores": {"Form": ["reduced", "conidiogenous cells"]}, "Conidiogenous cells": {"Arise": ["in clusters"], "Location": ["intercalary", "terminal"], "Shape": ["swollen", "subglobose to inequilaterally ampulliform", "cupshaped", "like tuberculiform"]}, "Conidia": {"Formation": ["solitary", "in chains"], "Shape": ["semicircular", "ellipsoidal", "subglobose"], "Base": ["truncate"]}}}
```

**Qwen2.5-1.5B bf16**
```json
{"species": {"name": "Not specified", "description": "Hyphae branched, septate; fertile parts of hyphae aerial, lateral or terminal, with short lateral secondary branches consisting of subglobose cells. Sexual morph: Not observed. Asexual morph: Conidiophores are usually reduced to conidiogenous cells. Conidiogenous cells arise in clusters on intercalary or terminal, swollen, subglobose to inequilaterally ampulliform, cupshaped, like tuberculiform. Conidia solitary or in chains, semicircular or ellipsoidal to subglobose, with truncate bases."}, "taxonomic_features": [{"level_1_entity": "hyphae", "level_2_subentity": "branching", "value": "branched"}, {"level_1_entity": "hyphae", "level_2_sub
```

**Qwen2.5-7B 4-bit**
```json
{"Hyphae": {"Branching": ["branched"], "Septation": ["septate"]}, "Fertile Parts": {"Location": ["aerial", "lateral", "terminal"], "Structure": ["short lateral secondary branches consisting of subglobose cells"]}, "Sexual Morph": {"Observation": ["Not observed"]}, "Asexual Morph": {"Conidiophores": {"Character": ["usually reduced to conidiogenous cells"]}, "Conidiogenous Cells": {"Origin": ["in clusters on intercalary or terminal, swollen, subglobose to inequilaterally ampulliform, cup-shaped, like tuberculiform"]}, "Conidia": {"Arrangement": ["solitary or in chains"], "Shape": ["semicircular or ellipsoidal to subglobose"], "Base": ["truncate bases"]}}}
```

**Phi-3.5-mini bf16**
```json
{"Hyphae": {"Branched": [], "Septate": []}, "Fertile Parts": {"Location": ["Aerial", "Lateral", "Terminal"], "Structure": {"Secondary Branches": {"Shape": ["Short lateral secondary branches"], "Cell Type": ["Subglobose"]}}}, "Sexual Morph": {"Observed": []}, "Asexual Morph": {"Conidiophores": {"Reduced": ["Conidiogenous cells"], "Location": ["Intercalary", "Terminal", "Swollen", "Subglobose", "Inequilaterally ampulliform", "Cupshaped", "Like tuberculiform"]}, "Conidiogenous Cells": {"Arrangement": ["Clusters"], "Shape": ["Subglobose", "Inequilaterally ampulliform", "Cupshaped", "Like tuberculiform"], "Conidia": {"Number": ["Solitary", "In chains"], "Shape": ["Semicircular", "Ellipsoidal", "S
```

### 10. `taxon_53a25d37a81344…`

**Description** (858 chars) — Ceratocystis musarum H. Ried], sp. n. Perithecia semi-immersa vel sublibera, globosa vel paulo depressa, 330 u fere diam. vel 300 u alta, 350 u lata, rostro 1100—1200 u longo, prope basin circ. 50 u, ostiolum versus ad circ. 17—18 u attenuato; paries peritheciorum brunneus, 14,5—30 u crassus, in speciminibus diversis crassitudine valde variabili, plectenchymaticus, stratis 5—7 hypharum formatus, hyphis laxis, brunneis, terminaliter rotundatis in parte libera densissime obtectus; hyphae rostrum formantes parallelae, apice plerumque penicillatae. Asci fugacissimi, obovato-saccati usque subglobosi, stipitati. Sporae hyalinae, cylindraceae, plerumque rectae, 6—11 p longae, 2—3,5 u crassae. Vindo

**Mistral-7B 4-bit**
```json
{"ScientificName": "Ceratocystis musarum", "Author": "H. Ried"}
```

**Mistral-7B bf16**
```json
(no JSON extracted)
```

**Qwen2.5-1.5B bf16**
```json
{"species": {"name": "Ceratocystis musarum", "author": "H. Ried", "sp_name": "n.", "perithecia": [{"type": "semi-immersa vel sublibera", "shape": "globosa vel paulo depressa"}], "size": [{"length": "330 u fere diam.", "height": "300 u alta", "width": "350 u lata"}, {"rostral_length": "1100—1200 u longo", "basin_circumference": "prope basin circ. 50 u", "ostiolum_attenuation": "ostiolum versus ad circ. 17—18 u attenuato"}], "perithecial_surface": [{"color": "brunneus", "crassity": "14,5—30 u crassus", "variability": "diversis crassitudine valde variabili", "plectenchymaticus": true, "stratum_formation": "stratis 5—7 hypharum formatus", "hyphal_structure": "hyphis laxis, brunneis, terminaliter
```

**Qwen2.5-7B 4-bit**
```json
{"Ceratocystis": {"musarum": {"H. Ried": {"sp. n": {}, "Perithecia": {"semi-immersa vel sublibera": {}, "globosa vel paulo depressa": {}, "330 μ fere diam. vel 300 μ alta": {}, "350 μ lata": {}, "rosto": {"1100—1200 μ longo": {}}, "prope basin": {"circ. 50 μ": {}}, "ostiolum versus ad": {"circ. 17—18 μ attenuato": {}}, "paries peritheciorum": {"brunneus": {}, "14,5—30 μ crassus": {}, "in speciminibus diversis crassitudine valde variabili": {}, "plectenchymaticus": {}, "stratis 5—7 hypharum formatus": {}, "hyphis laxis": {}, "brunneis": {}, "terminaliter rotundatis in parte libera": {"densissime obtectus": {}}}, "hyphae": {"rostrum formantes": {"parallelae": {}, "apice plerumque penicillatae"
```

**Phi-3.5-mini bf16**
```json
{"Species": {"Name": "Ceratocystis musarum H. Ried", "Description": {"Perithecia": {"Shape": ["semi-immersa", "sublibera", "globosa", "paulo depressa"], "Size": {"Diameter": ["330 u", "300 u"], "Height": ["350 u", "longo"], "Rostrum": ["1100—1200 u"]}, "Ostiolum": {"Position": ["prope basin circ. 50 u"], "Attenuation": ["circ. 17—18 u"]}, "Paries": {"Color": ["brunneus"], "Thickness": ["14,5—30 u crassus"], "Variation": ["diversis crassitudine valde variabili"], "Plectenchymaticus": true, "Hypharum": {"Formation": ["formatus"], "Hyphae": {"Rostrum": ["formantes parallelae"], "Apice": ["plerumque penicillatae"]}}}, "Asci": {"Shape": ["obovato-saccati", "subglobosi"], "Stipitation": true}, "Sp
```


## R7 — stop at the closing brace

| arm | whole | whole_512 | clause_seq | clause_batch | cap-hits |
|---|---:|---:|---:|---:|---:|
| stop_on_json=off | 56.00 | 38.91 | 69.79 | 35.66 | 0.20 |
| stop_on_json=on | 56.12 | 39.01 | 68.80 | 35.62 | 0.20 |

**Cap-hits is the honest check**: if it does not collapse toward zero with the flag on, the criterion is not firing whatever the clock says.


## What is NOT answered here

* **No ground truth.** R4's gold set does not exist, so nothing below is scored — your eye is the only instrument.
* **Clause outputs not shown.** 171 per model; the whole-description arm is what the pipeline runs today.
* **Key vocabulary is uncontrolled** in every model. R5 is the fix; the counts above are the size of the problem.

