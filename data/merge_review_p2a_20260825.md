# p2a merge-suspect review — verdicts

Stratified draw of 30 from the 7,632 treatments filtered out by
`n_terms_above_5 >= 10`.  Seed 20260825; ids in
`data/merge_review_p2a_20260825.txt`.

**The question for each: is this two or more treatments glued
together?**  Fill the `verdict` column with `merge`, `single`, or
`unsure`.  Dossiers: open [`p2a_dossiers/index.html`](p2a_dossiers/index.html)
and click a name, or click a name in the table below.  Hover any span
in a dossier to read its text; click a gap triangle to open it.

Deliverable once filled: precision of the metric as a function of
score, and a recommended threshold.  If precision in the 10–14 band
is poor, p1 is larger than 38,303 and every prior round sampled from
a frame smaller than it should have been.

| verdict | score | band | nom | desc | chars | binom | name |
|---|---:|---|---:|---:|---:|---|---|
| merge | 398 | >50 | 1 | 114 | 111,153 | Y | [in temp.; DK (oo). -A&N 97:131, Cet 2387, M&J Collybia 1](p2a_dossiers/taxon_25f858ca062272dddb2165fcc0b31f21e0ad694984b8611e6aac8bf914500544.html) | Looks like a key, at least at the start. Dozens of species.
| merge | 177 | >50 | 1 | 66 | 60,404 | Y | [Nomen ignotum](p2a_dossiers/taxon_b11c4c982ff357c212477f318fed36298a6be239eabdd9a103fe03c2aa20b233.html) | We have a genus description followed by a detailed key.
| merge | 69 | >50 | 1 | 7 | 12,058 |  | [Aspergillus asperescens Stolk, Antonie van Leeuwenhoek 2](p2a_dossiers/taxon_fd8c414d27344fd733392431d3b064af6f2caaf7d8ad8bdffc27017ee3da8703.html) | Multiple materials_examined misclassified as figure_caption. The diagnosis sections correctly name the species for their preceding descriptions.
| merge | 66 | >50 | 1 | 12 | 8,171 | Y | [Nomen ignotum](p2a_dossiers/taxon_c44f9f2647791b79a349b42080ca77259dcd8123c5ed0824281789a88a0afa21.html) | Almost every description is another species.
| merge | 65 | >50 | 1 | 10 | 16,725 | Y | [TRIANCUI.ARIA BAlofBUSM: (Beyma) .Bocdijn Trigonia bambu](p2a_dossiers/taxon_d831b03b3443cf7e0bbc85e2fbddb3f6446f4c600859f19f6eeaec7b79f125b3.html) | Massive OCR damage. Roughly every description is another species.
| merge | 42 | 15-50 | 1 | 4 | 5,993 | Y | [Nomen ignotum](p2a_dossiers/taxon_c9c812e46d84e8e5498e3cdf49ef610a0061ac2796da9c048e9e93ab9e1d7ead.html) | A massive list of authors and their addresses are under several different labels. Each description is a different genus.
| single | 25 | 15-50 | 1 | 2 | 4,326 |  | [Amanita brunneoumbonata Thongbai, Raspé & K.D. Hyde, sp.](p2a_dossiers/taxon_2d15ed00d380da16f80ffd6009a70cc0dcb35aa6082e19636d73b0f08596b9cf.html) | This is what a treatment should look like.
| single | 25 | 15-50 | 1 | 5 | 7,250 |  | [Nomen ignotum](p2a_dossiers/taxon_abadad5bf06ad054e9afff4939f1093ce712790c8c2278a669612930408db280.html) | Sections are very poorly identified. This article describes one species.
| merge | 20 | 15-50 | 2 | 6 | 4,593 |  | [Mycetinis ignobilis (Berk. & Broome) Desjardin & B.A. Pe](p2a_dossiers/taxon_426c764192c5a9933702d35fc55b2d2db4fff549abb3cd8e0381759162038b59.html) | 2 descriptions, identifiable by the two separate materials_examined. The gap (Misc-exposition) between the first two nomenclatures should have also been nomenclature. The next Misc-exposition should have been a type-designation. The remaining labels are pretty good.
| single | 20 | 15-50 | 1 | 3 | 4,594 | Y | [P. albostrigosa. 10. Pouzarella lasia (Berk. & Broome) L](p2a_dossiers/taxon_bcbca6e26de10ac6892cbf5e1dec42d44a4ff4a1bea33968a0ef3cd25446532e.html) | We have some paragraphs split between classification labels, but on the whole a pretty good treatment.
| merge | 20 | 15-50 | 1 | 4 | 2,391 | Y | [Nomen ignotum](p2a_dossiers/taxon_760c7d895cb910778f43e6723de1f3015b6b8169d97025de269559510b60599d.html) | This is a new genus and its type species.
| single | 18 | 15-50 | 2 | 1 | 790 |  | [Digitodochium anshunense K. Habib, Y.L. Ren & Q.R. Li, s](p2a_dossiers/taxon_efe693a5ce04678e4d21330cdeab8c9f28b7279f62461b7ac96a267ec810282c.html) | Misc-exposition blocks have absorbed parts of adjacent blocks. E.g. the first gap between etymology and description has the last line of etymology and the first line of description.
| merge | 17 | 15-50 | 1 | 2 | 2,363 | Y | [Gloeoporus variiformis Y.C. Dai, Chao G. Wang & Yuan Yua](p2a_dossiers/taxon_7b37dc1dddccae35f8e4b97bf550fbc70d82a3ae6aff81d068bd54caed1fcc43.html) | The treatment for Gloeoporus variiformis is almost perfect, but has the beginning of Meruliopsis Bondartsev stuck on the end.
| merge | 17 | 15-50 | 1 | 11 | 4,100 | Y | [This is Boletus vinaceus Frost Ms.](p2a_dossiers/taxon_18338ea5f118197de83538c38109e260ddb27dd5562f3c9c35aa53f7a844e2d5.html) | A whole bunch of boletes. Taxonomic citations have been consistently absorbed by Misc-exposition, Key, and Table blocks.
| merge | 15 | 15-50 | 1 | 6 | 4,695 | Y | [oleoides. Change manga, 25. III. 1950, Nr. 3129. — Auf d](p2a_dossiers/taxon_fdc837ce7490cf1d5d25f997933cdd74b1cb78ff1d5d974791b8f5b2eafa06ea.html) | German language with Latin. For papers of this era, Latin anywhere but in a description of diagnosis is almost certainly an error.
| single | 14 | 10-14 | 1 | 4 | 2,664 | Y | [Nomen ignotum](p2a_dossiers/taxon_25d9f5a1db9f1d27703c6974e771da080d380625a775d78536f04299531bda64.html) | This has the issue index prepended to a description of the genus Diabolidium gen. nov.
| single | 14 | 10-14 | 1 | 1 | 2,014 |  | [Samson, comb. nov. MycoBank MB809553. Fig. 10. Basionym:](p2a_dossiers/taxon_2f8232df55e35039f13118f7019a2fb0d5ea54161ca1b0e2172add84db759b5f.html) | Nomenclature is truncated above and below. The type designation, ITS Barcode references, and the first line of the description have been swallowed up by a figure_caption.
| single | 14 | 10-14 | 1 | 3 | 3,204 |  | [Boletellus emodensis (Berk.) Singer, Annls mycol. 40: 19](p2a_dossiers/taxon_4ce09139816f889436030d0b1750c0f448a8e901f073aaef78368a80c282f6b0.html) | This is an almost perfectly good treatment.
| single | 13 | 10-14 | 1 | 1 | 4,391 |  | [Pleonectria boothii Hirooka, Rossman & P. Chaverri, sp. ](p2a_dossiers/taxon_229f912ec5ae003a46e3c7cd7f01c1673a11ccfd6a8ded7b216b9d71a2c706b7.html) | Other than failing to detect the Etymology and Anamorph, this is a perfect treatment.
| single | 12 | 10-14 | 1 | 4 | 2,700 |  | [Phaeoramularia gomphrenicola CPC 23248. Phaeoramularia g](p2a_dossiers/taxon_8a18aefbc93e5614478b3e91ec021840a1a901028f6e224bd38e64ccd785e6a1.html) | We lst a few words of the description and one line of materials_examined to Misc-exposition blocks, otherwise perfect.
| merge | 12 | 10-14 | 1 | 3 | 3,637 |  | [Ascobolus castaneus Teng, Sinensia 11: 109 (1940)](p2a_dossiers/taxon_5aab72f2bc25a0fd87a1cf302cba0021a9892cbaf3c9f57df9a3358d1955a547.html) | Two descriptions concatenated. The nomenclature for the second one (Arnium hirtum) was lost to a notes section.
| merge | 12 | 10-14 | 1 | 2 | 2,950 |  | [Mallocybe from eastern North America](p2a_dossiers/taxon_ab3cb249f5fe0d17c69a12a6a6aa7c5b5c4f74a93f912aa22205382a2de6e46f.html) | Two similar species, Mallocybe tomentella, and Mallocybe tomentosula. Two diagnostic blocks were identified as Phylogeny
| single | 12 | 10-14 | 1 | 1 | 3,225 |  | [Sticta flakusiorum Ossowska, B. Moncada & Lücking sp. no](p2a_dossiers/taxon_01d4b61b8b4fb5ae7cd49b4837f6d660e74b033e27811ef66d12205cd090ad7e.html) | This example is perfect.
| merged | 12 | 10-14 | 1 | 1 | 556 |  | [Sistotremastrales ord. nov. (Basidiomycota). Mycosphere ](p2a_dossiers/taxon_a33e8dcbcc763fc3cfffa33ae6d23004138059b44e75774c492a1e4d2f58d418.html) | The paper introduces a new order. The multiple descriptions cover the type family, genus, and species.
| unsure | 12 | 10-14 | 1 | 3 | 4,689 | Y | [The genus Tubakia s. lat.](p2a_dossiers/taxon_19177a1cb6fa92fc9bce34cb96f3f4564f20a89b17ebabb67e11981d57f97b2e.html) | I think the two description sections are genus and type species, but they both have measurements and there appears to be context missing.
| single | 11 | 10-14 | 1 | 2 | 2,970 |  | [Cladosporium mucilaginosum C.M. Pereira & R.W. Barreto, ](p2a_dossiers/taxon_da1e42eb0998da5f75c9e0d7fd0b876fdcc4c1a2de5ee8a44588b9686b3193fb.html) | Pieces are missing. The first Misc-exposition is the last line of the etymology and the type designation. The last Misc-exposition is the end of a description and materials examined. The trailing diagnosis is truncated.
| single | 11 | 10-14 | 1 | 2 | 3,085 |  | [Polysphondylium paniculoides Y. Li, P. Liu et Y. Zou, sp](p2a_dossiers/taxon_8dd1330488e485be4db50455e47a135de0676f23eb8ab2fe66408caba0c34438.html) | Part of the taxonomic citation (Figures and MycoBank & GenBank id numbers) was consumed by a Figure-caption block. The distribution and ecology was classified as materials-examined. The next materials-examined is the correct one.
| single | 11 | 10-14 | 1 | 6 | 3,665 |  | [Podonectria kuwanaspidis X.L. Xu & C.L. Yang sp. nov. (F](p2a_dossiers/taxon_b160709fa5276e459adc244390b21cd4f5d0d23fa99ad982de248d03a3fdb9f5.html) | The different description sections seem to describe the same organism, though the line breaks are different.
| merged | 11 | 10-14 | 1 | 3 | 2,536 |  | [Nomen ignotum](p2a_dossiers/taxon_710585339d9c5920ad6db067d2a98fbe0f7c18e80cac9b1f0f61483bf94c02dd.html) | The article is a genus redescription. The treatment includes the description of (an unnamed) fungus in the genus, possibly the type species.
| single | 10 | 10-14 | 1 | 3 | 3,944 | Y | [Nomen ignotum](p2a_dossiers/taxon_5ec11f486be4aff32053757948d45aa8a1482d2834770a5a59169600a0354357.html) | Ramaria araiospora sp. nov. var. araiospora. Misc-exposition has absorbed adjacent lines...

## How to record a verdict

Edit **this file**.  Put `merge`, `single` or `unsure` in the empty
first column of each row, and add a note after the name if the case is
interesting.  Nothing else reads this file, so free text is fine.

## Column meanings

* **nom** — number of `nomenclature_spans`. More than one is close
  to decisive: two names means two treatments.
* **desc** — number of `description_spans`.
* **binom** — an authored binomial found *inside* the description,
  the other strong tell (a second taxon named in the prose).

## Before you start: what the draw already shows

* Only **2 of 30** have more than one nomenclature span.
* **13 of 30** carry an authored binomial in the description.
* The two highest scores are unmistakable — 398 with **114**
  description spans over 111,153 chars, and 177 with 66 spans over
  60,404 — the `taxon_2b793602` flora-chapter shape.
* The 10–14 band looks different: mostly 1 nomenclature span and
  ~3,000 chars, which is what an ordinary treatment looks like.
  That is the band the threshold decision rests on.
