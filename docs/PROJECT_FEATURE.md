# SKOL Projects

## Purpose

A Project groups multiple collections into a larger named unit — for example,
all the collections that will appear in a regional nature guide, or a thematic
checklist of a particular family.  Collections may belong to any number of
projects simultaneously.

---

## Governance and permissions

- **Anyone** can create a project.
- **Anyone** can add any collection to any project.
- **Anyone** can remove any collection from any project.
- The **creator** of a project is recorded permanently in the project record and
  has no special operational powers beyond that attribution.
- **Only admins** can delete a project record entirely.
- All add and remove events are recorded in a permanent audit log (see Data
  Model below).
- There are no lifecycle states (draft / active / archived).  Projects exist
  or they don't.
- All projects are **public** by default.  Privacy controls, ownership
  transfers, and per-project permissions are deferred until the public model
  is no longer sufficient (generalize-on-second-use).

---

## Data model

### `Project`

| Field | Type | Notes |
|---|---|---|
| `id` | UUID (system-generated) | Internal primary key; never exposed in URLs |
| `name` | CharField | Human-readable; must contain at least one alphanumeric character |
| `slug` | SlugField | Auto-generated from `name`; unique per user (see Namespacing) |
| `creator` | FK → User | Permanently recorded; never changes |
| `description` | TextField | Optional free-text description |
| `created_at` | DateTimeField | Auto-set on creation |

### `CollectionProject` (through table)

Many-to-many between Collection and Project, with audit fields.

| Field | Type | Notes |
|---|---|---|
| `collection` | FK → Collection | |
| `project` | FK → Project | |
| `added_by` | FK → User | Who added this membership |
| `added_at` | DateTimeField | When it was added |

Unique constraint on `(collection, project)`.

### `CollectionProjectRemoval` (audit log for removals)

| Field | Type | Notes |
|---|---|---|
| `collection` | FK → Collection | |
| `project` | FK → Project | |
| `removed_by` | FK → User | |
| `removed_at` | DateTimeField | |
| `added_by` | FK → User | Copied from the through-table row at removal time |
| `added_at` | DateTimeField | Copied from the through-table row at removal time |

The removal table provides a permanent record of every membership that has
ever existed, including who originally added it.

---

## Namespacing and slugs

Project slugs are scoped to the creating user.  Two different users may each
have a project whose slug is `french-guiana-fungi`; they are distinguished by
`(creator.username, slug)`.

**Slug generation algorithm:**

1. Lowercase the project name.
2. Replace runs of non-alphanumeric characters with `-`.
3. Strip leading and trailing `-`.
4. If the result contains no alphanumeric character, reject the name and
   prompt the user for a different one.
5. Check uniqueness within the creator's namespace.  On collision (same user
   creates "French Guiana Fungi" twice), append `-2`, `-3`, etc.

**URL representation** uses `username/slug` with a literal `/` separator,
e.g. `?project=jsmith/french-guiana-fungi`.  This is unambiguous and stable:
the slug half never changes, and usernames are separated by a character that
cannot appear in a slug.

---

## URL and search integration

### Query parameter

Projects are referenced in URLs as:

```
?project=jsmith/french-guiana-fungi
```

Multiple `?project=` parameters are supported with **OR semantics**: results
include collections that belong to any of the listed projects.

```
?project=jsmith/french-guiana-fungi&project=mjones/caribbean-guide
```

AND semantics (intersection) are not supported in the initial implementation
but may be added later via `?project_op=and` without changing the default
behaviour.

### Search facet

Project membership is a search facet on the main search page.  All projects
are shown in the facet panel regardless of whether the current result set
contains them.  Selecting a project from the facet applies the `?project=`
parameter and re-runs the search.

Selecting a project from the project pulldown (see Discovery) pre-applies the
project facet and lands the user on the search page already scoped to that
project.  The project's effective "home page" is therefore the search page
with the facet pre-set.

### Search within a project

Entering a text query while a project facet is active searches within that
project.  No separate UI is required.

---

## Project discovery

A site-wide project pulldown lists all projects.  It is implemented as a call
to a dedicated `/api/projects/?q=` endpoint that initially returns all
projects.  When the list grows unwieldy, the endpoint can add pagination or
typeahead without changing the UI component.

Projects are searchable by name and by creator username.

---

## User settings: default projects

A user may specify a list of default projects in their profile settings.
When a new collection is created, it is automatically added to each project
in this list.  This setting applies only to collections created after the
setting is saved; it does not retroactively modify existing collections.

---

## UI placement

When a collection does not belong to any project, project controls appear as
an unobtrusive button on the View Details page (and optionally a compact
button in the Active Collection / Nomenclature box).  When a collection
already belongs to one or more projects, those project names are displayed
with remove controls alongside the add button.

---

## Export

### JSON (primary format)

A project export produces a JSON document containing:

- Project metadata (name, slug, creator, description, created date).
- Full collection records for every collection in the project, including all
  treatment sections (Nomenclature, Description, Diagnosis, Materials-examined,
  etc.) in the same structure as the "extract all my data" export.
- Audit log of add/remove events for the project.

The JSON is a complete round-trip: it can be imported into another SKOL
instance.

### Future formats (Trello items)

| Format | Scope | Notes |
|---|---|---|
| Darwin Core Archive (DwC-A) | Collections (specimens) | Materials-examined → Occurrence core; Nomenclature → Taxon extension.  Enables direct GBIF submission. |
| Plazi/TaxPub XML — collections | Collections | Full JATS-based treatment XML per collection document |
| Plazi/TaxPub XML — treatments | Extracted treatments | One treatment record per extracted taxon |

DwC-A exports a subset of SKOL data (specimen and taxon records); it does not
capture treatment structure (Description, Diagnosis, Biology, etc.).

---

## How to create your first project

This walkthrough creates a project called **"Brail Trail demo project"** and
puts a collection into it, starting from the Search page.

**Step 1 — Open a collection's detail page.**

On the Search page, find the collection you want to add to a project.
Click **View Details** next to it.  This opens the collection detail page.

**Step 2 — Find the Projects panel.**

Scroll to the bottom of the collection detail page, below the search history
and external identifiers.  You will see a **Projects** section.  If the
collection is not yet in any project it says *"Not in any project."*

**Step 3 — Type the new project name.**

Click inside the **"Add to project… or type to create"** box.  Start typing
`Brail Trail demo project`.  Because this project does not exist yet, a
suggestion appears at the bottom of the dropdown:

> **Create project "Brail Trail demo project"**

Click that suggestion.

**Step 4 — Done.**

SKOL creates the project and immediately adds this collection to it.  You will
see the project name appear as a tag in the Projects panel, and a green
confirmation message:

> *Created project "Brail Trail demo project" and added this collection.*

The project's slug will be `brail-trail-demo-project` under your username
(e.g. `jsmith/brail-trail-demo-project`).

---

**Adding more collections to the same project**

For each additional collection, open its detail page, click inside the
**"Add to project…"** box, and search for `Brail Trail`.  Select the project
from the list (it already exists now), then click **Add**.

---

**Finding your project on the Search page**

On the main Search page, look for the **Filter by project** bar just below the
search form.  Click it, type `Brail Trail`, and select your project.  All
searches will then be scoped to the collections in that project.

---

## Deferred / out of scope

- Project privacy, per-project permissions, ownership transfer — deferred
  until the public-by-default model is insufficient.
- iNat Traditional Project integration — deferred.  Collection-level
  integration with iNat does not yet exist; project-level bridging before
  collection-level bridging inverts the dependency.  iNat Traditional Projects
  use a curator-managed permission model that conflicts with SKOL's democratic
  model.  Revisit when a concrete use case is identified.
- Project lifecycle states (draft / active / archived) — not needed.
