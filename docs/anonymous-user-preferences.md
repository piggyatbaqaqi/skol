# Anonymous-user preferences — design sketch

## Problem

Anonymous (not-logged-in) users on the skol web UI can interact with
controls that *look* persistent — most visibly the experiment-selector
dropdown — but every page request goes through
`get_user_experiment()` which unconditionally returns `'production'`
for anonymous users.  The dropdown is a no-op for them.  Symptom: pick
a different experiment, page reloads, dropdown reverts to 'production'.

Beyond the immediate case, several future preferences would benefit
from anonymous-side persistence (favourites, recent searches, page
density, theme).  We want a single mechanism, not one ad-hoc story
per preference.

## Constraints

- **No per-visitor DB rows.**  Skol may eventually serve casual
  unauthenticated browsers — provisioning a User row per visit would
  bloat the user table and confuse downstream consumers that assume
  User implies "a person with a credential."
- **Multi-tenant per browser.**  Each browser instance gets its own
  preference state.  Several people sharing one machine see one
  shared state — same as Django's own session behavior for anon users.
- **Same shape for auth and anon paths.**  Code that reads "what's
  this user's active experiment?" should work identically for both
  kinds of caller, with the storage layer hidden.
- **No PII.**  Anonymous storage holds preferences only, never
  identifying information.

## Proposed approach: Django sessions with the signed-cookie backend

Django's session middleware already provides:
- A `request.session` dict-like API for both authenticated and
  anonymous users.
- Auto-creation on first write, expiry handling, integrity signing.
- A pluggable storage backend (DB, cache, file, signed cookies).

Set `SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'`
in `django/skolweb/settings.py`.  Effects:
- Session data lives in the `sessionid` cookie itself, HMAC-signed
  with `SECRET_KEY`.
- No `django_session` table writes.
- ~4 KB cookie size ceiling — generous for preference dicts.
- Browser-side state, server-side validation.

The change is one setting + a tiny refactor of `get_user_experiment`:

```python
def get_user_experiment(request):
    """Get experiment config for this request's user (auth or anon).

    Auth: reads UserSettings.default_experiment (DB-persistent).
    Anon: reads request.session['default_experiment'] (cookie-persistent).
    Falls back to 'production' in either case if no preference set.
    """
    experiment_name = 'production'
    if request.user.is_authenticated:
        try:
            from .models import UserSettings
            us = UserSettings.objects.filter(user=request.user).first()
            if us and us.default_experiment:
                experiment_name = us.default_experiment
        except Exception:
            pass
    else:
        experiment_name = request.session.get('default_experiment', 'production')

    doc = get_experiment_config(experiment_name)
    if doc is None and experiment_name != 'production':
        doc = get_experiment_config('production')
        experiment_name = 'production'
    return experiment_name, doc
```

And the experiment-dropdown POST handler writes to whichever applies:

```python
if request.user.is_authenticated:
    UserSettings.objects.update_or_create(
        user=request.user,
        defaults={'default_experiment': new_experiment},
    )
else:
    request.session['default_experiment'] = new_experiment
```

## Implementation impact

| File                                  | Change                                                        |
|---------------------------------------|---------------------------------------------------------------|
| `django/skolweb/settings.py`          | Add `SESSION_ENGINE = '...signed_cookies'`                    |
| `django/search/views.py`              | Extend `get_user_experiment` to read `request.session` for anon |
| Wherever the experiment-set POST lives| Branch on `is_authenticated` for the write path               |
| Same pattern, future preferences      | Add per-key fallback in a `get_user_preferences(request)` helper if the list grows beyond a couple of items |

Likely 30-50 lines total, no new dependencies, no migration.

## Trade-offs and alternatives considered

| Option                          | Why not |
|---------------------------------|---------|
| Raw `request.COOKIES` + manual signing | Reinvents what `signed_cookies` session backend gives free; more code, same outcome. |
| DB-backed sessions (default)    | Creates `django_session` rows for every anonymous visitor — defeats the "no per-visitor rows" goal. |
| Cache-backed sessions (Redis)   | Anonymous-prefs become tied to cluster health; lose them on cache eviction.  Overkill. |
| Anonymous-User row in `auth_user` | Either everyone shares one row (single shared state — wrong) or you create a row per visitor (back to DB bloat). |
| Browser localStorage + JS sync  | Requires JS on every page that reads prefs; can't be read server-side at template render time. |

## Future considerations

- **Cookie size.**  If preferences grow past several KB total, switch
  the backend to cached_db or DB-backed and accept the row cost — at
  that scale we have a "real" user-state problem anyway.
- **Per-key expiry.**  Whole session expires together; no individual
  preference TTL.  Fine for current scope; consider redis-backed
  preferences-as-KV if individual TTLs become a requirement.
- **Auth migration path.**  When an anonymous user logs in, we could
  optionally migrate their session-stored preferences into their new
  `UserSettings` row.  Out of scope for the initial implementation;
  trivial to add if it's wanted (one helper called from the post-login
  signal).
- **CSRF / cookie attributes.**  Default Django session cookies are
  HttpOnly + SameSite=Lax + Secure (when behind HTTPS).  These are
  the right defaults — don't change them.

## Scope: out for now

This work is queued behind the skol→tsqali Redis migration (Phase 4
prod cutover) and shouldn't block it.  The current logged-out
behavior is misleading but not a regression — it's been this way
since the experiment selector was added.  Pick this up when the
prod migration is settled.
