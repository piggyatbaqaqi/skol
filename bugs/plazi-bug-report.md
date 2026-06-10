# Plazi `searchByDOI` returns unexpected responses for a batch of DOIs

Reporting 16 DOI lookup(s) against `/Treatments/searchByDOI` that did not return the expected result. Per the community guidance these are aggregated here with a CSV (`failure.csv`) rather than filed individually.

## Expected

HTTP 200 with a JSON array of at most 100 treatment objects (each with DocUuid / LnkDoi), or an empty array [] for a DOI that has no Plazi treatments.

## Observed (by failure mode)

### `runaway` — 16 case(s)

HTTP 200 with 698909 entries — far exceeds the 100-entry sanity cap; this looks like the full Plazi index rather than a match for this DOI.

- `10.3852/15-149` (doc `07a151f9-0581-5594-a9f8-0291bd7787da`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F15-149&format=json)
- `10.1016/s0181-1584(00)00105-6` (doc `087b1393-fe39-5801-9960-1bfe8fd50e4b`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.1016%2Fs0181-1584%2800%2900105-6&format=json)
- `10.3852/08-216` (doc `08c4320b-387a-5080-9164-3a5818eae70a`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F08-216&format=json)
- `10.3852/15-224` (doc `09233404-bd6f-5bd7-82c2-219b9cabe9fe`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F15-224&format=json)
- `10.3852/12-180` (doc `093e07e4-3b01-5310-a9d9-f1047b4de2b7`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F12-180&format=json)
- `10.1016/s1340-3540(14)00118-1` (doc `095fd6d3-e040-52a2-bfc6-e00cb210470f`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.1016%2Fs1340-3540%2814%2900118-1&format=json)
- `10.5252/cryptogamie-mycologie2019v40a1` (doc `09902565-4354-5fed-bea7-b90b4eb622e4`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.5252%2Fcryptogamie-mycologie2019v40a1&format=json)
- `10.3852/15-340` (doc `0a07a265-0ee6-50a6-8603-633f9fe19fbb`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F15-340&format=json)
- `10.23880/oajmms-16000169` (doc `0aac69bdcbbf5288952f87c1558d0615`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.23880%2Foajmms-16000169&format=json)
- `10.1016/s0181-1584(99)80006-2` (doc `0b209d28-e508-5a55-b019-54202a9c1a70`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.1016%2Fs0181-1584%2899%2980006-2&format=json)
- `10.1016/s0181-1584(00)00102-0` (doc `0ba34cbb-df41-51f5-bceb-4970ffd89851`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.1016%2Fs0181-1584%2800%2900102-0&format=json)
- `10.3852/12-145` (doc `0c009d36-35db-5793-9b9e-6685f2eaef57`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F12-145&format=json)
- `10.23880/oajmms-16000142` (doc `0c462f660df25a79b9ccc86c24e40770`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.23880%2Foajmms-16000142&format=json)
- `10.23880/oajmms-16000101` (doc `0d074ed021d25c74bc90b675c5d14f87`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.23880%2Foajmms-16000101&format=json)
- `10.1016/s0181-1584(01)01069-7` (doc `0d124922-98d3-54ce-99d3-ac49e16cba28`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.1016%2Fs0181-1584%2801%2901069-7&format=json)
- `10.3852/12-127` (doc `0de0715f-739a-52b1-b7e8-f32aac2f14ec`) — [exact query](https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F12-127&format=json)

## Reproduce all cases

`failure.csv` lists every case (columns: doc_id, doi, reason, detail, url, expected, observed). The script `reproduce.sh` replays each row:

```bash
./reproduce.sh failure.csv
```

A single case by hand:

```bash
curl -sS -A "skol-plazi-backfill/1.0 (https://synoptickeyof.life)" "https://api.plazi.org/GgSrvApi/v1/Treatments/searchByDOI?DOI=10.3852%2F15-149&format=json"
```

## Source documents

Retrieve the source document attachment for any case via the Django API (no credentials needed; it proxies CouchDB server-side). Append a name (e.g. `article.txt`) to fetch a specific attachment; the default is `article.pdf`:

```bash
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/07a151f9-0581-5594-a9f8-0291bd7787da/ -o 07a151f9-0581-5594-a9f8-0291bd7787da.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/087b1393-fe39-5801-9960-1bfe8fd50e4b/ -o 087b1393-fe39-5801-9960-1bfe8fd50e4b.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/08c4320b-387a-5080-9164-3a5818eae70a/ -o 08c4320b-387a-5080-9164-3a5818eae70a.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/09233404-bd6f-5bd7-82c2-219b9cabe9fe/ -o 09233404-bd6f-5bd7-82c2-219b9cabe9fe.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/093e07e4-3b01-5310-a9d9-f1047b4de2b7/ -o 093e07e4-3b01-5310-a9d9-f1047b4de2b7.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/095fd6d3-e040-52a2-bfc6-e00cb210470f/ -o 095fd6d3-e040-52a2-bfc6-e00cb210470f.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/09902565-4354-5fed-bea7-b90b4eb622e4/ -o 09902565-4354-5fed-bea7-b90b4eb622e4.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0a07a265-0ee6-50a6-8603-633f9fe19fbb/ -o 0a07a265-0ee6-50a6-8603-633f9fe19fbb.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0aac69bdcbbf5288952f87c1558d0615/ -o 0aac69bdcbbf5288952f87c1558d0615.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0b209d28-e508-5a55-b019-54202a9c1a70/ -o 0b209d28-e508-5a55-b019-54202a9c1a70.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0ba34cbb-df41-51f5-bceb-4970ffd89851/ -o 0ba34cbb-df41-51f5-bceb-4970ffd89851.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0c009d36-35db-5793-9b9e-6685f2eaef57/ -o 0c009d36-35db-5793-9b9e-6685f2eaef57.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0c462f660df25a79b9ccc86c24e40770/ -o 0c462f660df25a79b9ccc86c24e40770.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0d074ed021d25c74bc90b675c5d14f87/ -o 0d074ed021d25c74bc90b675c5d14f87.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0d124922-98d3-54ce-99d3-ac49e16cba28/ -o 0d124922-98d3-54ce-99d3-ac49e16cba28.pdf
curl -sS https://synoptickeyof.life/api/pdf/skol_dev/0de0715f-739a-52b1-b7e8-f32aac2f14ec/ -o 0de0715f-739a-52b1-b7e8-f32aac2f14ec.pdf
```
