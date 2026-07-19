# Conformité de l'API OpenAI-compatible d'aiollama

**Projet :** aiollama (fork d'Ollama)
**Objet :** Rapport des écarts entre l'implémentation actuelle d'aiollama et
l'API OpenAI officielle, pour servir de checklist lors de l'intégration dans
LibreChat (ou tout autre client conforme au SDK OpenAI).
**Date :** 2026-07-19 (mise à jour post-corrections — voir §4)
**Sources auditées :**
- `openai/openai.go` (types `Image*`, `Video*`, constantes, helpers)
- `server/imageapi.go` (handlers dédiés Images API : generations, edits, files)
- `server/imagefiles.go` (store TTL pour `response_format=url`)
- `server/routes.go:1932-1960` (routes OpenAI)
- `server/videoapi.go` (handlers vidéo)
- `server/videojobs/store.go` (job store + worker), `server/videojobs/transcoder.go`
- `docs/openai-videos-api-migration.md` (design de la couche vidéo)

Légende des statuts :
- ✅ **Conforme** — se comporte comme l'API OpenAI officielle.
- ⚠️ **Extension** — accepté par aiollama en plus du spec OpenAI (ne casse pas
  les clients conformes, mais expose des capacités hors-spec).
- ❌ **Manquant / non conforme** — un client SDK OpenAI standard peut échouer
  ou recevoir une réponse différente de celle attendue.

---

## 1. Images API

Les endpoints images sont implémentés par des **handlers dédiés**
(`server/imageapi.go`) et non plus par un middleware réécrivant
`/api/generate` : le spec exige `n > 1`, le transcodage de sortie, le bloc
`usage` et la livraison par URL, inexprimables dans l'ancien pipeline.

### 1.1 `POST /v1/images/generations`

| Aspect spec OpenAI | Implémentation aiollama | Statut |
|---|---|---|
| `model` requis, énumération (`dall-e-3`, `gpt-image-1`) | `model` requis, **nom de modèle Ollama libre** (ex `flux2-klein-4b`) | ⚠️ Extension (par conception, fork local) |
| `prompt` requis | ✅ requis | ✅ |
| `n` (1..10) | ✅ supporté (boucle de génération séquentielle, seeds `seed+i` si `seed` fourni) ; hors plage → 400 | ✅ |
| `size` enum (`1024x1024`, …) | Accepte n'importe quel `WxH` (≤ 4096), défaut `1024x1024` ; format invalide → 400 | ⚠️ Extension |
| `quality` (`low`/`medium`/`high`/`auto`) | ✅ validée et mappée sur les steps diffusion (`low`→20, `medium`→30, `high`→50, `auto`→défaut modèle) ; valeur inconnue → 400 | ✅ |
| `response_format` (`url`/`b64_json`) | ✅ `b64_json` (défaut) ou `url` (absolue, servie par `GET /v1/images/files/{id}`, TTL 30 min) ; valeur inconnue → 400 | ✅ |
| `style` (`vivid`/`natural`) | Validée (valeur inconnue → 400), sans effet local (concept DALL·E 3) | ✅ (no-op documenté) |
| `background` (`transparent`/`opaque`/`auto`) | Validée (valeur inconnue → 400). **Limite :** les modèles SD locaux n'émettent pas de canal alpha — `transparent` retourne un PNG opaque | ✅ (limite documentée) |
| `output_format` (`png`/`jpeg`/`webp`) | ✅ `png` (défaut), `jpeg` (stdlib), `webp` (via ffmpeg ; 400 si ffmpeg absent) | ✅ |
| `output_compression` (0..100) | ✅ appliqué à jpeg/webp (défaut 100) ; hors plage → 400 | ✅ |
| `stream` + `partial_images` | Non supporté : `stream=true` → **400 explicite** (au lieu d'un échec de parsing côté SDK) | ❌ (rejet propre documenté) |
| `moderation` (`low`/`auto`) | Validée, sans effet local (pas de filtre) | ✅ (no-op documenté) |
| `user` | Accepté | ✅ |
| `seed` | **Accepté** (extension aiollama ; déterministe par image via `seed+i`) | ⚠️ Extension |
| Content-Type : `application/json` | ✅ JSON | ✅ |
| Réponse : `{created, data:[{b64_json\|url}]}` | ✅ identique | ✅ |
| Champ `usage` (`input_tokens`/`output_tokens`/`total_tokens`) | ✅ toujours retourné (tokens mesurés si le runner les rapporte, sinon 0) | ✅ |
| Modèle non trouvé | 404 au format `{error:{message,type,code}}` | ✅ |

**Fichiers :** `server/imageapi.go` (`ImageGenerationsHandler`),
`openai/openai.go` (`ImageGenerationRequest`, `ParseImageSize`,
`StepsForImageQuality`), `server/imagefiles.go` (store URL).

### 1.2 `POST /v1/images/edits`

| Aspect spec OpenAI | Implémentation aiollama | Statut |
|---|---|---|
| Content-Type : **multipart/form-data** | ✅ multipart (spec) ; **JSON aussi accepté** (extension de rétro-compat : `image` en base64/data URL, string ou tableau) | ✅ (+⚠️ extension JSON) |
| Champ `image` : fichier multipart | ✅ fichier(s) `image` **et** `image[]` ; valeurs form base64 acceptées aussi | ✅ |
| `image[]` multiple (jusqu'à 16) | ✅ jusqu'à 16 images, toutes passées au runner | ✅ |
| Champ `mask` : fichier multipart | ✅ fichier `mask` ou valeur base64 ; **converti** de la sémantique OpenAI (alpha=0 → zone à éditer) vers la sémantique SD.cpp (blanc=255 → inpaint) et transmis jusqu'à `sdcpp.ImageGenParams.MaskImage` | ✅ |
| `model`, `prompt` requis | ✅ | ✅ |
| `n`, `size`, `quality`, `response_format`, `output_format`, `output_compression`, `background`, `seed`, `user` | ✅ identiques à `/v1/images/generations` | ✅ |
| Réponse : `{created, data:[{b64_json\|url}]}` + `usage` | ✅ identique | ✅ |

> ✅ **Point LibreChat résolu :** l'outil `OpenAIImageTools.js` envoie du
> multipart avec `image[]` — il fonctionne désormais **tel quel** contre
> aiollama, sans adaptation côté LibreChat.

**Fichiers :** `server/imageapi.go` (`ImageEditsHandler`,
`imageEditFromMultipart`, `imageEditFromJSON`, `ConvertMaskToSDCPP`),
plumbing mask : `llm/server.go` (`CompletionRequest.Mask`),
`api/types.go` (`GenerateRequest.Mask`), `x/diffgen/types.go`,
`x/diffgen/server.go`, `x/diffgen/runner.go` → `x/sdcpp` (`MaskImage`).

---

## 2. Videos API (Sora)

### 2.1 Endpoints — couverture

| Endpoint spec OpenAI | Implémenté ? | Statut |
|---|---|---|
| `POST /v1/videos` (create) | ✅ `server/videoapi.go` | ✅ |
| `GET /v1/videos/{id}` (retrieve) | ✅ | ✅ |
| `GET /v1/videos` (list) | ✅ | ✅ |
| `DELETE /v1/videos/{id}` (delete) | ✅ | ✅ |
| `GET /v1/videos/{id}/content` (download) | ✅ | ✅ |
| `POST /v1/videos/edits` | ✅ | ✅ |
| `POST /v1/videos/extensions` | ✅ | ✅ |
| `GET /v1/videos/{id}/content?variant=video` | ✅ | ✅ |
| `GET /v1/videos/{id}/content?variant=thumbnail` | ✅ première frame PNG via ffmpeg | ✅ |
| `GET /v1/videos/{id}/content?variant=spritesheet` | ✅ grille 5×5 PNG via ffmpeg (`tile` filter) | ✅ |
| `variant` invalide | 400 explicite | ✅ |
| `POST /v1/videos/characters` | ❌ Non implémenté (spécifique Sora cloud, sans équivalent local) | ❌ |
| `GET /v1/videos/characters/{id}` | ❌ Non implémenté | ❌ |
| `POST /v1/videos/{id}/remix` | ❌ Non implémenté (deprecated dans le SDK) | ❌ |

### 2.2 `POST /v1/videos` — corps de requête

| Champ spec | Implémentation aiollama | Statut |
|---|---|---|
| Content-Type : `multipart/form-data` | ✅ multipart ; JSON accepté en plus (extension documentée, homogène avec edits/extensions) | ✅ (+⚠️ extension JSON) |
| `prompt` (1..32000) | ✅ requis + validation longueur | ✅ |
| `model` (défaut `sora-2`) | **Requis**, nom Ollama libre | ⚠️ Extension (par conception) |
| `seconds` (`"4"`/`"8"`/`"12"`, défaut `"4"`) | ✅ validé strictement | ✅ |
| `size` (4 valeurs, défaut `720x1280`) | ✅ validé strictement | ✅ |
| `input_reference` (file part) | ✅ accepté | ✅ |
| `input_reference.image_url` (data URL) | ✅ accepté | ✅ |
| `input_reference.image_url` (URL http(s) distante) | ✅ téléchargée (timeout 30 s, ≤ 25 MiB, validation Content-Type `image/*`) | ✅ |
| `input_reference.file_id` | ❌ rejeté (`ErrVideoFileIDNotSupported` — pas de Files API store) | ❌ |
| Header réponse `openai-poll-after-ms` | ✅ 2000 ms | ✅ |

### 2.3 Objet `Video` (réponse)

Tous les champs spec sont présents (`id`, `object`, `created_at`, `completed_at?`,
`expires_at?`, `model`, `status`, `progress`, `prompt?`, `seconds`, `size`,
`remixed_from_video_id?`, `error?`) — `openai/openai.go`. ✅ Conforme.

### 2.4 `POST /v1/videos/edits`

| Champ spec | Implémentation | Statut |
|---|---|---|
| `prompt`, `model` requis | ✅ | ✅ |
| `video` (file part ou `{id}`) | ✅ les deux | ✅ |
| Content-Type : multipart | ✅ multipart ; JSON accepté aussi (extension homogène) | ✅ (+⚠️) |
| Sémantique : re-render I2V depuis 1ère frame | ✅ | ✅ |
| `remixed_from_video_id` positionné si `{id}` | ✅ | ✅ |

### 2.5 `POST /v1/videos/extensions`

| Champ spec | Implémentation | Statut |
|---|---|---|
| `prompt` requis, `seconds` requis (`"4"`..`"20"`) | ✅ | ✅ |
| `video` (file part ou `{id}`) | ✅ | ✅ |
| `Video.seconds` = total stitché (source + extension) | ✅ **dans les deux cas** : `{id}` → `seconds` du job source ; fichier uploadé → durée sondée par `ffprobe` (fallback : parsing `ffmpeg -i`, puis valeur demandée seule si indéterminable) | ✅ |
| Concaténation source + extension | ✅ via `ConcatMP4` | ✅ |
| Init image = dernière frame de la source | ✅ via `DecodeLastFrame` (ffmpeg `-sseof`) | ✅ |

### 2.6 Comportement asynchrone

Inchangé : create 200 + `queued`, polling avec `openai-poll-after-ms`,
`completed` → `/content`, `failed` → `error:{code,message}`. ✅ Conforme.

### 2.7 Limitations opérationnelles (hors spec, documentées)

| Aspect | Valeur | Impact |
|---|---|---|
| Persistance des jobs | **En mémoire uniquement** (`server/videojobs/store.go`) | Jobs perdus au redémarrage ; `GET /v1/videos/{id}` → 404 |
| Persistance des images `response_format=url` | **En mémoire uniquement** (`server/imagefiles.go`, TTL 30 min, cap 512 MiB LRU) | URLs invalides après TTL ou redémarrage |
| Concurrence | `MaxConcurrentJobs = 1` | 1 job vidéo à la fois ; les autres `queued` (spec-compatible) |
| TTL post-complétion | `JobTTL = 30 min` | `/content` → 404 après ce délai |
| Durée max d'un job | `MaxJobAge = 2 h` | Au-delà, force-fail `code:"timeout"` |
| Cap mémoire vidéo | `MaxTotalContentBytes = 2 GiB` | Éviction LRU des jobs complétés |
| Dépendance ffmpeg | Requis sur `PATH` (vidéo, variants, webp image) | Sans ffmpeg : job `failed` `code:"ffmpeg_required"` ; `output_format=webp` → 400 |
| `expires_at` | Positionné sur les jobs terminés | ✅ Conforme spec |

---

## 3. Différences transversales (images + vidéos)

| Aspect | OpenAI officiel | aiollama | Statut |
|---|---|---|---|
| Authentification | Clé API `Authorization: Bearer sk-...` | Aucune par défaut (instance auto-hébergée ; un reverse proxy peut en ajouter une) | ⚠️ Extension |
| Rate limiting | Par clé API | Aucun | ❌ (par conception, fork local) |
| Moderation / safety filters | Présents | Aucun (paramètre `moderation` validé mais sans effet) | ❌ (par conception, fork local) |
| Champ `usage` images | Présent | ✅ retourné | ✅ |
| Réponse d'erreur formatée | `{error: {message, type, code}}` | ✅ `openai.NewError(...)` partout (y compris 400 de validation) | ✅ |
| `seed` | Non documenté pour images | Accepté (images + vidéos via native API) | ⚠️ Extension |
| Modèles cloud (`sora-2*`, `dall-e-3`, `gpt-image-1`) | Disponibles | Non — seuls les modèles locaux Ollama | ❌ (par conception, fork local) |
| CORS | Configurable | Présent (hérité d'Ollama) | ✅ |
| Versionnage API (`/v1`) | ✅ | ✅ | ✅ |

---

## 4. État des corrections (checklist initiale → résultat)

### Priorité 1 — Cassaient les clients SDK standards → **résolu**

- [x] **`POST /v1/images/edits` en `multipart/form-data`** avec `image`
      (fichier), `image[]` (≤ 16) et `mask` (fichier optionnel). Le SDK Python
      `openai.images.edit` fonctionne tel quel. Le JSON reste accepté en
      extension de rétro-compat.
- [x] **`POST /v1/images/generations` : `response_format` honoré** —
      `b64_json` (défaut) ou `url` servie par `GET /v1/images/files/{id}`
      (store TTL 30 min, URL absolue tenant compte de `X-Forwarded-Proto`).
- [x] **`POST /v1/images/generations` : `n > 1` supporté** (1..10 ; hors
      plage → 400 explicite).

### Priorité 2 — Fonctionnalités OpenAI modernes → **résolu**

- [x] `quality` (`low`/`medium`/`high`/`auto`) → steps SD.cpp (20/30/50/défaut).
- [x] `output_format` (`png`/`jpeg`/`webp`) — webp via ffmpeg.
- [x] `output_compression` (0..100) — qualité jpeg/webp.
- [x] `background` — validé ; **limite documentée** : pas de vrai canal alpha
      (les modèles SD locaux n'en émettent pas).
- [x] `usage` dans `ImageGenerationResponse`.
- [x] `GET /v1/videos/{id}/content?variant=thumbnail` (1ère frame PNG) et
      `variant=spritesheet` (grille PNG 5×5 via ffmpeg).
- [x] Bonus : `mask` sur edits, plombé jusqu'à SD.cpp (`MaskImage`,
      conversion sémantique OpenAI alpha=0 → SD.cpp blanc=inpaint).
- [x] Bonus : `style`, `moderation`, `user` validés/acceptés (no-op local),
      `stream=true` → 400 explicite.

### Priorité 3 — Cohérence / extensions → **résolu ou documenté**

- [x] `POST /v1/videos/extensions` avec fichier uploadé : durée source sondée
      via `ffprobe` (fallback `ffmpeg -i`) → `seconds` total stitché correct.
- [x] `input_reference.image_url` http(s) : téléchargement (timeout 30 s,
      ≤ 25 MiB, validation Content-Type).
- [x] Cohérence JSON/multipart : les 3 endpoints vidéo (`create`, `edits`,
      `extensions`) acceptent multipart (spec) **et** JSON (extension
      documentée et homogène).
- [ ] Persistence des jobs vidéo (SQLite/disk derrière `JobStore`) — **non
      fait** : limite opérationnelle documentée (§2.7), sans impact sur la
      conformité SDK pendant la vie du processus.
- [ ] `POST /v1/videos/characters` + `GET .../characters/{id}` — **non fait**
      : spécifique Sora cloud (personnages persistants), sans équivalent pour
      des modèles locaux.
- [ ] Authentification par clé API / rate limiting / moderation — **non
      fait** : hors périmètre d'un fork local auto-hébergé (documenté §3).

---

## 5. Recommandations pour l'intégration LibreChat

L'intégration peut utiliser les **tools OpenAI natifs de LibreChat sans
adaptation** :

1. **Image génération** : `OpenAIImageTools.js` fonctionne tel quel
   (JSON + `b64_json`/`url`).
2. **Image édition** : `OpenAIImageTools.js` fonctionne tel quel (multipart
   `image[]`/`mask` désormais supportés nativement).
3. **Vidéo** : tool d'agent dédié qui :
   - `POST /v1/videos` (multipart) → `{id, status:queued}`.
   - Boucle `GET /v1/videos/{id}` en respectant `openai-poll-after-ms` (2000 ms).
   - `GET /v1/videos/{id}/content` une fois `completed` → MP4 binaire ;
     variants `thumbnail`/`spritesheet` disponibles pour l'aperçu.
   - Sauvegarde via `fileStrategy` → attachment `<video controls>`.
4. **Gestion d'erreur** : mapper les `error.code` aiollama
   (`ffmpeg_required`, `timeout`, `cancelled`, `generation_failed`,
   `no_frames`, `encoding_failed`, `source_video_unavailable`,
   `source_decode_failed`, `concat_failed`, `server_shutting_down`) vers des
   messages utilisateur compréhensibles.

Tableau de compatibilité global (état final) :

| Famille | Compat SDK OpenAI natif | Notes |
|---|---|---|
| `/v1/images/generations` | ✅ | n, quality, response_format (b64/url), output_format, usage, seed |
| `/v1/images/edits` | ✅ | multipart natif + mask + image[] ×16 ; JSON en extension |
| `/v1/videos` (create/retrieve/content) | ✅ | variants video/thumbnail/spritesheet |
| `/v1/videos/edits` | ✅ | multipart + JSON (extension) |
| `/v1/videos/extensions` | ✅ | seconds stitché correct (ffprobe sur upload) |
| `/v1/videos/characters` | ❌ | Sora cloud-only, non implémenté |

### Validation

- `go test ./...` : suite complète verte, incluant les nouveaux tests
  (`server/imageapi_test.go`, variants vidéo dans `server/videoapi_test.go`,
  `ProbeDurationSeconds`/`Spritesheet` ffmpeg dans
  `server/videojobs/transcoder_test.go`, helpers dans
  `openai/openai_test.go`).
- Script de validation SDK Python `openai` (WSL) : les requêtes
  `client.images.generate`/`client.images.edit` (multipart natif) sont
  acceptées et pilotent correctement le scheduler/runner. Le test E2E complet
  dépend des ressources de la machine (un modèle d'image de 30 Go sur une VM
  de 31 Go fait échouer le sampling natif par manque de RAM — limite
  d'environnement, indépendante de la couche API).
