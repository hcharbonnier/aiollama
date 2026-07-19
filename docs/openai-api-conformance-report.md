# Conformité de l'API OpenAI-compatible d'aiollama

**Projet :** aiollama (fork d'Ollama)
**Objet :** Rapport des écarts entre l'implémentation actuelle d'aiollama et
l'API OpenAI officielle, pour servir de checklist lors de l'intégration dans
LibreChat (ou tout autre client conforme au SDK OpenAI).
**Date :** 2026-07-19 (mise à jour post-corrections — voir §4 ; revue
openai-python **2.46.0** — voir §6 ; plan de correctifs §7 **appliqué** :
O1, O2, F1, F3, F4, F5 corrigés)
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
| `size` enum (`1024x1024`, …) | Accepte n'importe quel `WxH` (≤ 4096), défaut `1024x1024` ; format invalide → 400 ; **`"auto"` accepté** (valeur envoyée par les GPT image models récents) → taille par défaut | ⚠️ Extension |
| `quality` (`low`/`medium`/`high`/`auto`) | ✅ validée et mappée sur les steps diffusion (`low`→20, `medium`→30, `high`→50, `auto`→défaut modèle) ; **alias legacy acceptés** (`standard`→auto, `hd`→high) ; valeur inconnue → 400 | ✅ |
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
| Champ `usage` (`input_tokens`/`output_tokens`/`total_tokens`) | ✅ toujours retourné (tokens mesurés si le runner les rapporte, sinon 0) ; **sous-objets `input_tokens_details` / `output_tokens_details` peuplés** (`{image_tokens, text_tokens}`, conformes SDK 2.46.0) | ✅ |
| Modèle non trouvé | 404 au format `{error:{message,type,code}}` | ✅ |

**Fichiers :** `server/imageapi.go` (`ImageGenerationsHandler`),
`openai/openai.go` (`ImageGenerationRequest`, `ParseImageSize`,
`StepsForImageQuality`), `server/imagefiles.go` (store URL).

### 1.2 `POST /v1/images/edits`

| Aspect spec OpenAI | Implémentation aiollama | Statut |
|---|---|---|
| Content-Type : **multipart/form-data** | ✅ multipart (spec) ; **JSON aussi accepté** (extension de rétro-compat : `image` en base64/data URL, string ou tableau) | ✅ (+⚠️ extension JSON) |
| Champ `image` : fichier multipart | ✅ fichier(s) `image` **et** `image[]` ; valeurs form base64 acceptées aussi | ✅ |
| `image[]` multiple (jusqu'à 16) | ✅ jusqu'à 16 images : la première en `InitImage`, les suivantes en `RefImages` SD.cpp | ✅ |
| Champ `mask` : fichier multipart | ✅ fichier `mask` ou valeur base64 ; **converti** de la sémantique OpenAI (alpha=0 → zone à éditer) vers la sémantique SD.cpp (blanc=255 → inpaint) et transmis jusqu'à `sdcpp.ImageGenParams.MaskImage` | ✅ |
| `input_fidelity` (`high`/`low`, gpt-image-1+) | Ignoré silencieusement (champ non parsé) | ⚠️ (no-op) |
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
| `GET /v1/videos/{id}/content?variant=thumbnail` | ✅ première frame PNG via ffmpeg (calculée une fois, **mise en cache sur le job**) | ✅ |
| `GET /v1/videos/{id}/content?variant=spritesheet` | ✅ grille 5×5 PNG via ffmpeg (`tile` filter, **mise en cache sur le job**) | ✅ |
| `variant` invalide | 400 explicite | ✅ |
| `POST /v1/videos/characters` | ✅ **501 explicite** au format d'erreur OpenAI (spécifique Sora cloud, sans équivalent local) | ✅ (501 documenté) |
| `GET /v1/videos/characters/{id}` | ✅ **501 explicite** | ✅ (501 documenté) |
| `POST /v1/videos/{id}/remix` | ✅ JSON `{prompt}` ; source `:video_id` (404 si inconnu, 409 si non complété) ; `model`/`size`/`seconds` hérités du job source ; `remixed_from_video_id` positionné | ✅ |

### 2.2 `POST /v1/videos` — corps de requête

| Champ spec | Implémentation aiollama | Statut |
|---|---|---|
| Content-Type : `multipart/form-data` | ✅ multipart ; JSON accepté en plus (extension documentée, homogène avec edits/extensions) | ✅ (+⚠️ extension JSON) |
| `prompt` (1..32000) | ✅ requis + validation longueur | ✅ |
| `model` (défaut `sora-2`) | **Requis**, nom Ollama libre | ⚠️ Extension (par conception) |
| `seconds` (`"4"`/`"8"`/`"12"`, défaut `"4"`) | ✅ validé strictement | ✅ |
| `size` (4 valeurs, défaut `720x1280`) | ✅ validé strictement | ✅ |
| `input_reference` (file part) | ✅ accepté | ✅ |
| `input_reference.image_url` (data URL) | ✅ accepté : JSON string dans le form-field `input_reference` **et** notation à crochets du SDK (`input_reference[image_url]=...`) | ✅ |
| `input_reference.image_url` (URL http(s) distante) | ✅ idem (filtrage anti-SSRF conservé) | ✅ |
| `input_reference.file_id` | ✅ rejeté proprement (`ErrVideoFileIDNotSupported`), **y compris via la notation à crochets du SDK** (`input_reference[file_id]`) | ✅ (rejet documenté) |
| Header réponse `openai-poll-after-ms` | ✅ 2000 ms | ✅ |

### 2.3 Objet `Video` (réponse)

Tous les champs spec sont présents (`id`, `object`, `created_at`, `completed_at?`,
`expires_at?`, `model`, `status`, `progress`, `prompt?`, `seconds`, `size`,
`remixed_from_video_id?`, `error?`) — `openai/openai.go`. ✅ Conforme.

### 2.4 `POST /v1/videos/edits`

| Champ spec | Implémentation | Statut |
|---|---|---|
| `prompt` requis ; `model` hérité de la source | ✅ `model` (non envoyé par le SDK) **hérité du job source** quand `video={id}` ; 400 conservé si fichier uploadé sans `model` | ✅ |
| `video` (file part ou `{id}`) | ✅ file part **et** référence `{id}` via le champ bracket `video[id]=...` du SDK | ✅ |
| Content-Type : multipart | ✅ multipart ; JSON accepté aussi (extension homogène) | ✅ (+⚠️) |
| Sémantique : re-render I2V depuis 1ère frame | ✅ | ✅ |
| `remixed_from_video_id` positionné si `{id}` | ✅ | ✅ |

### 2.5 `POST /v1/videos/extensions`

| Champ spec | Implémentation | Statut |
|---|---|---|
| `prompt` requis, `seconds` requis (`"4"`..`"20"`) | ✅ ; `model` (non envoyé par le SDK) **hérité du job source** quand `video={id}` | ✅ |
| `video` (file part ou `{id}`) | ✅ file part **et** référence `{id}` via le champ bracket `video[id]=...` du SDK | ✅ |
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
      ≤ 25 MiB, validation Content-Type, **filtrage anti-SSRF** :
      destinations privées/loopback/link-local rejetées, IP publique épinglée
      au connect, erreurs génériques anti-oracle).
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

### Plan §7 (revue SDK 2.46.0) → **appliqué (2026-07-19)**

- [x] **O1** — champs multipart en notation bracket du SDK parsés
      (`video[id]`, `input_reference[image_url]`,
      `input_reference[file_id]`) sur create/edits/extensions ;
      `input_reference[file_id]` → 400 explicite au lieu d'un drop
      silencieux.
- [x] **O2** — `model`/`size` hérités du job source sur edits/extensions
      quand `video={id}` (le SDK ne les envoie pas, comme l'API cloud) ;
      400 conservé pour un fichier uploadé sans `model`.
- [x] **F1** — `POST /v1/videos/{id}/remix` implémenté : JSON `{prompt}`,
      source 404/409 selon son état, héritage `model`/`size`/`seconds`,
      `remixed_from_video_id` positionné.
- [x] **F3** — `size: "auto"` accepté (generations + edits) → taille par
      défaut.
- [x] **F4** — `usage.input_tokens_details` / `output_tokens_details`
      peuplés (`{image_tokens, text_tokens}`).
- [x] **F5** — `POST /v1/videos/characters` et
      `GET /v1/videos/characters/{id}` → 501 explicite au format OpenAI.
- Non retenus (§7.3, inchangés) : personnages persistants (capacité),
      `images/variations`, streaming images, `input_fidelity`.

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
| `/v1/images/generations` | ✅ | n, quality (+alias legacy), size auto, response_format (b64/url), output_format, usage (+details), seed |
| `/v1/images/edits` | ✅ | multipart natif + mask + image[] ×16 ; JSON en extension |
| `/v1/videos` (create/retrieve/content) | ✅ | variants video/thumbnail/spritesheet ; `input_reference` dict du SDK parsé (bracket) |
| `/v1/videos/edits` | ✅ | référence `{id}` via `video[id]` ; `model`/`size` hérités de la source |
| `/v1/videos/extensions` | ✅ | seconds stitché correct (ffprobe sur upload) ; héritage idem edits |
| `/v1/videos/{id}/remix` | ✅ | JSON `{prompt}` ; `model`/`size`/`seconds` hérités de la source |
| `/v1/videos/characters` | ✅ (501) | Sora cloud-only → 501 explicite |

---

## 6. Revue de conformité avec `openai-python 2.46.0`

Audit réalisé en lisant la lib installée dans WSL
(`/home/hugues/.local/lib/python3.12/site-packages/openai/`) — version
2.46.0 = dernière version publiée (et version utilisée côté client).
L'objectif : valider qu'un script `import openai; client = OpenAI(...)` ciblant
aiollama produit un comportement strictement identique à celui de l'API cloud
OpenAI.

### 6.1 Images — écarts tolérables (2) → **tous corrigés**

- **`quality: "standard" | "hd"`** : la valeur `quality` du SDK 2.46.0 accepte
  ces deux littéraux (legacy `dall-e-2/3`) en plus de
  `low|medium|high|auto`. ~~`ValidImageQuality` ne les connaît pas → 400~~
  **CORRIGÉ (2026-07-19) :** les alias sont acceptés et mappés
  (`standard`→`auto`, `hd`→`high`) dans `validateImageRequest`
  (`server/imageapi.go`).
- **`size: "auto"`** : le SDK autorise cette valeur (les GPT image models
  récents la supportent nativement). ~~`ParseImageSize` (Sscanf `%dx%d`)
  échoue sur `"auto"` → 400~~ **CORRIGÉ (2026-07-19) :** `"auto"` est
  court-circuité dans `validateImageRequest` et mappé sur la taille par
  défaut (1024×1024), sur generations et edits.

### 6.2 Vidéos — `client.videos.create/edit/extend` : 3 cassures via le SDK → **toutes corrigées (2026-07-19)**

Trois écarts de wire format entre la sérialisation multipart du SDK et le
parsing serveur. Vérifié empiriquement en exécutant
`client._serialize_multipartform({"video": {"id": "x"}, ...})` :

```
input_reference[image_url]   => data:image/png;base64,...
input_reference[file_id]    => file_abc
video[id]                   => vid_abc
```

(notation `qs.stringify_items(array_format="brackets")`, présente depuis
au moins la 2.40.0 — ce n'est pas un changement récent, mais le code serveur
ne l'a jamais couverte.)

**a. `video={"id": "..."}` sur `/v1/videos/edits` et `/v1/videos/extensions`**

~~Le serveur lit `r.FormValue("video")` (chaîne JSON) ou un *file part*
nommé `video` ; `video[id]` n'est reconnu par aucun des deux → 400
"video is required (a file part or a {id} object)".~~ **CORRIGÉ
(2026-07-19) :** `videoSourceInput.fromForm` lit aussi
`r.FormValue("video[id]")` ; test multipart au format exact du SDK
(`mw.WriteField("video[id]", ...)`) dans `server/videoapi_test.go`.

**b. `input_reference={...}` sur `/v1/videos` (create)**

~~Le serveur lit `r.FormValue("input_reference")` (JSON string) ou un
*file part* ; `input_reference[image_url]` n'est pas reconnu →
`hasInputReference()` renvoie `false` et la génération part **sans**
image d'init (échec silencieux). Un `input_reference={"file_id": "..."}`
est aussi silencieusement ignoré au lieu de renvoyer
`ErrVideoFileIDNotSupported`.~~ **CORRIGÉ (2026-07-19) :**
`videoCreateInput.fromForm` lit aussi `input_reference[image_url]` et
`input_reference[file_id]` ; ce dernier alimente `FileID` et déclenche
donc `ErrVideoFileIDNotSupported` (400 explicite) au lieu d'un drop
silencieux.

**c. `model` absent du wire sur `edit` et `extend`**

~~`VideoEditParams`/`VideoExtendParams` du SDK 2.46.0 ne portent **que**
`prompt`, `video` (+ `seconds` pour extend) — pas de `model` ni `size`.
Le serveur exige `model` → 400 "model is required" à chaque appel.~~
**CORRIGÉ (2026-07-19) :** `parseEditExtendRequest` résout le job source
via `s.videoJobs.Get` **avant** la validation `model`/`size` quand la
vidéo est une référence `{id}` et hérite `model`/`size` du job source
(une seule résolution, fusionnée avec la lookup `sourceSeconds`) ; le
400 est conservé pour un fichier uploadé sans `model` (déviation
documentée : pas de modèle à hériter).

### 6.3 Endpoints vidéo absents → **traités (2026-07-19)**

- `POST /v1/videos/{video_id}/remix` (corps JSON `{prompt}`) — méthode
  `client.videos.remix()` du SDK. ~~404~~ **CORRIGÉ (2026-07-19) :**
  implémenté (`VideoRemixHandler`) — source en path (404 si inconnue,
  409 si non complétée), `model`/`size`/`seconds` hérités du job source,
  `remixed_from_video_id` positionné, re-render I2V depuis la 1ère frame
  (même machinerie que `VideoEditHandler`).
- `POST /v1/videos/characters` et `GET /v1/videos/characters/{id}` —
  API Sora cloud (personnages persistants), sans équivalent local
  naturel. ~~404~~ **CORRIGÉ (2026-07-19) :** 501 explicite au format
  d'erreur OpenAI (`VideoCharactersHandler`) — la capacité reste absente
  mais le statut est désormais non ambigu pour les clients.

### 6.4 Images — bloc `usage` incomplet → **corrigé (2026-07-19)**

Le SDK 2.46.0 attend dans `ImagesResponse.usage` les champs
`input_tokens_details: {image_tokens, text_tokens}` (requis) et
`output_tokens_details: {image_tokens, text_tokens}` (optionnel).
~~Le serveur n'émet que `input_tokens`/`output_tokens`/`total_tokens`~~
**CORRIGÉ (2026-07-19) :** `openai.ImageUsage` porte désormais
`InputTokensDetails`/`OutputTokensDetails` (`{image_tokens, text_tokens}`),
peuplés dans `runImageGeneration` — input : `{image_tokens: nombre
d'images d'entrée, text_tokens: tokens du runner}` ; output :
`{image_tokens: images générées, text_tokens: 0}`. Forme JSON verrouillée
par un test (`TestImageUsageWireShape`).

### 6.5 Streaming, input_fidelity, variations (mineur)

- `stream=true` / `partial_images` sur images → 400 explicite
  ("streaming is not supported by this server"), comportement documenté
  et acceptable.
- `input_fidelity` (nouveau, `gpt-image-1+`) sur `images.edit` →
  ignoré silencieusement, pas d'erreur.
- `POST /v1/images/variations` (legacy `dall-e-2`) → 404, jamais
  implémenté (hors scope).

### 6.6 Tableau récapitulatif des écarts SDK 2.46.0

| # | Écart | Gravité | Casse le client ? | Correctif proposé |
|---|---|---|---|---|
| 6.1a | ~~`quality: standard/hd` rejetés~~ **CORRIGÉ** (alias acceptés : `standard`→auto, `hd`→high) | — | non | fait |
| 6.1b | ~~`size: "auto"` rejeté~~ **CORRIGÉ** (accepté → taille par défaut, fix F3) | — | non | fait |
| 6.2a | ~~`video[id]` non parsé~~ **CORRIGÉ** (champ bracket parsé, fix O1) | — | non | fait |
| 6.2b | ~~`input_reference[*]` non parsé~~ **CORRIGÉ** (bracket parsé ; `file_id` → `ErrVideoFileIDNotSupported`, fix O1) | — | non | fait |
| 6.2c | ~~`model` manquant sur `edit`/`extend`~~ **CORRIGÉ** (hérité du job source quand `{id}`, fix O2) | — | non | fait |
| 6.3  | ~~`remix`/`characters` absents~~ **CORRIGÉ** (remix implémenté, fix F1 ; 501 pour characters, fix F5) | — | non | fait |
| 6.4  | ~~`usage.input_tokens_details` manquant~~ **CORRIGÉ** (détails peuplés, fix F4) | — | non | fait |
| 6.5  | streaming, `input_fidelity`, variations | basse | partiel | **non retenus** (§7.3) : 400 explicite conservé pour streaming, `input_fidelity` ignoré, variations en 404 |

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
- **Revue `openai-python 2.46.0`** (juillet 2026, dernière version
  publiée) : les trois cassures bloquantes via le SDK (`videos.create`
  avec `input_reference` dict ignoré silencieusement,
  `videos.edit`/`extend` avec référence `{id}` et `model` manquant → 400)
  ainsi que `remix` (404) sont **corrigées** — voir §6 et §7. Les
  nouveaux tests reproduisent le **wire format réel du SDK 2.46.0**
  (champs multipart bracket `video[id]`, `input_reference[image_url]`,
  `input_reference[file_id]`) et non seulement les chemins JSON/curl.

---

## 7. Plan de correctifs

Critère de tri : *le backend local peut-il honorer la sémantique ?* ×
*un client réel l'appelle-t-il ?* × *le coût est-il proportionné ?*.
Résultat : 3 fixes **obligatoires** (§7.1), 5 fixes **facultatifs**
(§7.2), 3 écarts **non retenus** (§7.3, décisions documentées).
**Statut : plan appliqué le 2026-07-19 — tous les fixes retenus sont
implémentés et testés.**

Definition of done commune : `go test ./...` vert dans WSL, et les
nouveaux tests reproduisent le **wire format réel du SDK 2.46.0**
(champs bracket `video[id]`, `input_reference[image_url]`) — pas
seulement les chemins JSON/curl déjà couverts. ✅ Atteinte.

### 7.1 Fixes obligatoires (cassent les chemins principaux du SDK)

- [x] **O1 — Parser les champs multipart en notation bracket du SDK**
      (corrige §6.2a + §6.2b) — **FAIT (2026-07-19)**. Le SDK openai-python
      sérialise les dicts imbriqués via `qs.stringify_items(array_format="brackets")` :
      `video[id]=...`, `input_reference[image_url]=...`,
      `input_reference[file_id]=...`.
  - `server/videoapi.go` — `videoCreateInput.fromForm` : après le file
    part et la valeur JSON-string, lire aussi
    `r.FormValue("input_reference[image_url]")` →
    `v.InputReference = &openai.ImageInputReferenceParam{ImageURL: val}`
    et `r.FormValue("input_reference[file_id]")` → `FileID` (afin que
    `resolveInputReference` renvoie `ErrVideoFileIDNotSupported` au lieu
    d'ignorer silencieusement).
  - `server/videoapi.go` — `videoSourceInput.fromForm` : lire aussi
    `r.FormValue("video[id]")` → `v.videoID`.
  - Tests (`server/videoapi_test.go`) : requêtes multipart construites
    avec `mw.WriteField("video[id]", "vid_abc")` et
    `mw.WriteField("input_reference[image_url]", "data:image/png;base64,...")`
    (format exact produit par `client.videos.edit/create` du SDK) ;
    `input_reference[file_id]` → 400 avec le message
    `ErrVideoFileIDNotSupported` (et non un drop silencieux).

- [x] **O2 — Hériter `model`/`size` du job source sur edits/extensions**
      (corrige §6.2c) — **FAIT (2026-07-19)**. Le SDK n'envoie ni `model`
      ni `size` sur `videos.edit`/`videos.extend` ; le cloud les hérite de
      la vidéo source.
  - `server/videoapi.go` — `parseEditExtendRequest` : quand
    `in.src.videoID != ""`, résoudre le job source via
    `s.videoJobs.Get(in.src.videoID)` **avant** la validation
    `model`/`size` ; si résolu : `model` vide → modèle du job source,
    `size` vide → taille du job source (au lieu du défaut `720x1280`).
    Fusionné avec la lookup `sourceSeconds` existante (une seule
    résolution).
  - Si la source est un fichier uploadé et `model` vide → 400 conservé
    (déviation documentée : pas de modèle à hériter).
  - Tests : edit/extend multipart avec `video[id]` d'un job complété et
    **sans** `model`/`size` → job créé avec modèle et taille hérités ;
    fichier uploadé sans `model` → 400 inchangé.

### 7.2 Fixes facultatifs (robustesse / surface additionnelle justifiée)

- [x] **F1 — Implémenter `POST /v1/videos/{video_id}/remix`** (§6.3) —
      **FAIT (2026-07-19)**.
  - `server/routes.go` :
    `r.POST("/v1/videos/:video_id/remix", cloudPassthroughMiddleware(...), s.VideoRemixHandler)`.
  - `server/videoapi.go` — `VideoRemixHandler` : corps JSON `{prompt}`
    (le SDK n'envoie pas de multipart ici — aucun fichier dans
    `VideoRemixParams`) ; `prompt` requis (≤ 32000) ; source =
    `:video_id` en path ; job source doit exister (sinon 404) et être
    `completed` (sinon 409) ; `model`/`size`/`seconds` hérités du job
    source (même logique que O2) ; crée le job avec
    `RemixedFromID = video_id`, `Extend = false`.
  - Tests : remix d'un job complété → 200 + `queued` +
    `remixed_from_video_id` ; prompt manquant → 400 ; id inconnu → 404 ;
    source non complétée → 409.

- [x] **F2 — Accepter `quality: "standard"` et `"hd"`** (§6.1a) — **FAIT
      (2026-07-19)** : alias mappés dans `validateImageRequest`
      (`standard`→auto, `hd`→high) + tests (`server/imageapi_test.go`).

- [x] **F3 — Accepter `size: "auto"`** (§6.1b) — **FAIT (2026-07-19)**.
  - `server/imageapi.go` — `validateImageRequest` :
    `if size == "" || size == "auto" { size = openai.ImageDefaultSize }`.
  - Tests : `size=auto` accepté sur generations et edits.

- [x] **F4 — Peupler `usage.input_tokens_details` /
      `output_tokens_details`** (§6.4) — **FAIT (2026-07-19)**.
  - `openai/openai.go` : `ImageUsage` gagne
    `InputTokensDetails *ImageUsageTokensDetails` et
    `OutputTokensDetails *ImageUsageTokensDetails`
    (`{image_tokens, text_tokens}`, `omitempty`).
  - `server/imageapi.go` — `runImageGeneration` : input
    `{image_tokens: len(p.images), text_tokens: inputTokens}`, output
    `{image_tokens: n, text_tokens: 0}`.
  - Tests : assertion sur la forme JSON du bloc `usage`
    (`server/imageapi_test.go`, `TestImageUsageWireShape`).

- [x] **F5 — Renvoyer 501 explicite sur `characters`** (§6.3) — **FAIT
      (2026-07-19)**.
  - `server/routes.go` : `POST /v1/videos/characters` et
    `GET /v1/videos/characters/:character_id` → handler renvoyant
    `openai.NewError(http.StatusNotImplemented, "characters are a Sora cloud feature; not supported by aiollama")`.
  - Test : 501 + format d'erreur OpenAI.

### 7.3 Non retenus (décisions documentées — aucun code)

| Écart | Décision | Justification |
|---|---|---|
| `videos.create_character` / `get_character` (implémentation) | **Non retenu** (501 seulement, fix F5) | Concept Sora cloud (personnages persistants) sans équivalent SD.cpp/WAN/LTX ; exigerait un store d'identités pour zéro capacité backend. Le besoin local (cohérence de personnage) se couvre via LoRA / `input_reference`. |
| `images.create_variation` (`POST /v1/images/variations`) | **Non retenu** (404 conservé) | Endpoint legacy dall-e-2, abandonné par OpenAI pour les modèles récents ; cas d'usage couvert par `images.edit` ; aucun client moderne (LibreChat inclus) ne l'appelle. |
| Streaming images (`stream=true`, `partial_images`) | **Non retenu** (400 explicite conservé) | Gros chantier (flux SSE d'événements `partial_image`) pour une valeur niche ; à reconsidérer uniquement si la preview progressive devient un besoin UX réel. |
| `input_fidelity` sur `images.edit` | **Non retenu** (ignoré silencieusement, documenté §1.2) | Concept gpt-image-1 (préservation des traits) ; l'équivalent local (denoising strength) n'a pas de mapping fidèle. |

### 7.4 Ordre d'exécution (suivi le 2026-07-19)

1. **O1 + O2** (ensemble, même fichier `server/videoapi.go`) — débloque
   `videos.create` (input_reference dict), `videos.edit`, `videos.extend`
   via le SDK.
2. **F1** (remix) — dépend de la logique d'héritage introduite par O2.
3. **F2 + F3 + F4** (images, one-liners) — même passage sur
   `openai/openai.go` + `server/imageapi.go`.
4. **F5** (501 characters) — trivial, à glisser dans n'importe quel
   passage.
