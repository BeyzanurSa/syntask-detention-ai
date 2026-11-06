# detention-ai — Case 2: Detention Fee Önleme (Lokal AI)

Hat şirketi, sürücü ve depo mesajlarını okuyup her konteyner için detention riskini
hesaplayan, **tamamen lokal** çalışan bir pipeline. Harici API yok; model
Hugging Face'ten indirilip `llama-cpp-python` (Metal) ile bu makinede çalışıyor.

```
data/messages.txt
   │  messages.py   başlık + gövde ayrıştırma (deterministik)
   ▼
extract.py ── container regex ön filtresi ──► numara yok → "ignored" (M13, modele gitmez)
   │  Qwen3-4B + JSON-schema grammar → Pydantic → evidence + makullük kontrolleri
   ▼  (model SADECE çıkarım yapar: her alan için metinden alıntı + ham tarih ifadesi)
dates.py       "perşembe" → mesaj tarihine göre 08.10; "haftaya bir ara" → bilinmiyor
merge.py       konteyner başına alan alan birleştirme, en son mesaj kazanır
calculator.py  son serbest gün, detention günü, kademeli ücret, risk, kalan_gun
alerts.py      Türkçe uyarı; tüm sayılar koddan enjekte edilir
__main__.py    6 sütunlu tablo + açıklamalar + uyarılar + notlar / --json
```

Terminal çıktısı genişlikten bağımsız: ana tabloda sadece `container_no, last_free_day,
remaining_days, planned_return, estimated_fee, risk` var, hücreler kırılmaz/kısaltılmaz.
≥ 92 sütunda tek satır başlık, ≥ 65 sütunda başlıklar son `_`'den iki satıra bölünür,
daha darda her konteyner küçük bir kart olarak basılır. Gerekçeler tablo altında
konteyner başına tek madde; ISO 6346 uyarısı tek satırda toplanır. `--json` çıktısı case'in
alan adlarını (`konteyner_no, son_serbest_gun, kalan_gun, ...`) korur.

## Çalıştırma

```bash
python3.12 -m venv .venv
CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install llama-cpp-python
.venv/bin/pip install pydantic rich pytest huggingface_hub psutil
.venv/bin/hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models/Qwen3-4B-GGUF
```

```bash
.venv/bin/python -m detention_ai data/messages.txt                 # gerçek model
.venv/bin/python -m detention_ai data/messages.txt --fixture tests/fixtures/qwen3_4b_raw.json  # kayıtlı gerçek model çıktısı, modelsiz demo
.venv/bin/python scripts/record_fixture.py   # ham model çıktısını yeniden kaydet
.venv/bin/python -m detention_ai data/messages.txt --as-of 2026-10-08 --json
.venv/bin/python -m pytest            # hızlı testler (modelsiz)
.venv/bin/python -m pytest -m llm     # gerçek modelle uçtan uca
.venv/bin/python scripts/benchmark.py --label warm   # yükleme ve çıkarım süreleri ayrı, benchmark.json'a eklenir
```

## Sonuç (as_of = son mesaj tarihi, 05.10.2026)

| konteyner | son serbest gün | kalan_gun | iade | ücret | risk |
|---|---|---|---|---|---|
| MSCU4829153 | 03.10 | -2 | 08.10 (M9 "perşembe") | $750 | Yüksek |
| MAEU7731042 | 07.10 (4 iş günü) | 2 | 09.10 | $200 | Orta |
| HLCU5520187 | 07.10 | 2 | 12.10 (M7, M4'ü ezer) | $760 (3×120 + 2×200) | Yüksek |
| CMAU3301986 | 07.10 | 2 | bilinmiyor ("haftaya bir ara") | — | Belirsiz |
| TGHU9004512 | 13.10 | 8 | 08.10 | $0 | Düşük |
| OOLU6612408 | — | — | — | — | Kapsam dışı (demurrage) |

M13 (muhasebe) konteyner numarası içermediği için modele hiç gönderilmez ve yok sayılır.

## Model seçimi: Qwen3-4B, GGUF Q4_K_M

**Ölçüm (MacBook Air M2, 16 GB, Metal, `scripts/benchmark.py`, 06.11.2026):**

Üç ayrı süreç koşusu (`warm-1`, `warm-2`, `cold`). Ham değerler `benchmark.json` içinde, v2 prompt
ile (yaklaşık 1250 prompt token'ı, medyan 81 çıktı token'ı).

**Model yükleme** (`Llama(...)` çağrısı, çıkarımdan ayrı ölçüldü)

| durum | süre | not |
|---|---|---|
| sıcak (GGUF dosyası OS page cache'te) | 0.45 s / 0.46 s | iki koşu |
| soğuk (`sudo purge` sonrası) | **2.77 s** | tek koşu, 06.11.2026 05:02 |

Soğuk ölçüm komutu: `sudo purge && .venv/bin/python scripts/benchmark.py --label cold`.
İndirmeden hemen sonraki ilk yüklemede 14.3 s görmüştüm, ama o sırada cache durumu kontrol
edilmemişti. Kontrollü soğuk ölçüm bunu tekrarlamadı, bu yüzden 14.3 s raporlanmıyor.

Not: llama.cpp ağırlıkları mmap ile açıyor. Dosya sayfalarının bir kısmı yüklemede değil,
ilk kullanımda diskten okunuyor olabilir. Yani soğuk maliyetin bir kısmı ilk çağrıya kaymış
olabilir. Ama soğuk koşunun ilk çağrısı (13.0 s) sıcak koşularınkinden (15.7–15.9 s) yavaş
değil. SSD'den 2.3 GiB okumanın maliyeti, koşular arasındaki oynaklığın içinde kalıyor.

**Çıkarım (mesaj başına, model yüklendikten sonra)**

| ölçüm | warm-1 | warm-2 | cold |
|---|---|---|---|
| sürecin ilk çağrısı (M1) | 15.9 s | 15.7 s | 13.0 s |
| M1 tekrar, prefix yeniden kullanılarak | 9.9 s | 10.3 s | 8.0 s |
| M1 tekrar, `llm.reset()` sonrası (prefix yok) | 15.7 s | 17.2 s | 13.1 s |
| sadece prompt (`max_tokens=1`), tam prompt | 5.8 s | 7.4 s | 5.2 s |
| sadece prompt, system prompt prefix'i yeniden kullanılarak | 0.33 s | 0.43 s | 0.30 s |
| 12 mesaj, prefix yeniden kullanımı açık (normal akış) | medyan **5.4 s** (3.7–11.5) | medyan **5.6 s** (3.8–12.8) | medyan **5.1 s** (3.5–10.0) |
| 12 mesaj, her çağrıdan önce `reset()` | medyan 10.4 s (9.0–15.7) | medyan 13.4 s (11.2–20.7) | medyan 10.6 s (8.9–16.4) |
| bellek (psutil RSS, yükleme sonrası / bitiş) | 3.1 / 3.5 GB | 3.1 / 3.5 GB | 3.0 / 3.3 GB |

Bu ölçümlerden çıkanlar:
- İlk çağrının yavaşlığı "ısınma" değil, tam prompt değerlendirmesi. İlk çağrı, `reset()`
  sonrası çağrıyla aynı sürüyor (15.9 s ve 15.7 s).
- llama-cpp-python ardışık çağrılarda ortak prompt başlangıcını yeniden kullanıyor. Bu ölçüldü:
  prompt süresi 5.8 s'den 0.33 s'ye iniyor, mesaj başına medyan yaklaşık 2 kat düşüyor.
- Geri kalan süre çıktı üretimi. M1 en uzun çıktı (139 token) olduğu için en yavaş mesaj. Bunun
  nedeni evidence alanları. v1 şemasında medyan 4.6 s idi. Doğruluk için yaklaşık 1 s/mesaj ödendi.
- Koşular arasında belirgin oynaklık var (`reset()` modunda 10.4 / 13.4 / 10.6 s). Soğuk koşu,
  sıcak koşulardan yaklaşık 75 dk sonra yapıldı ve en hızlısı oldu. Fansız Air'de termal
  kısma olası bir neden, ama bunu ölçmedim.
- 13 mesajlık uçtan uca CLI koşusu (ölçülen): 12 LLM çağrısı, medyan 5.6 s, toplam çıkarım 95 s.
  İlk çağrının yaklaşık 16 s'si dahil.

Bellek: psutil RSS, mmap edilmiş ağırlıkları + KV cache + Metal ayırmalarını kapsıyor.
`time -l`'in "peak memory footprint" değeri dosyaya bağlı mmap sayfalarını saymadığı için
daha düşük görünüyor (v1 ölçümünde 0.5 GB), bu yüzden RSS'i raporluyorum. Her ölçüm tek
makine, üç koşu: yaklaşık değerler.

- **Boyut:** 4B parametre, görev kısa mesajlardan alan çıkarımı. 7-8B modeller
  16 GB'lık bir makinede de çalışır ama token başına süre kabaca parametre sayısıyla arttığından latency'nin yaklaşık 2 katına çıkmasını beklerim (ölçmedim) ve RAM payı daralır;
  1-2B modeller Türkçe gün adı, "aynı gün iade" gibi ifadelerde daha kırılgan. 4B, Türkçe + İngilizce karışık
  metinde yeterli kalite / hız dengesi.
- **Quantization:** Q4_K_M, 4-bit K-quant'ların en yaygın "kalite/boyut" noktası
  (~2.5 GB). Q8_0 yaklaşık 2 kat büyük, bu iş için ölçülebilir bir kazanç beklemiyorum. Görev serbest
  metin üretimi değil, şemalı çıkarım; grammar ve doğrulama katmanı kalite kaybını tolere ediyor.
- **Donanım:** MacBook Air M2, 16 GB unified memory, fan yok. `n_gpu_layers=-1` ile
  tüm katmanlar Metal GPU'da. Model + 4K context ile RSS yaklaşık 3.5 GB, tarayıcı ve IDE
  açıkken de rahat sığıyor.
- **Runtime:** `llama-cpp-python`. Python içinden doğrudan çağrılıyor, ayrı servis yok ve
  JSON-schema → GBNF grammar desteği yerleşik. Ollama yedek seçenek, ikisi birden kurulmadı.
- **Lisans:** Qwen3 Apache-2.0.
- **Qwen3 "thinking":** kapalı (`/no_think`). Ayrıca grammar ilk token'dan itibaren JSON
  zorladığı için `<think>` bloğu üretilemiyor.

## Neden aritmetik kodda, modelde değil?

- Tarih ve ücret hesabı kural tabanlıdır: kurallar case'te birebir yazılı. Kod, **aynı girdi ve config
  ile tekrarlanabilir** sonuç üretir, birim testleriyle sınır durumları
  (iade = son serbest gün, $500 eşiği, hafta sonu) doğrulanabilir.
- Küçük bir LLM'in "28 Eyl + 5 iş günü" ya da kademeli ücret toplaması yapması
  hem hata yapmaya açık hem denetlenemez. temperature=0 sadece rastgeleliği
  **azaltır**, hesabın doğru olduğunu garanti etmez.
- Bu yüzden model tarihleri **ham ifade** olarak döner (`"perşembe"`, `"haftaya bir ara"`).
  Çözümleme `dates.py`'da mesaj tarihine göre yapılır. Modele mesaj tarihi bilerek verilmez.
- Uyarı metni de şablon: sayılar hesaplayıcıdan gelir, model metin yazmaz. Böylece
  uyarıdaki $ değeri ile tablodaki $ değeri asla ayrışmaz.

## Şemaya uygunluk nasıl garanti ediliyor?

1. **Grammar (sözdizimi + tip):** Pydantic şeması `$ref`'ler açılarak JSON Schema'ya,
   llama.cpp tarafından GBNF grammar'a çevrilir. Sampling her adımda sadece şemaya
   uyan token'lara izin verir. Çıktı her zaman parse edilebilir, alanlar ve enum'lar doğru.
2. **Pydantic doğrulama (anlam):** `extra="forbid"` (model `last_free_day` gibi alan
   uyduramaz), `free_days` 0–60, `usd_per_day > 0`, enum'lar. Hata olursa 1 kez tekrar
   denenir, yine olmazsa mesaj **needs_review** olur; tahmin yapılmaz.
3. **Ön filtre:** konteyner numarası olmayan mesaj modele hiç gitmez.
4. **Konteyner grounding:** modelin verdiği numara, mesaj metnindeki regex eşleşmesiyle
   karşılaştırılır. Uydurma veya hatalı yazılmış numara metindekiyle değiştirilir ya da atılır.
5. **Evidence kontrolleri (deterministik):** şemada her kritik alan bir `evidence` alıntısı taşır,
   tarihler ayrıca `raw` ifade olarak gelir (`{"evidence": "perşembe iade ediyoruz", "raw": "perşembe"}`).
   Grammar alan sırasını sabitlediği için model önce alıntıyı, sonra değeri yazar. Kod şunları kontrol eder:
   - alıntı mesaj metninde birebir geçiyor mu,
   - `raw` tarih ya da sayı kendi alıntısının içinde mi,
   - alıntı doğru türde bir kelime içeriyor mu: `gate_out` için gate-out/released/picked up,
     `planned_return` için iade/return, terminal durumu için terminal/liman/gümrük,
     serbest süre için free/serbest, ücret için USD/detention.

   Kontrolü geçemeyen alan atılır, yani "bahsedilmedi" sayılır. Böylece merge'de önceki
   doğru bilgiyi ezemez. Terminal durumu bool değil, null olabilen bir alıntı: modelin
   "tahmin ederek" `true` yazabileceği bir alan yok.
6. **Makullük:** mesaj tarihinden sonraki bir gate-out, ya da aynı mesajda gate-out ile aynı
   veya ondan önceki bir iade atılır (ikinci savunma hattı).

Grammar şekli garanti eder, değerlerin doğruluğunu garanti etmez. Değerler için 2–6 ve
`pytest -m llm` (gerçek modelle ham çıktı ve case sonuçları) var.

### Gerçek modelin yaptığı hatalar ve nasıl düzeldi

| sürüm | gözlenen hata | çözüm |
|---|---|---|
| v0 | Şemanın ilk alanı `category` idi. Model içeriği "okumadan" `"unrelated"` yazdı, 12 mesajın hepsini ilgisiz saydı. | Alan kaldırıldı, ilgililik kararını regex veriyor. |
| v0 | "USD 150/day" → `days: 1` | Alan `first_n_days` oldu, `usd_per_day` öne alındı, few-shot eklendi. |
| v1 | M9/M10: iade tarihini gate-out'a da kopyaladı. M7/M11: kanıt yokken "terminalde" dedi. M3: gate-out'u iade olarak yazdı. | Bunları o sürümde kod sonradan düzeltiyordu. v2'de şema yeniden tasarlandı: evidence + raw, terminal durumu için bool yerine alıntı, kurallar prompt'ta açıkça yazıldı. |
| v2 ilk deneme | M10: `{"evidence": "iade", "raw": "Cuma"}`. Evidence kontrolü reddetti, MAEU Belirsiz'e düşüyordu. | "aynı gün iade" kalıbı için farklı değerlerle bir few-shot örneği eklendi (`PQRU…`, "14 Mart Salı"). |
| v2 | 12 mesajın 12'sinde ham çıktı doğru, doğrulama katmanı hiçbir alanı düzeltmedi. | `tests/fixtures/qwen3_4b_raw.json` + `tests/test_llm.py` |

v1'in hata kalıpları `tests/test_extract.py` içinde birim testi olarak duruyor (M3, M7, M9, M10,
M11). Model aynı hatayı yeniden yaparsa evidence kontrolleri yakalıyor.

Dürüst not: few-shot örnekleri ve kelime listeleri bu 13 mesajı görerek yazıldı. Örnekler
case mesajlarını birebir içermiyor, ama kalıp olarak yakınlar (örneğin "aynı gün iade").
Yeni mesaj tipleri yeni örnek veya kelime gerektirebilir. Bilinmeyen bir kalıp gelirse
sistem tahmin etmek yerine alanı atar ve sonuç "Belirsiz"/manuel inceleme olur. Prompt,
model sürümü veya quantization değişince ham çıktı da değişebilir, bu yüzden
`pytest -m llm` tekrar koşulmalı.

## Kurallar ve varsayımlar

- Serbest süre gate-out'un **ertesi günü** başlar. İş günü ise Cmt/Paz atlanır (resmî tatiller
  modellenmedi). Tür belirtilmemişse ("7 days free") takvim günü kabul edilir (case kuralı).
- Detention günü = iade − son serbest gün (iade günü dahil, takvim günü).
- Kademeli ücret sırayla uygulanır. `first_n_days=null` olan kademe kalan tüm günleri kapsar.
- Risk: ücret > 0 ve (as_of > son serbest gün veya ücret ≥ $500) → Yüksek. Ücret > 0 → Orta.
  Ücret 0 → Düşük. İade bilinmiyor → Belirsiz.
- `kalan_gun` = son serbest gün − as_of (takvim günü, negatif olabilir). `as_of` varsayılanı
  en son mesajın tarihi (05.10.2026), `--as-of` ile değiştirilebilir.
- "En son mesaj" = mesaj tarihi, aynı gün ise akıştaki sıra. **Alan alan** birleştirme:
  M7 sadece iade tarihini ezer, M3'teki gate-out ve ücret korunur.
- Sonradan gelen belirsiz ifade, önceki kesin tarihi ezer: en güncel bilgi "plan belirsiz".
- Tek başına gün adı, mesaj tarihinden itibaren (o gün dahil) ilk denk gelen gündür.
  "haftaya perşembe" = bir sonraki takvim haftasının perşembesi. Yıl yoksa mesaj yılı
  kullanılır, 180 günden eski düşerse sonraki yıl.
- Konteyner no: boşluk/tire silinir, `^[A-Z]{4}\d{7}$` zorunlu. ISO 6346 kontrol hanesi
  **sadece uyarı**: case'teki 6 numaranın hiçbiri tutmuyor, reddetmek tüm veriyi düşürürdü.
- Terminalde bekleyen konteyner (M12) detention kapsamı dışında, demurrage olarak işaretlenir.

## Bilinçli olarak yapılmayanlar

Resmî tatil takvimi, hat bazlı free-time sözleşme tablosu, DB/API/UI, çoklu konteynerli
mesajların ayrıntılı testi, demurrage hesabı. Hepsi mevcut modüllere eklenebilir; 45
dakikalık canlı görevde kapsam dışı bırakıldı.

## 45 dakikalık prova planı

| dk | adım | ne gösterilir / söylenir |
|---|---|---|
| 0–3 | Problemi özetle | "Model sadece okur, hesap kodda." Mimari şemayı çiz (README üstü). |
| 3–8 | Model seçimi | Qwen3-4B Q4_K_M, M2 16 GB, ölçülen load/latency/RAM (yukarıdaki tablo). Neden 7B değil, neden Q4. |
| 8–13 | Konteyner regex + ISO 6346 | `container.py`, `pytest tests/test_container.py`. "MSCU 482915-3" → MSCU4829153. Kontrol hanesi neden uyarı. |
| 13–20 | LLM çıkarımı | `schema.py` + `extract.py`: grammar, Pydantic, evidence + raw, kontroller. Canlı: M9'un ham JSON'unu göster (`evidence: "perşembe iade ediyoruz"`). Sonra `pytest tests/test_extract.py -k m9` ile v1 hatasının (`gate_out: "perşembe"`) nasıl yakalandığını göster. "category alanı" dersini anlat. |
| 20–26 | Tarih çözümleme | `dates.py`: "perşembe" → 08.10, "haftaya bir ara" → bilinmiyor. Testler. |
| 26–31 | Merge | `merge.py`: M7 vs M4, alan alan birleştirme, aynı gün sıra. |
| 31–37 | Hesaplayıcı + risk | `calculator.py`: MAEU iş günü, HLCU kademeli, $500 eşiği, as_of. |
| 37–41 | Uçtan uca | `python -m detention_ai data/messages.txt`: tablo + uyarılar. `--as-of 2026-10-08` ile MAEU'nun Yüksek'e çıktığını göster. |
| 41–45 | Soru-cevap | Aşağıdaki olası sorular. |

**Canlıda gelebilecek değişiklikler ve nereye dokunulur:**
- Yeni mesaj eklenir → `data/messages.txt`. Kod değişmez.
- $500 eşiği değişir → `calculator.HIGH_FEE_THRESHOLD`.
- Resmî tatil → `last_free_day`'e `holidays: set[date]` parametresi.
- Yeni tarih ifadesi ("ay sonu", "yarından sonra") → `dates.py` + test. Bilinmiyorsa `unknown` kalır.
- Demurrage da hesaplansın → `still_at_terminal` + yeni kural modülü.
- Bir mesajda birden fazla konteyner → şema zaten `containers: list`, grounding çoklu eşleşmeyi destekler.

**Yedek plan:** model yüklenmezse veya makine yavaşlarsa `--fixture tests/fixtures/qwen3_4b_raw.json`
ile aynı akış saniyeler içinde koşar. Görüşmeden önce modeli bir kez çalıştırıp page cache'i ısıtın.
Yükleme soğukta 2.8 s, sıcakta 0.5 s. Canlı akış yaklaşık 95 s sürer, bu sürede
`extract.py`'yi anlatmak iyi bir kullanım.

**Olası sorular:**
- *Model yanlış tarih ifadesi kopyalarsa?* Evidence kontrolü (alıntı metinde mi, tarih alıntının içinde mi, alıntıda iade/gate-out kelimesi var mı) alanı atar. Notlarda ve gerekçede kaynak mesaj id'si ile ham ifade görünür.
- *Neden fine-tune yok?* 13 mesaj, kurallar net: prompt + grammar yeterli, fine-tune veri ve zaman ister.
- *temperature=0 ne sağlıyor?* Rastgeleliği azaltır. Doğruluğu doğrulama katmanları sağlar.
- *Güven skoru?* Bu runtime per-field güven vermiyor. Onun yerine kural tabanlı kontroller
  (grounding, Pydantic) ve bilinmeyeni "Belirsiz" olarak işaretleme.
