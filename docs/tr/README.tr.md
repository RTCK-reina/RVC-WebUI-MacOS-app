<div align="center">

# RVC-WebUI-MacOS

**Retrieval-based Voice Conversion'ın yerel macOS `.app`'i.**
SwiftUI ön yüzü + gömülü Python arka ucu. Tarayıcı yok, ağ yok, pip install yok.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](../jp/README.ja.md) · [**中文简体**](../cn/README.cn.md) · [**한국어**](../kr/README.ko.md) · [**Français**](../fr/README.fr.md) · [**Português**](../pt/README.pt.md) · [**Türkçe**](./README.tr.md)

</div>

---

## Bu nedir

RVC-WebUI-MacOS, [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) projesini Apple Silicon için **tek bağımsız `.app`** olarak yeniden paketler. Her şey — PyTorch, fairseq, tüm önceden eğitilmiş modeller (HuBERT, RMVPE, UVR5, pretrained_v2) — paketin içinde gelir. İlk açılış bir çift tıklamadır; conda yok, pip yok, Homebrew yok, localhost URL'si yok, indirdikten sonra internet gerekmez.

Orijinal proje tarayıcıda Gradio'yu, gerçek zamanlı VC penceresinde FreeSimpleGUI'yi kullanır. Bu çatal her ikisini de bir **SwiftUI ön yüzü** ile değiştirir ve alt süreç olarak çalışan **Python arka ucu** ile stdin/stdout üzerinde JSON-RPC ile konuşur.

## Özellikler

- **Tamamen çevrimdışı** — tüm ML ağırlıkları paketin içindedir. Varlık indirme adımı yok, HuggingFace'den çekme yok.
- **Apple Silicon öncelikli** — PyTorch MPS arka ucu doğrudan kullanılabilir. MPS'in işleyemediği bir op olduğunda düzgün şekilde CPU'ya geri düşer.
- **Her zaman açık kaynak monitörü** — araç çubuğunda CPU / birleşik bellek / MPS kullanımı, her saniye yenilenir.
- **Dürüst ilerleme çubukları** — görev başına yüzde, faz etiketi, ETA. İptal düğmeleri yalnızca işlemin gerçekten kesilebilir olduğu yerlerde görünür.
- **Tüm RVC özellikleri tek uygulamada**:
  - Tek dosya ve toplu çıkarım
  - Model seçim rehberi ile UVR5 vokal/enstrüman ayrıştırma (hangi HP/DeEcho/DeReverb'ı ne zaman seçeceğiniz)
  - Vokal çıkarımı sonrası isteğe bağlı otomatik ince ayar zinciri (ikinci geçiş DeReverb)
  - Tam eğitim hattı: ön işleme → F0 / özellik çıkarma → eğitim → indeks
  - Model yönetimi: karşılaştırma, birleştirme, çıkarma (yalın), bilgi düzenleme
  - ONNX dışa aktarma
  - Aygıt seçici + sıcak parametre güncelleme ile gerçek zamanlı ses değiştirici
- **Okunaklı dosya düzeni** — her kullanıcı dosyası `~/Documents/RVC-WebUI/` altında yaşar, gizli Application Support klasörlerine dağılmaz.
- **Sesi bozmayan varsayılanlar** — çıktı varsayılan olarak FLAC (kayıpsız); WAV / MP3 / M4A hâlâ kullanılabilir.

## Sistem gereksinimleri

| | Minimum | Önerilen |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma veya üstü |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro veya daha iyisi |
| RAM | 8 GB | 16 GB+ (eğitim bellek yiyicidir) |
| Disk | 8 GB boş | Eğitim için 20 GB+ |

Intel Mac'ler **desteklenmez** — paketlenmiş PyTorch yalnızca ARM64'tür.

## Kurulum

### Son kullanıcılar için

1. En son [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases)'ten `RVC-WebUI.app.zip`'i indirin.
2. Çıkarın, `RVC-WebUI.app`'i `/Applications`'a sürükleyin.
3. Başlatmak için çift tıklayın. İlk çalıştırmada Gatekeeper onay isteyebilir — uygulamaya sağ tıklayın → **Aç** → iletişim kutusunda **Aç**.

İlk başlatmada uygulama, girişleriniz, çıktılarınız, modelleriniz ve günlükleriniz için `~/Documents/RVC-WebUI/` ve alt dizinleri oluşturur. Yazdığı tek yer burasıdır.

### Geliştiriciler için / kaynaktan derleme

```bash
# Ön koşullar: Homebrew, Xcode CLT, Miniforge/conda
brew install xcodegen
conda install -n base -c conda-forge conda-pack

# 1. Klonla
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. conda ortamını oluştur (Python 3.10 + PyTorch MPS + fairseq vb.)
./setup_conda_env.sh
conda activate rvc

# 3. (İsteğe bağlı) Python arka ucunu tek başına test et
python tools/test_rpc.py
# beklenen: "ready" bildirimi → initialize yanıtı → her saniye resource_stats

# 4. HuggingFace üzerinden model varlıklarını indir (hubert / rmvpe / pretrained_v2 / uvr5_weights, yaklaşık 2 GB)
./tools/download_assets.sh --all

# 5. Tam .app paketini derle
./build_app.sh
# Üretilen: build/RVC-WebUI.app  (PyTorch ve tüm modeller dahil yaklaşık 4 GB)
```

Derleme bayrakları:

- `--skip-conda` — daha önce paketlenmiş Python ortamını yeniden kullan (`build/python_env/`)
- `--skip-xcode` — daha önce derlenmiş Swift ikili dosyasını yeniden kullan
- `--skip-sign` — kod imzalamayı atla (yerel geliştirme için uygun; dağıtım için değil)

Dağıtım için imzalı derlemeler:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## Mimari

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   araç çubuğu: CPU / MEM / MPS monitörü      │
└───────────────────┬──────────────────────────┘
                    │ stdio üzerinden JSON-RPC 2.0
                    │ (ağ yok, soket yok)
┌───────────────────▼──────────────────────────┐
│        Python alt süreci (rpc_server.py)     │
│   VC · UVR5 · Eğitim · Gerçek zaman · ONNX   │
│   psutil + torch.mps ile kaynak örnekleme    │
└──────────────────────────────────────────────┘
```

- Ön yüz: `RVCApp/` — SwiftUI, `project.yml`'den `xcodegen` ile üretilir
- Köprü: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — Python alt sürecini başlatır, RPC çağrılarını dağıtır, ilerleme/kaynak bildirimlerini `@Published` duruma yönlendirir
- Arka uç: `rpc_server.py` + `rpc_training.py` — JSON-RPC yöntemleri `infer/modules/vc`, `infer/modules/uvr5` ve eğitim betiklerini sarmalar; erken ilk yanıt için stdout satır tamponlanmıştır
- Varlıklar: `assets/hubert/`, `assets/rmvpe/`, `assets/pretrained_v2/`, `assets/uvr5_weights/` — derleme sırasında `.app/Contents/Resources/rvc_backend/assets/`'a kopyalanır
- Python çalışma zamanı: `conda-pack` ile `build/python_env/`, ardından `.app/Contents/Resources/python/`'a gömülür

Tam derleme hattı ve mimari notları için [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md)'a bakın.

## Dosya düzeni

**Paket içinde** (`RVC-WebUI.app/Contents/Resources/`) — salt okunur:

```
rvc_backend/    # Depodan kopyalanan Python kodu + varlıklar
python/         # Tüm bağımlılıklarla gömülü Python 3.10 çalışma zamanı
```

**Ev dizininizde** (`~/Documents/RVC-WebUI/`) — tüm verileriniz:

```
input/
  audio/          # Çıkarım için buraya dosya bırakın
  training/       # Eğitim veri kümeleri
output/
  inference/      # Tek dosya dönüştürme sonuçları (varsayılan FLAC)
  batch/          # Toplu dönüştürme sonuçları
  separation/     # UVR5'in vocals/ ve accompaniment/ klasörleri
  onnx/           # ONNX dışa aktarımları
models/           # Eğitilmiş .pth ses modelleriniz
indices/          # FAISS .index dosyaları
logs/             # Eğitim kontrol noktaları + günlükleri, deney başına bir dizin
configs/inuse/    # Çalışma zamanı yapılandırması
temp/             # Geçici alan, başlangıçta temizlenir
```

## Sorun giderme

**"RVC-WebUI.app hasarlı ve açılamıyor"** — Ad-hoc imzalı derlemeler yeni indirilmelerde Gatekeeper'a takılır. Çözüm:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — Beklenen. Uygulama MPS üzerinde çalışır; bu, üst kaynak kod yolundan gelen bir günlük satırıdır, hata değildir.

**Eğitim özellik çıkarımında hemen başarısız oluyor** — Bu çatalda düzeltildi. Çok eski bir kontrol noktasından derliyorsanız, `infer/lib/torch_compat.py`'nin var olduğundan ve `extract_feature_print.py`, `infer/modules/vc/utils.py`, `infer/lib/rtrvc.py`'de `fairseq`'ten önce içe aktarıldığından emin olun. Bu shim, fairseq'in HuBERT yükleyicisinin takıldığı PyTorch 2.6+'nın `weights_only=True` varsayılanını devre dışı bırakır.

**Eğitim sırasında MPS belleği yetersiz** — `batch_size_per_gpu`'yu düşürün, diğer uygulamaları kapatın veya `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` ayarlayın (başlangıçta zaten ayarlanmış, ancak `~/Documents/RVC-WebUI/logs/<exp>/train.log`'da doğrulamaya değer).

**İlk çalıştırma yavaş** — fairseq + torch soğuk içe aktarma M1'de ~3 s, M3'te ~2 s sürer. Splash, `alive` gelene kadar "arka uç bekleniyor" gösterir; işlem gerekmez.

## Geliştirme

SwiftUI projesi her derlemede xcodegen tarafından `RVCApp/project.yml`'den yeniden üretilir, bu yüzden `RVCApp.xcodeproj`'u elle düzenlemeyin. Xcode'da `RVCApp.xcodeproj`'u açın ve Run'a basın — geliştirme modunda uygulama, gömülü Python yerine etkin conda ortamınızla deponun `rpc_server.py`'sini başlatır, bu da çok daha hızlı iterasyon sağlar.

Python tarafı değişiklikleri:
- Kaynak, depo kökünde yaşar (`rpc_server.py`, `rpc_training.py`, `infer/`, `rvc/`, `configs/`, `i18n/`, `tools/`)
- `./build_app.sh --skip-conda --skip-xcode`, Swift ikilisini yeniden derlemeden veya Python'u yeniden paketlemeden mevcut bir `.app`'e Python arka ucunu yeniden senkronize eder
- Halihazırda derlenmiş bir `.app`'e karşı geçici iterasyon için `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/` yeterlidir

## Teşekkürler

- Üst kaynak ses dönüştürme çerçevesi: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Yapı taşları: [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HIFIGAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (önceden eğitilmiş model [yxlllc](https://github.com/yxlllc/RMVPE) ve [RVC-Boss](https://github.com/RVC-Boss) tarafından)
- İlk macOS çatalı: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- Yerel `.app` yeniden tasarım: bu depo

## Lisans

MIT. [LICENSE](../../LICENSE)'a bakın.
