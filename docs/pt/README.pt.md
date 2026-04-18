<div align="center">

# RVC-WebUI-MacOS

**Um `.app` nativo do macOS do Retrieval-based Voice Conversion.**
Frontend SwiftUI + backend Python empacotado. Sem navegador, sem rede, sem pip install.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](../jp/README.ja.md) · [**中文简体**](../cn/README.cn.md) · [**한국어**](../kr/README.ko.md) · [**Français**](../fr/README.fr.md) · [**Português**](./README.pt.md) · [**Türkçe**](../tr/README.tr.md)

</div>

---

## O que é isto

O RVC-WebUI-MacOS reempacota o [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) como um **`.app` autônomo único** para Apple Silicon. Tudo — PyTorch, fairseq, todos os modelos pré-treinados (HuBERT, RMVPE, UVR5, pretrained_v2) — vai dentro do bundle. A primeira execução é um duplo clique; sem conda, sem pip, sem Homebrew, sem URL de localhost e sem necessidade de internet após o download.

O projeto original usa Gradio num navegador e FreeSimpleGUI para a janela de VC em tempo real. Este fork substitui ambos por um **frontend SwiftUI** que conversa com um **backend Python em subprocesso** via JSON-RPC no stdin/stdout.

## Funcionalidades

- **Totalmente offline** — todos os pesos de ML estão dentro do bundle. Sem etapa de download de recursos, sem busca no HuggingFace.
- **Apple Silicon em primeiro lugar** — backend PyTorch MPS pronto para uso. Volta corretamente à CPU quando o MPS não consegue processar uma operação.
- **Monitor de recursos sempre visível** — uso de CPU / memória unificada / MPS na barra de ferramentas, atualizado a cada segundo.
- **Barras de progresso honestas** — porcentagem por tarefa, rótulo de fase, ETA. Botões de cancelar aparecem apenas onde a operação é realmente interrompível.
- **Todos os recursos do RVC em um app**:
  - Inferência de arquivo único e em lote
  - Separação de vocais/instrumentos UVR5 com guia de escolha (qual HP/DeEcho/DeReverb escolher e por quê)
  - Cadeia opcional de auto-polimento (segunda passada de DeReverb após extração de vocais)
  - Pipeline completo de treinamento: pré-processamento → extração de F0 / características → treinamento → índice
  - Gerenciamento de modelos: comparar, fundir, extrair (slim), editar informações
  - Exportação para ONNX
  - Mudador de voz em tempo real com seletor de dispositivos + atualização a quente de parâmetros
- **Layout legível** — cada arquivo do usuário vive sob `~/Documents/RVC-WebUI/`, nada espalhado em pastas Application Support ocultas.
- **Padrões que não degradam o áudio** — saída em FLAC (sem perdas) por padrão; WAV / MP3 / M4A ainda disponíveis.

## Requisitos de sistema

| | Mínimo | Recomendado |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma ou superior |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro ou superior |
| RAM | 8 GB | 16 GB ou mais (treinamento consome muita memória) |
| Disco | 8 GB livres | 20 GB ou mais para treinar |

Macs Intel **não são suportados** — o PyTorch empacotado é exclusivo para ARM64.

## Instalação

### Para usuários finais

1. Baixe `RVC-WebUI.app.zip` do [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases) mais recente.
2. Descompacte e arraste `RVC-WebUI.app` para `/Applications`.
3. Dê duplo clique para iniciar. Na primeira execução, o Gatekeeper pode pedir confirmação — clique com o botão direito no app → **Abrir** → **Abrir** na caixa de diálogo.

Na primeira execução, o app cria `~/Documents/RVC-WebUI/` e subdiretórios para suas entradas, saídas, modelos e logs. É o único lugar onde ele escreve.

### Para desenvolvedores / compilando do código-fonte

```bash
# Pré-requisitos: Homebrew, Xcode CLT, Miniforge/conda
brew install xcodegen
conda install -n base -c conda-forge conda-pack

# 1. Clonar
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. Criar o ambiente conda (Python 3.10 + PyTorch MPS + fairseq etc.)
./setup_conda_env.sh
conda activate rvc

# 3. (Opcional) Teste rápido do backend Python isolado
python tools/test_rpc.py
# esperado: notificação "ready" → resposta de initialize → resource_stats a cada segundo

# 4. Baixar os assets de modelos do HuggingFace (hubert / rmvpe / pretrained_v2 / uvr5_weights, cerca de 2 GB)
./tools/download_assets.sh --all

# 5. Construir o bundle .app completo
./build_app.sh
# Produto: build/RVC-WebUI.app  (cerca de 4 GB incluindo PyTorch e todos os modelos)
```

Flags de build:

- `--skip-conda` — reutilizar o ambiente Python previamente empacotado (`build/python_env/`)
- `--skip-xcode` — reutilizar o binário Swift previamente compilado
- `--skip-sign` — pular assinatura de código (aceitável em dev local; não para distribuição)

Para builds assinados para distribuição:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## Arquitetura

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   barra de ferramentas: monitor CPU/MEM/MPS  │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 sobre stdio
                    │ (sem rede, sem sockets)
┌───────────────────▼──────────────────────────┐
│      Subprocesso Python (rpc_server.py)      │
│   VC · UVR5 · Treino · Tempo real · ONNX     │
│   psutil + torch.mps para amostrar recursos  │
└──────────────────────────────────────────────┘
```

- Frontend: `RVCApp/` — SwiftUI, gerado com `xcodegen` a partir de `project.yml`
- Ponte: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — inicia o subprocesso Python, despacha chamadas RPC, encaminha notificações de progresso / recursos para o estado `@Published`
- Backend: `rpc_server.py` + `rpc_training.py` — métodos JSON-RPC envolvem `infer/modules/vc`, `infer/modules/uvr5` e scripts de treinamento; stdout é bufferizado por linha para resposta rápida na inicialização
- Assets: `assets/hubert/`, `assets/rmvpe/`, `assets/pretrained_v2/`, `assets/uvr5_weights/` — todos copiados para `.app/Contents/Resources/rvc_backend/assets/` na build
- Runtime Python: `build/python_env/` via `conda-pack`, depois embutido em `.app/Contents/Resources/python/`

Veja [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md) para o pipeline completo de build e notas de arquitetura.

## Layout de arquivos

**Dentro do bundle** (`RVC-WebUI.app/Contents/Resources/`) — somente leitura:

```
rvc_backend/    # Código Python + assets, copiado do repositório
python/         # Runtime Python 3.10 empacotado com todas as dependências
```

**No seu diretório pessoal** (`~/Documents/RVC-WebUI/`) — todos os seus dados:

```
input/
  audio/          # Coloque arquivos aqui para inferência
  training/       # Conjuntos de dados de treinamento
output/
  inference/      # Resultados de conversão de arquivo único (FLAC por padrão)
  batch/          # Resultados de conversão em lote
  separation/     # vocals/ e accompaniment/ do UVR5
  onnx/           # Exportações ONNX
models/           # Seus modelos de voz .pth treinados
indices/          # Arquivos FAISS .index
logs/             # Checkpoints + logs de treinamento, um diretório por experimento
configs/inuse/    # Configuração de runtime
temp/             # Espaço temporário, limpo na inicialização
```

## Solução de problemas

**"RVC-WebUI.app está danificado e não pode ser aberto"** — Builds com assinatura ad-hoc são barrados pelo Gatekeeper em downloads novos. Correção:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — Esperado. O app roda em MPS; é uma linha de log de um caminho do código upstream, não um erro.

**O treinamento falha imediatamente na extração de características** — Corrigido neste fork. Se você estiver compilando a partir de um checkout muito antigo, garanta que `infer/lib/torch_compat.py` exista e seja importado antes de `fairseq` em `extract_feature_print.py`, `infer/modules/vc/utils.py` e `infer/lib/rtrvc.py`. Esse shim desliga o padrão `weights_only=True` do PyTorch 2.6+ no qual o carregador HuBERT do fairseq tropeça.

**Memória MPS esgotada durante o treinamento** — diminua `batch_size_per_gpu`, feche outros apps, ou defina `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (já definido na inicialização, mas vale conferir em `~/Documents/RVC-WebUI/logs/<exp>/train.log`).

**Primeira execução lenta** — o cold-import de fairseq + torch leva ~3 s no M1, ~2 s no M3. O splash mostra "aguardando backend" até o `alive` chegar; sem ação necessária.

## Desenvolvimento

O projeto SwiftUI é regenerado a cada build via xcodegen a partir de `RVCApp/project.yml`, então não edite `RVCApp.xcodeproj` à mão. Abra `RVCApp.xcodeproj` no Xcode e clique Run — em modo dev, o app inicia o `rpc_server.py` do repositório usando seu ambiente conda ativo (não o Python embutido), o que torna a iteração muito mais rápida.

Alterações no lado Python:
- O código-fonte vive na raiz do repositório (`rpc_server.py`, `rpc_training.py`, `infer/`, `rvc/`, `configs/`, `i18n/`, `tools/`)
- `./build_app.sh --skip-conda --skip-xcode` ressincroniza o backend Python em um `.app` existente sem recompilar o binário Swift nem reempacotar o Python
- Para iterações rápidas contra um `.app` já construído, basta `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/`

## Créditos

- Framework upstream de conversão de voz: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Blocos de construção: [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HIFIGAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (modelo pré-treinado por [yxlllc](https://github.com/yxlllc/RMVPE) e [RVC-Boss](https://github.com/RVC-Boss))
- Fork macOS inicial: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- Reestruturação nativa `.app`: este repositório

## Licença

MIT. Veja [LICENSE](../../LICENSE).
