# Plano e implementação — Stack inteligente e sessão temporal

Este documento consolida os cinco blocos entregues para o Ryzen 7 5800X. A
execução continua CPU-only, com `Stable` como perfil numérico e sem alteração
dos arquivos de origem.

## 1. Seleção explicável de frames

- `stacking_features.py` calcula percentis determinísticos para FWHM,
  roundness, estrelas, SNR, cobertura e RMS de alinhamento.
- Os perfis nativos são `Sharpness`, `Balanced` e `Signal`; `Custom` aceita
  JSON ou `fwhm=0.4,roundness=0.3,snr=0.3`.
- O relatório `<saída>/*_selection.json` registra score, componentes, métricas
  ausentes e motivo de cada decisão. Empates usam o caminho do frame.
- Se um Flow antigo não tiver métricas suficientes, a seleção cai explicitamente
  para `legacy_quality`; nenhum frame é perdido silenciosamente.

## 2. Trailing e qualidade temporal

- O Flow persiste `trail_coherence`, ângulo, excesso em pixels e coerência
  tangencial junto das métricas de forma existentes.
- O Stack classifica cada frame relativamente à própria sessão (`none`,
  `moderate`, `severe` ou `unreliable`) usando mediana/MAD robustos.
- `exclude_severe` exclui somente trailing grave com confiança alta;
  `weight_only` penaliza casos moderados; `report` apenas informa. Sugestões
  continuam revisáveis e não removem arquivos.
- `temporal_analysis.py` normaliza `DATE-OBS`, mantém timezone quando presente,
  ordena por instante, cria novo grupo somente quando o intervalo é maior que
  15 minutos e mantém frames sem horário no grupo `unknown`. O relatório
  temporal é atômico e marcado como `review_only`.

## 3. Empilhamento ponderado e determinístico

- `QualityWeightedMean` aplica o peso de qualidade depois de calcular a mesma
  fronteira de rejeição (`Stable`, sem `fastmath` ou reassociação das somas).
- Pesos são normalizados pela mediana e limitados a `[0.25, 2.0]`; pesos e
  coberturas por pixel acompanham cada folha/branch.
- A árvore binária restaura sempre a ordem lógica, independentemente da ordem
  de conclusão dos workers. Produtos `uint16`, máscaras e contagens permanecem
  compatíveis com o caminho legado.

## 4. Modelo cromático global RGB

- O modo `session-auto` faz uma amostragem limitada, uma imagem por vez, ajusta
  similaridade (com fallback de translação), agrega por mediana e valida MAD,
  confiança e deslocamento máximo.
- O modelo aceito é salvo atomicamente em `chromatic_session_model.json` e sua
  revisão entra no sidecar de cada Align. Modelo ausente, inválido ou instável
  volta automaticamente ao `hybrid` por frame.
- O ajuste nunca altera o modo `translation` legado nem inventa correção de
  rotação de campo; a previsão de rotação continua fora do escopo.

## 5. RAM-first, spill explícito e controles nativos

- O perfil `Intelligent` usa redução por bandas diretamente dos FITS e árvore
  binary-carry em RAM. O caminho `Stable` conserva as quatro folhas
  intercaladas do reducer legado para manter a ordem de rejeição bit a bit;
  `Fast` pode usar folhas contíguas limitadas. Nenhum substack FITS
  intermediário é criado e as folhas RAM são consumidas online, sem reter a
  sessão inteira.
- O orçamento efetivo respeita o limite configurado, reserva 20% da RAM física
  (mínimo de 4 GiB), usa no máximo 75% da memória segura e verifica folhas,
  merges e saída antes de alocar. Se não couber, retorna erro controlado.
- `ram_spill` é opt-in, exige pasta e limite explícitos, usa sessão temporária
  única, limite duro e limpeza em cancelamento/erro. A pasta de spill não pode
  ficar dentro da entrada.
- `stack_ram_report.json` registra orçamento, RSS, memória livre mínima, folhas,
  bytes de spill e `files_unchanged`. A UI expõe perfil, pesos, política de
  trailing, armazenamento e limite de spill sem editar JSON manualmente.

## Compatibilidade e validação

- Configuração nova/sem arquivo inicia `Intelligent` na UI; projetos com
  configurações salvas continuam `Legacy`. APIs diretas preservam os defaults
  históricos.
- O parser aceita JSONs de Flow antigos e só lê o cabeçalho FITS quando o
  timestamp normalizado não está presente.
- A suíte do checkout: **223 testes passados** e **13 subtestes passados**.
  Os testes dedicados cobrem pesos, trailing, timestamps, RAM/spill,
  igualdade bit a bit `Stable`, fallback AVX2 e modelo cromático.
- A pasta `B:\AstroImages\Notebook\Lagoon` é tratada como origem somente
  leitura; resultados e relatórios devem ser escolhidos fora dela.

## Medição operacional

O benchmark end-to-end versionado foi executado no checkout atual com sete
repetições quentes por variante, quatro workers, afinidade lógica `0–7`,
separação de frio/JIT e telemetria por estágio. O host reportou 8 núcleos
físicos, 16 lógicos, AVX2 e instruções YMM nos três probes Numba. Os digests,
máscaras, contagens e `uint16` permaneceram idênticos.

As medições históricas anteriores ao modo compacto chegaram a reduções acima
de 25% no caminho de arquivos individuais. A revalidação atual substitui essas
referências para o gate do checkout. Com `batch_compact` como padrão, sete
execuções quentes em quatro workers mediram `Stable` em **1,8136 s** contra
**2,4863 s** do baseline (**27,05%**, speedup **1,3709×**). O pico de RSS foi
aproximadamente **326,8 MiB**; digests, máscaras, contagens e produtos
`uint16` permaneceram idênticos e os probes AVX2/YMM continuaram positivos. O
gate obrigatório de 25% está **atingido** nesta variante compacta, sem relaxar
precisão nem atribuir artificialmente o ganho ao AVX2. O relatório da execução
foi gravado em `%TEMP%\\astrobatch_pipeline_current_quantized.json`.

Ensaios anteriores (1,7658 s / 28,98% e 2,1311 s / 14,28%) continuam
registrados no histórico para mostrar a dispersão entre variantes; não são
usados para substituir a medição atual do gate.
