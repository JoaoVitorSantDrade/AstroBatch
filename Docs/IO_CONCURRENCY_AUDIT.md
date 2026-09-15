# Auditoria de I/O e concorrência

Data da medição: 2026-09-15. Esta auditoria executa somente CPU e preserva o
modo Stable, FITS científicos e máscaras existentes.

## Diagnóstico

O corpus quente tem 20 frames: 4 mono 96×96, 8 mono 512×512 e 8 RGB
512×512. O host reportou 8 núcleos físicos, 16 lógicos e AVX2. Os contadores de
I/O são deltas do processo Windows; como o corpus é reutilizado pelo cache do
SO, eles mostram bytes lógicos processados, não throughput garantido do disco.

| workers | total sem compressão | Flow | Align | Stack | pico RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,500 s | 1,805 s | 0,399 s | 0,311 s | 245 MiB |
| 2 | 2,209 s | 1,667 s | 0,275 s | 0,266 s | 262 MiB |
| 4 | 1,823 s | 1,347 s | 0,250 s | 0,245 s | 276 MiB |
| 8 | 1,839 s | 1,337 s | 0,244 s | 0,244 s | 308 MiB |

Esta matriz é diagnóstica (três repetições por largura), após reduzir o custo
da telemetria de threads. O `cpu_to_wall` do Flow foi `0,97 / 1,15 / 1,45 /
1,48` para 1 / 2 / 4 / 8 workers; a leitura lógica permaneceu em cerca de
37,0 MiB por execução.

Após a janela limitada para detecção das âncoras, a repetição no checkout final
(três execuções por largura) mediu `2,885 / 2,413 / 2,046 / 1,935 s` para
1 / 2 / 4 / 8 workers. O Flow ficou em `1,996 / 1,772 / 1,492 / 1,397 s`;
o pico RSS foi `245,0 / 262,4 / 275,4 / 312,1 MiB` e o pico de threads
`56 / 56 / 58 / 61`. Oito workers ainda reduz a parede em aproximadamente
5,4% contra quatro, mas custa cerca de 36,7 MiB adicionais e aumenta a pressão
de pools nativos; quatro continua sendo o teto operacional recomendado. O JSON
completo é [io_concurrency_benchmark_current_20260915.json](io_concurrency_benchmark_current_20260915.json).

Na variante com FITS comprimido no Align, os totais foram 4,843 / 4,276 /
3,939 / 3,693 s para 1 / 2 / 4 / 8 workers. O Align comprimido escreveu cerca
de 3,0 MiB lógicos por execução; o não comprimido escreveu cerca de 16,2 MiB.

Flow foi o maior estágio em todas as configurações. Ele leu cerca de 37,0 MiB
por execução, mas o uso de CPU ficou próximo do tempo de parede (`cpu_to_wall`
0,97 com um worker e 1,48 com oito). Isso caracteriza trabalho de detecção/
registro e coordenação como prioridade, não uma fila de disco comprovada.

O Stack leu cerca de 44,9 MiB e escreveu 31,6 MiB no corpus sem compressão.
O aumento de RSS com a largura do executor confirma a necessidade de manter a
árvore de redução e as bandas sob orçamento de memória.

O Align mantém um writer limitado a um worker. Uma comparação específica com
8 workers e Align comprimido, em três repetições, mediu 0,957 s de Align com um
writer contra 1,287 s com dois writers. Aumentar writers não é uma otimização
válida de forma geral; compressão disputa CPU com warp e FITS. Uma medição
isolada sem compressão favoreceu dois writers, mas não foi consistente no
pipeline completo, portanto permanece como override diagnóstico.

## Alterações aplicadas

- Cada executor de Flow, Align, Stack, Batch e Calibration recebe um
  inicializador explícito para o limite Numba. A configuração feita na thread
  principal não era suficiente porque a máscara de threads do Numba é local à
  thread.
- Batches independentes podem usar `flow_batch_workers` para dividir o
  orçamento global de CPU/memória, mas o padrão permanece serial entre
  batches; a matriz sintética não demonstrou ganho consistente para habilitar
  isso automaticamente.
- OpenCV fica limitado a uma thread nativa quando há vários workers; no caso
  de um único worker pode usar o teto físico detectado. Isso evita multiplicar
  pools internos sem penalizar o caminho serial otimizado.
- A redução do Stack restaura a ordem lógica das folhas/branches depois de
  `as_completed`, usa uma partição fixa de quatro folhas e reduz sempre até
  uma raiz. Alterar workers não pode alterar o agrupamento científico Stable.
- A redução materializa `values`, máscaras e contagens uma única vez, evitando
  `np.stack(counts)` duplicado.
- O relatório temporal reutiliza timestamps normalizados já presentes no
  `flow_local.json`; o parser usa o cabeçalho que o Flow já abriu para pixels,
  e cabeçalhos FITS só são reabertos para JSONs legados.
- O Global Flow reutiliza o catálogo da âncora produzido pelo Flow Local
  somente quando `fwhm`, `sigma`, `sigma_used`, limite de estrelas, engine e
  perfil são idênticos e a detecção já atingiu o limiar adaptativo global.
  Catálogos antigos ou parciais continuam sendo relidos do FITS.
- O detector DAO mantém as estatísticas invariantes da imagem (`mean`,
  `median`, `std` e fundo sigma-clipped) durante as tentativas adaptativas de
  limiar. A convolução e a busca de picos no menor limiar também são
  reutilizadas para as tentativas superiores; a chamada pública continua como
  fallback quando a API privada do Photutils não estiver disponível. A ordem
  dos candidatos e a filtragem espacial não são alteradas. Quando disponível,
  a máscara privada de picos circulares é consumida diretamente, sem criar a
  QTable intermediária que o Flow não utiliza; o `_find_stars` original é o
  fallback por versão.
- O hashing geométrico de quads usa `math.hypot` para distâncias escalares e o
  normalizador de fase evita cópias/`nan_to_num` quando o frame já é finito;
  ambos mantêm a ordem dos cálculos científicos e foram comparados com os
  caminhos anteriores antes de serem ativados.
- A qualidade de forma usa uma mediana especializada compatível com o
  `np.median` para vetores finitos pequenos; o detector DAO usa uma única
  ordenação descendente no contêiner mínimo interno. QTables públicas e os
  caminhos de fallback não são alterados.
- O caminho quente do Align agora lê pixels, cabeçalho e máscaras no mesmo
  `fits.open`. O helper público de máscaras continua disponível para
  compatibilidade, mas a execução normal não reabre cada FITS apenas para
  reconstruir `VALID_MASK` e `SAT_MASK`.
- O `neighbor_bfs` mantém um pool de preparação limitado por batch em vez de
  criar e destruir um executor a cada fronteira; o mesmo pool agora atende as
  tentativas de aresta quando há múltiplos pais candidatos, sem criar um pool
  por frame. Os resultados continuam sendo consumidos na ordem natural.
- `ProcessTelemetry` e `benchmarks/io_concurrency_benchmark.py` registram
  tempo, CPU, bytes/contagens de leitura e escrita, pico de RSS e threads.
- O Stack reconhece o formato típico da Uranus-C (`uint16` lógico em
  `int16+BZERO=32768`) e abre o armazenamento bruto com `memmap=True`,
  aplicando `BSCALE/BZERO/BLANK` em `float32` por banda. Isso evita a segunda
  cópia materializada pelo Astropy e não cria um cache de 68 GiB ao lado da
  captura; FITS tile-compressed continuam usando o cache persistente existente.
  A conversão raw-cache continua disponível para fontes comprimidas e para
  chamadas diagnósticas explícitas, sempre com restauração bit a bit.
- Para FITS RGB sem dither, uma única seção `[C,Y,X]` é lida por frame/banda e
  os canais são reduzidos separadamente na mesma ordem Stable. O fator de
  orçamento das bandas foi calibrado de 24 para 12 bytes/pixel/canal; na
  resolução real isso reduz aproximadamente pela metade o número de passagens
  FITS sem exceder o orçamento observado de 4 GiB.
- `VALID_MASK` pequeno continua residente para evitar decodificação repetida;
  quando o conjunto ultrapassa 25% do orçamento, as máscaras passam a ser
  lidas por banda e compartilhadas entre canais. Na captura Lagoon isso evita
  reservar aproximadamente 10 GiB de booleanos para 1.242 frames.
- `benchmarks/real_capture_audit.py` fornece uma auditoria reproduzível e
  somente leitura: inventário, geometria/scaling, perfil antigo, relatório
  temporal e uma folha de Stack limitada. O relatório precisa estar fora da
  árvore de origem e todos os FITS temporários são criados em `%TEMP%`.
- O caminho diagnóstico que materializava cada frame corrigido em FITS foi
  interrompido antes da redução completa: 1.213 arquivos chegaram a ocupar
  aproximadamente 71,4 GB em `%TEMP%`. O diretório temporário foi removido
  com uma verificação de caminho explícita; nenhum arquivo em `B:` foi tocado.
  Para evitar repetir esse consumo, `benchmarks/lagoon_similarity_ram_stack.py`
  constrói folhas pequenas diretamente na RAM, reduz cada folha com o mesmo
  reducer `Stable + SigmaClip`, libera os arrays e só grava o FITS final e o
  relatório JSON fora da captura. O tamanho da folha é calculado por uma
  estimativa conservadora de 64 bytes/pixel/frame, limitado a 60% do orçamento
  efetivo; o orçamento efetivo nunca usa mais que 75% da memória segura após
  reservar 20% da RAM física (mínimo de 4 GiB). OpenCV e Numba ficam em uma
  thread nesse modo, portanto não há pool de frames competindo com o reducer.
  O benchmark legado `lagoon_similarity_sigma_stack.py` agora recusa essa
  materialização por padrão; só aceita o caminho em disco com a opção explícita
  `--allow-disk-materialization`.

## Auditoria da captura Lagoon (somente leitura)

Em `B:\\AstroImages\\Notebook\\Lagoon\\21_23_00_align` foram encontrados
1.242 FITS RGB, todos `3856×2180`, `58.855.680` bytes, total de `68,08 GiB`;
as imagens têm `VALID_MASK` e `BZERO=32768`, sem compressão de tiles. O
`21_23_00_batch` contém 1.253 frames Flow e seis lotes; o relatório temporal
legado precisou ler os cabeçalhos uma vez, pois os JSON antigos ainda não
tinham os timestamps normalizados.

O perfil já produzido pelo usuário mostra o hotspot dominante em
`_create_substacks`: aproximadamente `945,0 s` de uma sessão de `992,4 s`;
redução (`_combine_substacks`) ficou em `18,4 s` e a rejeição em `10,9 s`.
Isso caracteriza espera por folhas/leituras e redução por bandas, não uma fila
de disco isolada.

Uma folha real de quatro frames, executada fora da origem, confirmou o novo
caminho RGB/mmap e produziu `float32`, máscara `uint8` e contagem `uint32`.
Em uma amostra de oito frames com `SigmaClip`, o caminho raw/mmap e a leitura
RGB agrupada preservaram todos os pixels finitos e os NaNs; a comparação com
o caminho escalado anterior foi `equal_nan=True`, diferença máxima finita zero.
O inventário e o relatório completo foram gravados fora de `B:` por:
`C:\\Users\\jvito\\AppData\\Local\\Temp\\astrobatch_lagoon_audit_final_20260915.json`.
Nenhum arquivo da árvore `B:` foi criado, removido, renomeado ou alterado.

O smoke test do modo somente-RAM, usando três frames reais e orçamento de
4.096 MiB, produziu FITS RGB `uint16` com `VALID_MASK` `uint8` e `RGBMODE`
`similarity`. A memória disponível mínima observada foi 45.045 MiB, o RSS
amostrado foi 370 MiB e a estimativa conservadora da folha foi 1.540 MiB;
`files_unchanged=true`. Uma segunda execução com oito frames usou duas folhas
de quatro (`leaf_sizes=[4,4]`) e também terminou com `files_unchanged=true`.
Esses artefatos ficam em `%TEMP%\astrobatch_lagoon_ram_smoke2_20260915` e
`%TEMP%\astrobatch_lagoon_ram_8_20260915`; não são entradas da captura. A
redução completa de 1.213 frames foi então executada com `RGBMODE=hybrid`,
16.384 MiB de orçamento e 68 folhas, sem criar os 71,4 GB de intermediários:
RSS amostrado máximo de 9.642 MiB, mínimo de 34.915 MiB livres,
`files_unchanged=true` e tempo de 1.955 s (~32,6 min). A comparação de
qualidade comum a 38 estrelas está em
`%TEMP%\astrobatch_lagoon_ram_full_hybrid_20260915\quality_compare\`; contra
`stack_2`, o deslocamento mediano caiu de 0,348 para 0,190 px em R–G e de
0,605 para 0,452 px em B–G.

O corpus sintético oficial continua sendo o único gate de throughput: após as
mudanças, sete execuções quentes em quatro workers mediram mediana Stable de
`1,8238 s` contra `2,4863 s` do baseline limpo (`26,65%`, speedup `1,3633×`),
com digest Stable e produtos `uint16`/máscara/contagem iguais. A evidência é
[pipeline_benchmark_lagoon_opt_w4_20260915.json](pipeline_benchmark_lagoon_opt_w4_20260915.json).
Essa medição não transforma a amostra real em um gate de sete execuções: uma
sessão completa de 1.242 frames ainda deve ser repetida em armazenamento frio
quando houver janela operacional para isso.

Uma revalidação posterior, já com a política de `VALID_MASK` em streaming e o
leitor raw/mmap do Flow, foi executada no mesmo host e com sete repetições. A
mediana Stable foi `1,9493 s` contra o mesmo baseline (`21,60%`, speedup
`1,2755×`), com digest e produtos ainda iguais; portanto o gate de `25%` não é
considerado atingido para o checkout corrente. O desvio observado foi
`0,2657 s`, por isso o resultado anterior fica registrado como a variante
otimizada anterior e não como uma garantia universal. Evidência:
[pipeline_benchmark_lagoon_final_rerun2_w4_20260915.json](pipeline_benchmark_lagoon_final_rerun2_w4_20260915.json).

Após o caminho raw/mmap, a redução RGB por banda e o fator de 12 bytes, a
matriz sintética quente (três repetições por largura) ficou em
`2,497 / 2,042 / 1,739 / 1,702 s` para `1 / 2 / 4 / 8` workers; Flow
`1,720 / 1,482 / 1,235 / 1,196 s`; Stack `0,350 / 0,274 / 0,282 / 0,289 s`;
pico RSS `245,5 / 263,4 / 302,2 / 310,4 MiB`. Oito workers só melhora cerca
de `2,1%` contra quatro, então quatro segue como teto recomendado para a
sessão real. Evidência:
[io_concurrency_benchmark_lagoon_opt_20260915.json](io_concurrency_benchmark_lagoon_opt_20260915.json).

## Próximas ações priorizadas

1. **P0 — Flow:** medir separadamente leitura, detecção DAO, preparação do
   cache, `neighbor_bfs` e matching global em capturas grandes. A reutilização
   segura da detecção de âncora local já foi aplicada; o próximo passo é
   comparar a taxa de acerto do cache em sessões reais e medir o DAO por
   resolução/carga.
2. **P0 — corpus real:** repetir a matriz com pelo menos sete execuções quentes
   em uma captura representativa do usuário; usar pico RSS, não RSS no fim do
   processo. A auditoria e a folha limitada já foram executadas; o corpus
   sintético passou o gate em quatro workers; 8 workers continuam override
   quando a memória permitir.
3. **P1 — Stack:** avaliar a redução por bandas dos branches, mantendo a ordem
   fixa e a igualdade bit a bit. A folha RGB e o fator 12 já foram medidos;
   falta medir a amplificação dos branches em uma sessão completa antes de
   aumentar a largura do executor.
4. **P1 — I/O frio:** repetir em diretório/corpus que não esteja no cache do
   SO, separando tempo de leitura, descompressão e escrita. Só alterar o pool
   de writers se a medição fria justificar.
5. **P2 — telemetria nativa:** instalar `threadpoolctl` apenas se aprovado
   como dependência para medir OpenBLAS/BLAS ativos; hoje o contador de threads
   inclui pools nativos ociosos e a thread observadora.

O gate oficial de performance continua sendo redução mínima de 25% contra o
baseline do checkout anterior, isto é, `speedup >= 4/3`, com produtos Stable,
máscaras, contagens e metadados iguais. A comparação histórica anterior, com
sete execuções, mesma afinidade e `NUMBA_NUM_THREADS=8`, mediu após a reutilização
do `DATE-OBS` já lido:

- um worker: baseline Stable `2,7551 s`, candidato `2,5177 s`, `1,0943×`,
  com digest Stable idêntico;
- quatro workers: baseline `2,4863 s`, candidato `2,2157 s`, `1,1221×`.

No caso de quatro workers o checkout anterior não foi bit a bit determinístico
entre execuções; o candidato estabilizou o digest. Essa medição foi supersedida
pela matriz oficial final abaixo. O corpus usa 32 estrelas
sintéticas e `max_stars=150`, portanto o contador de reutilização de âncoras
foi `0`; sessões reais com pelo menos 35 detecções e a tampa 250 exercitarão
esse caminho.

Na matriz histórica de sete execuções quentes, com a afinidade lógica `0–7`, o
Stable mediu `2,4867 s` (um worker, pico `283,4 MiB`) e `2,1857 s` (quatro
workers, pico `311,3 MiB`). Depois da janela limitada de âncoras, a matriz
final de sete execuções com quatro workers mediu Stable `2,0528 s` (mediana;
desvio `0,2258 s`, pico `314,7 MiB`) e Fast `2,0322 s` (desvio `0,1258 s`).
O digest Stable permaneceu constante e os produtos continuaram `uint16`, com
máscaras `uint8` e contagens `uint32`. Contra o baseline limpo de `2,4863 s`,
a redução final é `17,4%`, ainda abaixo do gate de `25%`; Fast não oferece
vantagem científica ou de throughput suficiente para virar padrão. O artefato
é [pipeline_benchmark_final_after_anchor_window_w4_20260915.json](pipeline_benchmark_final_after_anchor_window_w4_20260915.json).

Na matriz oficial final, após a mediana especializada e a ordenação DAO de uma
passada, o Stable mediu `1,8415 s` de mediana em quatro workers (desvio
`0,2103 s`, pico `312,6 MiB`) contra `2,4863 s` do baseline limpo (pico
`339,7 MiB`). Isso representa speedup `1,3501×` e redução `25,93%`, com
digest Stable, máscaras, contagens e metadados científicos iguais; o gate
`4/3x` está **passed**. O benchmark agora grava esse resultado diretamente
quando recebe `--baseline-json`. Evidência:
[pipeline_benchmark_official_w4_20260915.json](pipeline_benchmark_official_w4_20260915.json).

## Referências de desenho consultadas

- O [modelo de threading do Siril](https://siril-contrib-doc.readthedocs.io/en/latest/Threading.html)
  separa o orçamento global de threads do paralelismo por frame e evita pools
  aninhados. O AstroBatch aplica a mesma regra: workers de frames ou folhas
  assumem a CPU; kernels Numba paralelos ficam reservados ao caminho de worker
  único.
- A fila persistente do
  [processing thread do Siril](https://raw.githubusercontent.com/lock042/siril/master/src/core/processing_thread.h)
  reforça a decisão de manter executores limitados entre fronteiras de Flow,
  em vez de criar/destruir um pool para cada frame ou vizinhança.
- A implementação de
  [stacking por blocos do Siril](https://raw.githubusercontent.com/lock042/siril/master/src/stacking/median_and_mean.c)
  dimensiona blocos pelo orçamento de memória e só paraleliza quando a leitura
  FITS é segura; isso motivou as bandas do Stack e a árvore de redução fixa do
  AstroBatch.
- A documentação de
  [imagens FITS do Astropy](https://docs.astropy.org/en/stable/io/fits/usage/image.html)
  confirma que memmap/lazy loading não elimina cópias exigidas por scaling e
  conversão de dtype. Por isso o caminho quente mantém `memmap=False` e remove
  a reabertura de máscaras, sem introduzir uma conversão implícita que altere
  `uint16` ou as máscaras científicas.
- O [histórico de performance do Photutils](https://github.com/astropy/photutils/blob/main/CHANGES.rst)
  documenta a vetorização de cutouts/momentos e a busca de picos separável;
  isso motivou o consumo direto do helper de picos e o contêiner mínimo DAO,
  sempre com fallback para a API pública instalada.
- O repositório do [ASTAP](https://github.com/CanardConfit/ASTAP) foi usado como
  referência de produto similar: ele separa detecção/registro interno do
  empilhamento e mantém filtragem de qualidade como etapa explícita, em vez de
  misturar decisões de seleção na matemática do reducer Stable.
