# Revisão de qualidade — Nebulosa da Lagoa / SV48P 102 mm + Uranus-C

Data da revisão: 2026-09-15  
Fonte: `B:\AstroImages\Notebook\Lagoon` (somente leitura)

Nenhum arquivo da captura foi criado, removido, renomeado ou alterado. As
visualizações e medições auxiliares foram gravadas somente em `%TEMP%`.

## Evidência observada

- A sessão `21_23_00` contém 1.253 frames de 0,5 s, ganho 210, `GRBG`,
  focal 601,4 mm e escala aproximada de 0,994 arcsec/pixel.
- O Flow já estima transformações de similaridade (translação, rotação e
  escala) por frame. O Global Flow também registra a transformação entre os
  seis lotes. As rotações de lote foram `0,000`, `-0,422`, `-0,413`, `-0,319`,
  `-0,258` e `-0,185` graus; a amplitude total é `0,422°`, todas as seis
  decisões foram aceitas e o RMS máximo foi aproximadamente `0,824` pixel.
- Portanto, a rotação de campo já é corrigida durante Align. Aplicar uma
  segunda derrotação no Stack só interpolaria os pixels novamente e pode
  aumentar halos. O relatório temporal agora publica esses valores em
  `field_rotation`, sempre com `review_only=true`.
- Nos transforms locais, a rotação mediana é `-0,0009°` e o maior módulo é
  aproximadamente `0,0866°`. Como o Global Flow já remove a variação de
  `0,422°` entre lotes, o trailing residual observado não parece ser dominado
  por rotação de campo; rejeição seletiva e correção óptica/cromática são mais
  promissoras.
- Nos metadados de forma do Flow, a mediana foi roundness `0,6434`, FWHM
  `7,8756` e elongação `1,5543`. Há 28 frames alinhados com baixa confiança e
  11 frames rejeitados pelo Flow. A captura tem sinais de arrasto/seeing, mas
  roundness é uma métrica aproximada, não uma medição de PSF calibrada.
- O filtro atualmente configurado no checkout (`min_roundness=0,65`) é
  agressivo para esta sessão: a distribuição observada colocaria cerca de
  61% dos frames abaixo desse corte. Um corte inicial de `0,60` deixaria a
  revisão manual concentrada em aproximadamente 4% dos frames; o limite
  robusto mediana−2×1,4826×MAD é cerca de `0,595`, aproximadamente 2,3%.
  Esses números são orientação, não exclusão automática.
- A inspeção visual do stack existente mostra halos/franjas azuis e cianos
  em estrelas brilhantes, além de fundo ruidoso e uma região central muito
  brilhante. Os FITS alinhados antigos não registram `RGBMODE` nem modelos
  cromáticos; eles foram produzidos antes da correção de escala/rotação por
  canal.
- A visualização temporal também plota roundness e a linha de corte robusta
  calculada para a própria sessão. Essa linha apenas marca grupos/frame para
  revisão; não vira uma regra automática de exclusão.
- O `MasterDark.fits` é compatível em câmera, geometria, exposição (`0,5 s`),
  ganho (`210`), focal (`601,4 mm`) e padrão Bayer (`GRBG`). A temperatura do
  dark é `28,6 °C`, contra `29,0–29,2 °C` nos lights. Em três lights, a
  dispersão robusta do sinal bruto caiu aproximadamente de `18,55` para
  `16,76` (unidades ADU) após subtração simulada do dark; a calibração também
  remove os pixels quentes raros do master. Isso deve ser feito antes de
  debayer/Flow/Align quando se parte dos lights crus. Os lights crus em
  `21_23_00` não têm `CALNORM`; não se deve aplicar esse dark novamente aos
  produtos que já estejam calibrados.
- Os FITS de `21_23_00_align` trazem `CALNORM=True`, `CALMIN=-13008` e
  `CALMAX=39833`. Isso prova que há normalização de calibração no produto
  alinhado, embora o cabeçalho não registre qual master foi usado. A origem
  exata do dark deve ser confirmada antes de qualquer nova calibração.
- Os dois stacks existentes confirmam que o corte muda radicalmente o
  resultado: `21_23_00_stack` usou `roundness=0,65` e reteve apenas `480/1242`
  frames; `21_23_00_stack_2` usou `roundness=0,55` e reteve `1241/1242`.
  O segundo stack é o melhor ponto de partida para preservar integração, mas
  ainda foi produzido sem o novo modelo cromático. Seus inputs já carregam
  normalização de calibração; não é correto recalibrá-los sem confirmar a
  procedência do master.
- No primeiro stack, os 762 frames excluídos tinham roundness mediana `0,631`,
  mas qualidade mediana `4,34`, ligeiramente maior que os selecionados
  (`4,04`). Isso mostra que o ranking de qualidade sozinho não identifica bem
  trailing; roundness/FWHM devem continuar explícitos na revisão.
- No recorte do `batch_001`, o limite robusto `0,595` manteve os 31 frames,
  enquanto `0,65` manteve 18. A roundness de segundo momento do recorte mudou
  de aproximadamente `0,927` para `0,945`, sem ganho claro de FWHM. O corte
  estrito melhora a circularidade, mas custa integração rapidamente.

## Alteração entregue

O Align agora mantém o comportamento anterior como padrão (`translation`) e
oferece os modos opt-in `similarity` e `hybrid` no controle nativo **Modelo
cromático**. `similarity` usa o canal verde como referência e estima, para
vermelho e azul, uma transformação limitada de translação + escala + rotação.
`hybrid` mantém a translação conservadora em R e usa a similaridade apenas em
B; foi o melhor compromisso medido nesta Uranus-C. A máscara de validade e a
máscara de saturação acompanham a mesma matriz, evitando que a correção marque
bordas inválidas como pixels científicos.

Em uma amostra real do primeiro FITS alinhado, o modelo encontrou correções
subpixel de aproximadamente R `(-0,74,-0,25)` e B `(-0,08,-0,30)` pixel, com
confiança 1,0. Em uma fixture sintética com escala e translação conhecidas,
o erro quadrático médio caiu para menos de 20% do erro sem correção. A correção
não é aplicada silenciosamente a produtos antigos: é necessário
selecionar `similarity` ou `hybrid` e executar Align para uma saída nova.

Como verificação prática, processei somente os 31 frames do `batch_001` em
memória, sem criar FITS na origem. O modelo similarity foi aceito nos 31
frames. Em um recorte de estrela, a separação de centróide R–G caiu de cerca
de `1,04` para `0,98` pixel e B–G de `0,39` para `0,33` pixel. Isso confirma
correção mensurável, mas não substitui um stack completo: os halos residuais
dependem de flats, saturação e do seeing.

Também executei uma redução diagnóstica completa fora de `B:` usando os 1.213
frames com roundness `>= 0,595`. A máscara de validade foi transformada com a
mesma matriz RGB antes da média; sem essa interseção, bordas preenchidas por
`cv2.warpAffine` criavam faixas artificiais. O relatório registrou 1.063
frames com dois modelos similarity completos e 150 em fallback seguro. A
imagem derivada está em
`%TEMP%\astrobatch_lagoon_similarity_masked_20260915\` e confirmou
`files_unchanged=true`. Ela é deliberadamente uma média mascarada de
diagnóstico, não um substituto do `SigmaClip` do Stack: a média expõe
banding/fundo fixo que precisa ser tratado por calibração e redução robusta.
Não considero essa execução uma prova de ganho visual do stack final; ela
serve para validar o caminho cromático e a segurança de leitura da captura.
Após a guarda adicional de deslocamento nos quatro cantos, uma amostra de 100
frames aceitou o modelo completo em `100/100`, com cobertura mediana de 100
frames por pixel, zero pixels sem cobertura e `files_unchanged=true`.

### Redução limitada pela RAM

O primeiro protótipo de SigmaClip escrevia uma FITS corrigida por frame antes
de chamar o Stack. Em uma tentativa com 1.213 frames, isso consumiu cerca de
71,4 GB em `%TEMP%`; a execução foi cancelada antes de alterar a origem e o
diretório temporário foi removido. A rota final para esse diagnóstico é
`benchmarks/lagoon_similarity_ram_stack.py`: ela lê e corrige uma folha
pequena, reduz na RAM com `Stable + SigmaClip`, descarta os arrays e só grava o
produto final e o sidecar fora de `B:`. Há uma guarda de memória baseada em
`psutil` que preserva 20% da RAM física (mínimo de 4 GiB), aplica o orçamento
efetivo a 60% para as folhas e reduz automaticamente a folha em caso de
pressão; OpenCV/Numba ficam limitados a uma thread para evitar
oversubscription.

O smoke test real com três frames e orçamento de 4.096 MiB gerou FITS RGB
`uint16`, `VALID_MASK` `uint8`, `RGBMODE=similarity`, RSS amostrado de 370 MiB,
estimativa conservadora de 1.540 MiB e `files_unchanged=true`. A árvore de oito
frames também passou com duas folhas de quatro. Em seguida, a sessão completa
foi processada pela rota RAM com `RGBMODE=hybrid`: 1.213 frames, 68 folhas de
18 (última de 7), 1.955 s (~32,6 min), pico RSS amostrado de 9.642 MiB, mínimo de 34.915
MiB livres e `files_unchanged=true`. O produto está fora de `B:` em
`%TEMP%\astrobatch_lagoon_ram_full_hybrid_20260915\`.
A tabela de centróides e a prévia comparativa estão em
`%TEMP%\astrobatch_lagoon_ram_full_hybrid_20260915\quality_compare\`.

Na comparação de 38 estrelas comuns com `stack_2`, o híbrido completo mediu
roundness mediana `0,671` e FWHM `6,46 px` (baseline `0,689` e `6,57 px`).
Os deslocamentos R–G e B–G caíram de `0,348` para `0,190 px` e de `0,605`
para `0,452 px`, respectivamente. Como a integração completa ainda contém
frames de seeing/trailing, o ensaio de revisão com corte `0,65` recupera
roundness `0,694` e reduz o número de frames para 480; a escolha final deve
ser manual, conforme a prioridade entre sinal e nitidez.

## Procedimento recomendado para uma nova redução

1. Preserve `B:` como origem e escolha uma pasta de saída nova fora dela.
2. Execute Flow/Align com `quality_gate` ativo, **RGB registration** ativo e
   **Modelo cromático = `hybrid`** para esta Uranus-C (R por translação, B por
   similaridade). Mantenha Stable e a interpolação já
   validada pelo seu fluxo; a mudança cromática é a única alteração necessária
   nesta etapa.
3. No Stack, gere primeiro uma seleção de diagnóstico sem excluir arquivos.
   Revise a distribuição e comece com `roundness >= 0,60` (ou o limite robusto
   próximo de `0,595`), comparando também FWHM e contagem de estrelas. Não use
   `0,65` como regra universal para esta sessão.
4. Compare um stack com todos os frames aceitos e outro com o subconjunto
   revisado. A recomendação é apenas para revisão manual; nenhum grupo
   temporal é removido, reponderado ou enviado automaticamente ao Stack.
5. Use o `MasterDark.fits` existente se a calibração corresponder à Uranus-C,
   ganho e exposição. A listagem da sessão não mostrou um conjunto de flats;
   flats adequados continuam sendo a próxima melhoria para gradiente,
   vinhetagem e halos de fundo.

Para gerar o diagnóstico cromático sem ocupar dezenas de gigabytes no disco,
use a rota limitada pela RAM (a saída deve ficar fora de `B:`):

```
.venv/Scripts/python.exe benchmarks/lagoon_similarity_ram_stack.py `
  --source B:\AstroImages\Notebook\Lagoon\21_23_00_align `
  --flow-root B:\AstroImages\Notebook\Lagoon\21_23_00_batch `
  --output-dir C:\Users\jvito\AppData\Local\Temp\astrobatch_lagoon_ram_final `
  --memory-budget-mb 4096 `
  --rgb-mode hybrid
```

O comando mantém uma reserva de RAM, reduz o tamanho da folha se necessário e
grava somente `lagoon_similarity_ram_sigma_stack.fits` e
`lagoon_similarity_ram_report.json` no diretório de saída.

Na captura já alinhada, a variante `--rgb-mode hybrid` com o corte
`--roundness 0.65` foi o melhor resultado de revisão: 480 frames, roundness mediana
0,694 (contra 0,689 no `stack_2`) e FWHM 6,44 px (contra 6,57 px). Nos mesmos
38 candidatos, o deslocamento cromático mediano caiu de 0,348 para 0,247 px
em R–G e de 0,605 para 0,485 px em B–G. Isso é uma sugestão para revisão
manual; não remove frames da sessão e não altera o Stack padrão.

## O que não deve ser feito nesta entrega

- Não aplicar uma segunda rotação de campo no Stack.
- Não descartar automaticamente os blocos de 15 minutos nem substituir
  frames sem horário; o agrupamento temporal é somente informativo.
- Não interpolar darks/flats por temperatura nem prever rotação por modelo de
  tempo; isso permanece fora do escopo.
- Não escrever sidecars, caches ou stacks dentro de `B:` durante a validação.

## Próxima decisão necessária

Para escolher o corte final, falta uma preferência científica: você quer
maximizar a nitidez aceitando perder aproximadamente 2–4% dos frames ruins,
ou preservar a maior integração possível aceitando mais halos/arrasto? Também
é importante confirmar se a montagem é equatorial ou alt-azimutal (e se há
derrotador); isso define se a amplitude de `0,422°` deve ser apenas auditada
ou usada como alerta operacional.
