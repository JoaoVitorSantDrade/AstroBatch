# Compatibilidade FIT/FITS e TIF/TIFF

O pipeline aceita imagens mono e RGB nas duas famílias:

- FIT/FITS/FTS: a ciência fica no HDU de imagem e as máscaras continuam em
  extensões `VALID_MASK`/`SAT_MASK`.
- TIF/TIFF: a imagem linear `uint16` fica no TIFF; máscaras, cabeçalhos FITS e
  telemetria ficam em sidecars atômicos (`.astrobatch.npz` e `.json`).

Uma sessão é validada antes de alocar workers. Misturar FIT/FITS e TIF/TIFF é
recusado para evitar uma saída ambígua e uma calibração inconsistente.

## Align compacto e Stack final

O padrão `batch_compact` mantém um acumulador por vez em RAM, grava
`batch_stack.<ext>` e libera o estado antes de iniciar o próximo batch. O Stack
descobre os manifests `batch_stack.<ext>.json` e combina os masters ponderando
as contagens/weights persistidas, sem tratar cada batch como uma única
exposição. Frames alinhados individuais só são gravados quando
`keep_aligned_frames` está habilitado.

Mean/Sum/QualityWeightedMean com estado compatível usam a fusão associativa
persistida. Mediana e rejeições são explicitamente hierárquicas: primeiro o
master de cada batch, depois a decisão sobre os masters no Stack. Nenhum frame
é excluído ou reponderado automaticamente por essa escolha.

O acumulador estima o estado e reserva margem para temporários antes de
alocar. Se a imagem exceder o limite seguro de RAM, o batch falha de forma
controlada e informa como reduzir a resolução/lote ou aumentar o orçamento;
não há tentativa de preencher toda a memória do sistema.

## Saída e edição externa

O sufixo da saída é normalizado para a família da entrada: uma sessão iniciada
em TIF/TIFF termina em `.tif` (ou `.tiff` solicitado), e uma sessão iniciada em
FIT/FITS termina em `.fit`/`.fits`. Os produtos são lineares `uint16`; os
sidecars não alteram o plano científico que pode ser aberto em ferramentas
externas como o Siril. Ao retornar ao AstroBatch, os sidecars restauram
máscaras e metadados necessários para continuar o fluxo.

As medições históricas anteriores ao modo compacto registram reduções acima de
25% no caminho de arquivos individuais, com digest Stable, máscaras, contagens,
`uint16` e probes AVX2/YMM idênticos. Elas permanecem como referência, mas não
são usadas para declarar o gate do modo compacto.

Na revalidação do checkout atual, o corpus versionado foi executado sete vezes
por variante com quatro workers. O modo compacto mediu mediana `Stable` de
`1,8136 s` contra `2,4863 s` do baseline (`27,05%`, speedup `1,3709x`), com
pico de RSS de aproximadamente `326,8 MiB`. Digests, máscaras, contagens,
`uint16` e instruções AVX2/YMM permaneceram idênticos. O gate obrigatório de
`25%` está **atingido** nesta variante; frio/JIT e importação ficaram fora da
medição quente. O JSON completo foi gravado fora do checkout durante a
verificação (`%TEMP%\\astrobatch_pipeline_current_quantized.json`).
