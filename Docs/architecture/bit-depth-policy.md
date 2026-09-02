# Plano de Ação — Profundidade de Bits FITS

## Decisão técnica

- Aquisição, alinhamento e saída padrão do stack usam FITS de 16 bits.
- `float32` é usado apenas em RAM durante Dark/Flat/mediana; nenhum FITS
  intermediário de 32 bits é criado.
- A calibração percorre todos os LIGHTs, encontra uma faixa mínima/máxima
  global após Dark/Flat e grava cada resultado como `uint16` nessa mesma faixa.
- Stack, alinhamento e aquisição seguem exclusivamente o padrão FIT 16-bit.

## Etapa 1 — Padrões e regressão — Concluída

- Alterado o padrão do modelo, parser e interface de Stack para `16-bit`.
- Adicionados testes para o padrão 16-bit, a conversão final e a preservação
  FIT 16-bit da calibração.
- Substituída a gravação de calibração/Masters em float32 por FIT 16-bit com
  `CALMIN`, `CALMAX` e `CALNORM` no cabeçalho.
- Removida a opção de saída 32-bit do Stack; configurações antigas são
  normalizadas silenciosamente para `16-bit`.

## Etapa 2 — Contrato de saída — Concluída

- Documentada a faixa global de normalização da calibração nos cabeçalhos FITS.
- A normalização ocorre somente na calibração; as etapas posteriores recebem
  dados já normalizados em 16-bit.

## Etapa 3 — Validação em dados reais — Pendente

- Comparar histograma, saturação e tamanho de arquivo de uma sessão RAW16.
- Confirmar que todos os FITS persistidos na sessão permanecem em 16-bit.
