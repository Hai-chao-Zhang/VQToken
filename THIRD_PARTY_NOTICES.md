# Third-party notices

This source repository contains modified, vendored source code. The runtime
wheel and source distribution contain only the `llava/` and `VQToken/`
packages. Copyright notices and license terms retained in individual source
files remain in effect.

## LLaVA-NeXT / LLaVA-OneVision

- Source: <https://github.com/LLaVA-VL/LLaVA-NeXT>
- License: Apache License 2.0
- License text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Vendored revision: not recorded in this repository

The `llava/` package is derived from LLaVA-NeXT and related LLaVA work. Some
files also retain notices for FastChat, Stanford Alpaca, Hugging Face
Transformers, OpenAI CLIP, and other upstream implementations.

## TRL

- Source: <https://github.com/huggingface/trl>
- License: Apache License 2.0
- License text: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Vendored version: `0.7.11.dev0`

The source checkout's vendored `trl/` package retains its upstream copyright
and attribution headers. It is not included in the runtime distribution.

## lmms-eval

- Source: <https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.2.4>
- License: MIT for the main pipeline; Apache License 2.0 for upstream
  `lmms_eval/models/` and `lmms_eval/tasks/`
- License text: [`LICENSES/lmms-eval-v0.2.4-LICENSE.txt`](LICENSES/lmms-eval-v0.2.4-LICENSE.txt)
- Vendored version: `0.2.4`

The source checkout's vendored `lmms_eval/` package is derived from lmms-eval,
which in turn retains relevant attribution to lm-evaluation-harness. It is not
included in the runtime distribution.

## OpenAI CLIP

- Source: <https://github.com/openai/CLIP>
- License: MIT
- License text: [`LICENSES/OpenAI-CLIP-MIT.txt`](LICENSES/OpenAI-CLIP-MIT.txt)

The LLaVA vision-encoder sources include code derived from OpenAI CLIP.

## VQToken-specific code

Original VQToken-specific code is distributed under the repository's BSD
3-Clause License in [`LICENSE`](LICENSE).

This top-level inventory does not replace file-level notices in the vendored
source tree.
