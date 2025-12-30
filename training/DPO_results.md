Starting VGPT2 v3 Stage 2: DPO Training
========================================

Repo root: C:\Github\LLM_fine-tuning

SFT adapter found. Starting DPO...

DPO pairs: 1,427
Epochs: 2
Expected time: 30-60 minutes

Activating venv: C:\Github\LLM_fine-tuning\venv
[INFO|2025-12-30 19:57:05] llamafactory.hparams.parser:465 >> Process rank: 0, world size: 1, device: cuda:0, distributed training: False, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file vocab.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\vocab.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file merges.txt from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\merges.txt
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file tokenizer.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\tokenizer.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file tokenizer_config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\tokenizer_config.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:06,727 >> loading file chat_template.jinja from cache at None
[INFO|tokenization_utils_base.py:2364] 2025-12-30 19:57:06,958 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:765] 2025-12-30 19:57:08,647 >> loading configuration file config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\config.json
[INFO|configuration_utils.py:839] 2025-12-30 19:57:08,649 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,222 >> loading file vocab.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\vocab.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,222 >> loading file merges.txt from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\merges.txt
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,222 >> loading file tokenizer.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\tokenizer.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,223 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,223 >> loading file special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,223 >> loading file tokenizer_config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\tokenizer_config.json
[INFO|tokenization_utils_base.py:2095] 2025-12-30 19:57:09,223 >> loading file chat_template.jinja from cache at None
[INFO|tokenization_utils_base.py:2364] 2025-12-30 19:57:09,523 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|2025-12-30 19:57:09] llamafactory.data.loader:143 >> Loading dataset vgpt2_v3_dpo.json...
Setting num_proc from 16 back to 1 for the train split to disable multiprocessing as it only contains one shard.
Generating train split: 1427 examples [00:00, 94880.82 examples/s]
Converting format of dataset (num_proc=16): 100%|██████████████████████████| 1427/1427 [00:10<00:00, 141.43 examples/s]
Running tokenizer on dataset (num_proc=16): 100%|██████████████████████████| 1355/1355 [00:09<00:00, 139.16 examples/s]
training example:
chosen_input_ids:
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 12292, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 448, 348, 47, 5609, 2230, 7111, 261, 151645, 198, 151644, 77091, 198, 73594, 3544, 198, 4858, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 9610, 915, 11, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 65453, 7339, 11, 348, 47, 5609, 2230, 7111, 261, 65453, 7339, 11, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 30093, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 4769, 320, 8996, 8044, 340, 58124, 13069, 348, 47, 5609, 2230, 7111, 261, 4769, 320, 8996, 8044, 340, 220, 6197, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 836, 706, 9490, 284, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 26513, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 52114, 284, 569, 7339, 198, 13874, 19324, 16429, 279, 4583, 7214, 1376, 5025, 25351, 821, 16614, 13, 151645, 198]
chosen_inputs:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Join vPRAUEmployerATOItems with vPRAUEmployer<|im_end|>
<|im_start|>assistant
```sql
SELECT vPRAUEmployerATOItems.KeyID, vPRAUEmployerATOItems.PRCo, vPRAUEmployer.PRCo, vPRAUEmployer.TaxYear
FROM vPRAUEmployerATOItems WITH (NOLOCK)
INNER JOIN vPRAUEmployer WITH (NOLOCK)
  ON vPRAUEmployerATOItems.TaxYear = vPRAUEmployer.TaxYear
WHERE vPRAUEmployerATOItems.Co = @Co
```

Using the complete foreign key relationship ensures data integrity.<|im_end|>

chosen_label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 73594, 3544, 198, 4858, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 9610, 915, 11, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 65453, 7339, 11, 348, 47, 5609, 2230, 7111, 261, 65453, 7339, 11, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 30093, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 4769, 320, 8996, 8044, 340, 58124, 13069, 348, 47, 5609, 2230, 7111, 261, 4769, 320, 8996, 8044, 340, 220, 6197, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 836, 706, 9490, 284, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 26513, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 52114, 284, 569, 7339, 198, 13874, 19324, 16429, 279, 4583, 7214, 1376, 5025, 25351, 821, 16614, 13, 151645, 198]
chosen_labels:
```sql
SELECT vPRAUEmployerATOItems.KeyID, vPRAUEmployerATOItems.PRCo, vPRAUEmployer.PRCo, vPRAUEmployer.TaxYear
FROM vPRAUEmployerATOItems WITH (NOLOCK)
INNER JOIN vPRAUEmployer WITH (NOLOCK)
  ON vPRAUEmployerATOItems.TaxYear = vPRAUEmployer.TaxYear
WHERE vPRAUEmployerATOItems.Co = @Co
```

Using the complete foreign key relationship ensures data integrity.<|im_end|>

rejected_input_ids:
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 12292, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 448, 348, 47, 5609, 2230, 7111, 261, 151645, 198, 151644, 77091, 198, 73594, 3544, 198, 4858, 1213, 30093, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 198, 75486, 348, 47, 5609, 2230, 7111, 261, 6197, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 836, 706, 9490, 284, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 73594, 151645, 198]
rejected_inputs:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Join vPRAUEmployerATOItems with vPRAUEmployer<|im_end|>
<|im_start|>assistant
```sql
SELECT *
FROM vPRAUEmployerATOItems
JOIN vPRAUEmployer ON vPRAUEmployerATOItems.TaxYear = vPRAUEmployer.TaxYear
```<|im_end|>

rejected_label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 73594, 3544, 198, 4858, 1213, 30093, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 198, 75486, 348, 47, 5609, 2230, 7111, 261, 6197, 348, 47, 5609, 2230, 7111, 261, 58108, 4353, 836, 706, 9490, 284, 348, 47, 5609, 2230, 7111, 261, 836, 706, 9490, 198, 73594, 151645, 198]
rejected_labels:
```sql
SELECT *
FROM vPRAUEmployerATOItems
JOIN vPRAUEmployer ON vPRAUEmployerATOItems.TaxYear = vPRAUEmployer.TaxYear
```<|im_end|>

Running tokenizer on dataset (num_proc=16): 100%|███████████████████████████████| 72/72 [00:09<00:00,  7.47 examples/s]
eval example:
chosen_input_ids:
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 12292, 348, 9501, 14940, 929, 448, 293, 58367, 1162, 151645, 198, 151644, 77091, 198, 73594, 3544, 198, 4858, 348, 9501, 14940, 929, 808, 11604, 535, 91981, 11, 348, 9501, 14940, 929, 77268, 7339, 11, 293, 58367, 1162, 35989, 519, 2808, 11, 293, 58367, 1162, 727, 535, 929, 198, 30093, 348, 9501, 14940, 929, 4769, 320, 8996, 8044, 340, 58124, 13069, 293, 58367, 1162, 4769, 320, 8996, 8044, 340, 220, 6197, 348, 9501, 14940, 929, 3503, 3706, 535, 929, 284, 293, 58367, 1162, 727, 535, 929, 198, 26513, 348, 9501, 14940, 929, 52114, 284, 569, 7339, 198, 13874, 19324, 16429, 279, 4583, 7214, 1376, 5025, 25351, 821, 16614, 13, 151645, 198]
chosen_inputs:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Join vSMCostType with bJCCT<|im_end|>
<|im_start|>assistant
```sql
SELECT vSMCostType.SMCostTypeID, vSMCostType.SMCo, bJCCT.PhaseGroup, bJCCT.CostType
FROM vSMCostType WITH (NOLOCK)
INNER JOIN bJCCT WITH (NOLOCK)
  ON vSMCostType.JCCostType = bJCCT.CostType
WHERE vSMCostType.Co = @Co
```

Using the complete foreign key relationship ensures data integrity.<|im_end|>

chosen_label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 73594, 3544, 198, 4858, 348, 9501, 14940, 929, 808, 11604, 535, 91981, 11, 348, 9501, 14940, 929, 77268, 7339, 11, 293, 58367, 1162, 35989, 519, 2808, 11, 293, 58367, 1162, 727, 535, 929, 198, 30093, 348, 9501, 14940, 929, 4769, 320, 8996, 8044, 340, 58124, 13069, 293, 58367, 1162, 4769, 320, 8996, 8044, 340, 220, 6197, 348, 9501, 14940, 929, 3503, 3706, 535, 929, 284, 293, 58367, 1162, 727, 535, 929, 198, 26513, 348, 9501, 14940, 929, 52114, 284, 569, 7339, 198, 13874, 19324, 16429, 279, 4583, 7214, 1376, 5025, 25351, 821, 16614, 13, 151645, 198]
chosen_labels:
```sql
SELECT vSMCostType.SMCostTypeID, vSMCostType.SMCo, bJCCT.PhaseGroup, bJCCT.CostType
FROM vSMCostType WITH (NOLOCK)
INNER JOIN bJCCT WITH (NOLOCK)
  ON vSMCostType.JCCostType = bJCCT.CostType
WHERE vSMCostType.Co = @Co
```

Using the complete foreign key relationship ensures data integrity.<|im_end|>

rejected_input_ids:
[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 12292, 348, 9501, 14940, 929, 448, 293, 58367, 1162, 151645, 198, 151644, 77091, 198, 73594, 3544, 198, 4858, 1213, 30093, 348, 9501, 14940, 929, 198, 75486, 293, 58367, 1162, 6197, 348, 9501, 14940, 929, 3503, 3706, 535, 929, 284, 293, 58367, 1162, 727, 535, 929, 198, 73594, 151645, 198]
rejected_inputs:
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Join vSMCostType with bJCCT<|im_end|>
<|im_start|>assistant
```sql
SELECT *
FROM vSMCostType
JOIN bJCCT ON vSMCostType.JCCostType = bJCCT.CostType
```<|im_end|>

rejected_label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 73594, 3544, 198, 4858, 1213, 30093, 348, 9501, 14940, 929, 198, 75486, 293, 58367, 1162, 6197, 348, 9501, 14940, 929, 3503, 3706, 535, 929, 284, 293, 58367, 1162, 727, 535, 929, 198, 73594, 151645, 198]
rejected_labels:
```sql
SELECT *
FROM vSMCostType
JOIN bJCCT ON vSMCostType.JCCostType = bJCCT.CostType
```<|im_end|>

[INFO|configuration_utils.py:765] 2025-12-30 19:57:40,687 >> loading configuration file config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\config.json
[INFO|configuration_utils.py:839] 2025-12-30 19:57:40,688 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

[INFO|2025-12-30 19:57:40] llamafactory.model.model_utils.kv_cache:143 >> KV cache is disabled during training.
[WARNING|logging.py:328] 2025-12-30 19:57:40,864 >> `torch_dtype` is deprecated! Use `dtype` instead!
[INFO|modeling_utils.py:1172] 2025-12-30 19:57:40,865 >> loading weights file model.safetensors from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\model.safetensors.index.json
[INFO|modeling_utils.py:1243] 2025-12-30 19:57:40,865 >> Will use dtype=torch.bfloat16 as defined in model's config object
[INFO|modeling_utils.py:2341] 2025-12-30 19:57:40,865 >> Instantiating Qwen2ForCausalLM model under default dtype torch.bfloat16.
[INFO|configuration_utils.py:986] 2025-12-30 19:57:40,866 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "use_cache": false
}

Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.52s/it]
[INFO|configuration_utils.py:941] 2025-12-30 19:57:47,337 >> loading configuration file generation_config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\generation_config.json
[INFO|configuration_utils.py:986] 2025-12-30 19:57:47,337 >> Generate config GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.8
}

[INFO|dynamic_module_utils.py:423] 2025-12-30 19:57:47,627 >> Could not locate the custom_generate/generate.py inside Qwen/Qwen2.5-7B-Instruct.
[INFO|2025-12-30 19:57:47] llamafactory.model.model_utils.checkpointing:143 >> Gradient checkpointing enabled.
[INFO|2025-12-30 19:57:47] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
[INFO|2025-12-30 19:57:47] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.
[INFO|2025-12-30 19:57:47] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-12-30 19:57:56] llamafactory.model.adapter:143 >> Loaded adapter(s): saves/vgpt2_v3/sft
[INFO|2025-12-30 19:57:56] llamafactory.model.loader:143 >> trainable params: 645,922,816 || all params: 8,261,539,328 || trainable%: 7.8184
[WARNING|trainer.py:906] 2025-12-30 19:57:56,146 >> The model is already on multiple devices. Skipping the move to device specified in `args`.
[INFO|trainer.py:749] 2025-12-30 19:57:56,151 >> Using auto half precision backend
[WARNING|trainer.py:982] 2025-12-30 19:57:56,151 >> The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
[INFO|trainer.py:2519] 2025-12-30 19:57:56,487 >> ***** Running training *****
[INFO|trainer.py:2520] 2025-12-30 19:57:56,487 >>   Num examples = 1,355
[INFO|trainer.py:2521] 2025-12-30 19:57:56,487 >>   Num Epochs = 2
[INFO|trainer.py:2522] 2025-12-30 19:57:56,487 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2525] 2025-12-30 19:57:56,487 >>   Total train batch size (w. parallel, distributed & accumulation) = 16
[INFO|trainer.py:2526] 2025-12-30 19:57:56,487 >>   Gradient Accumulation steps = 16
[INFO|trainer.py:2527] 2025-12-30 19:57:56,487 >>   Total optimization steps = 170
[INFO|trainer.py:2528] 2025-12-30 19:57:56,491 >>   Number of trainable parameters = 645,922,816
{'loss': 0.0017, 'grad_norm': 0.0223772581666708, 'learning_rate': 2.647058823529412e-06, 'rewards/chosen': 10.391068458557129, 'rewards/rejected': 1.7056540250778198, 'rewards/accuracies': 1.0, 'rewards/margins': 8.68541431427002, 'logps/chosen': -94.6958236694336, 'logps/rejected': -54.099029541015625, 'logits/chosen': -0.864666759967804, 'logits/rejected': -0.7128506898880005, 'epoch': 0.12}
{'loss': 0.0004, 'grad_norm': 0.0028016450814902782, 'learning_rate': 4.99789221722016e-06, 'rewards/chosen': 13.229782104492188, 'rewards/rejected': 0.7363645434379578, 'rewards/accuracies': 1.0, 'rewards/margins': 12.49341869354248, 'logps/chosen': -65.17558288574219, 'logps/rejected': -65.19561767578125, 'logits/chosen': -0.8794652819633484, 'logits/rejected': -0.7379876971244812, 'epoch': 0.24}
{'loss': 0.0001, 'grad_norm': 0.0003481156309135258, 'learning_rate': 4.9244923400875245e-06, 'rewards/chosen': 14.224032402038574, 'rewards/rejected': -0.0529547818005085, 'rewards/accuracies': 1.0, 'rewards/margins': 14.276987075805664, 'logps/chosen': -51.70466232299805, 'logps/rejected': -73.89064025878906, 'logits/chosen': -0.9100059866905212, 'logits/rejected': -0.7513304352760315, 'epoch': 0.35}
{'loss': 0.0001, 'grad_norm': 0.0002536873798817396, 'learning_rate': 4.749230677304114e-06, 'rewards/chosen': 14.725430488586426, 'rewards/rejected': -0.7080540060997009, 'rewards/accuracies': 1.0, 'rewards/margins': 15.433484077453613, 'logps/chosen': -49.962093353271484, 'logps/rejected': -78.94232177734375, 'logits/chosen': -0.8936958312988281, 'logits/rejected': -0.7190784811973572, 'epoch': 0.47}
{'loss': 0.0001, 'grad_norm': 0.0011127631878480315, 'learning_rate': 4.479470611971646e-06, 'rewards/chosen': 14.9791898727417, 'rewards/rejected': -0.5262429118156433, 'rewards/accuracies': 1.0, 'rewards/margins': 15.50543212890625, 'logps/chosen': -50.314876556396484, 'logps/rejected': -77.67593383789062, 'logits/chosen': -0.9371671676635742, 'logits/rejected': -0.7888380289077759, 'epoch': 0.59}
{'loss': 0.0, 'grad_norm': 0.00017090792243834585, 'learning_rate': 4.126545750510605e-06, 'rewards/chosen': 14.977594375610352, 'rewards/rejected': -0.7204844951629639, 'rewards/accuracies': 1.0, 'rewards/margins': 15.698081016540527, 'logps/chosen': -51.0827522277832, 'logps/rejected': -79.9855728149414, 'logits/chosen': -0.8878900408744812, 'logits/rejected': -0.7268831729888916, 'epoch': 0.71}
{'loss': 0.0, 'grad_norm': 0.00035094560007564723, 'learning_rate': 3.7052837564258728e-06, 'rewards/chosen': 14.955886840820312, 'rewards/rejected': -0.8535234332084656, 'rewards/accuracies': 1.0, 'rewards/margins': 15.809409141540527, 'logps/chosen': -48.56490707397461, 'logps/rejected': -81.39305877685547, 'logits/chosen': -0.9336888194084167, 'logits/rejected': -0.7915822863578796, 'epoch': 0.83}
{'loss': 0.0, 'grad_norm': 0.00012250254803802818, 'learning_rate': 3.2333833859621155e-06, 'rewards/chosen': 14.813385009765625, 'rewards/rejected': -1.3072599172592163, 'rewards/accuracies': 1.0, 'rewards/margins': 16.12064552307129, 'logps/chosen': -47.024295806884766, 'logps/rejected': -86.09481811523438, 'logits/chosen': -0.8863739967346191, 'logits/rejected': -0.7081775069236755, 'epoch': 0.94}
{'loss': 0.0, 'grad_norm': 0.00013337517157196999, 'learning_rate': 2.730670898658255e-06, 'rewards/chosen': 14.645892143249512, 'rewards/rejected': -1.2517166137695312, 'rewards/accuracies': 1.0, 'rewards/margins': 15.897608757019043, 'logps/chosen': -45.567813873291016, 'logps/rejected': -83.41233825683594, 'logits/chosen': -0.8890722393989563, 'logits/rejected': -0.769817054271698, 'epoch': 1.06}
{'loss': 0.0, 'grad_norm': 0.00021175394067540765, 'learning_rate': 2.2182670837116975e-06, 'rewards/chosen': 15.074655532836914, 'rewards/rejected': -1.2557381391525269, 'rewards/accuracies': 1.0, 'rewards/margins': 16.33039665222168, 'logps/chosen': -47.187984466552734, 'logps/rejected': -85.6415786743164, 'logits/chosen': -0.8680412173271179, 'logits/rejected': -0.7159360647201538, 'epoch': 1.18}
{'loss': 0.0, 'grad_norm': 0.0001237803080584854, 'learning_rate': 1.7176998984196148e-06, 'rewards/chosen': 15.241778373718262, 'rewards/rejected': -1.0268176794052124, 'rewards/accuracies': 1.0, 'rewards/margins': 16.268598556518555, 'logps/chosen': -49.51673126220703, 'logps/rejected': -83.54036712646484, 'logits/chosen': -0.8687471747398376, 'logits/rejected': -0.703424334526062, 'epoch': 1.3}
 65%|███████████████████████████████████████████████████████████████▉                       66%|█████████████████████████████████▌        {'loss': 0.0, 'grad_norm': 0.0002501589187886566, 'learning_rate': 1.2500000000000007e-06, 'rewards/chosen': 14.894533157348633, 'rewards/rejected': -1.0893439054489136, 'rewards/accuracies': 1.0, 'rewards/margins': 15.98387622833252, 'logps/chosen': -47.03776168823242, 'logps/rejected': -84.73597717285156, 'logits/chosen': -0.9191311001777649, 'logits/rejected': -0.754830002784729, 'epoch': 1.41}
{'loss': 0.0, 'grad_norm': 0.001410153810866177, 'learning_rate': 8.348171708068748e-07, 'rewards/chosen': 14.410351753234863, 'rewards/rejected': -1.3747704029083252, 'rewards/accuracies': 1.0, 'rewards/margins': 15.785122871398926, 'logps/chosen': -46.717742919921875, 'logps/rejected': -85.95464324951172, 'logits/chosen': -0.9024469256401062, 'logits/rejected': -0.7441760897636414, 'epoch': 1.53}
{'loss': 0.0, 'grad_norm': 0.00016057105676736683, 'learning_rate': 4.895947591461456e-07, 'rewards/chosen': 15.355006217956543, 'rewards/rejected': -1.2433570623397827, 'rewards/accuracies': 1.0, 'rewards/margins': 16.598363876342773, 'logps/chosen': -50.7266731262207, 'logps/rejected': -83.50852966308594, 'logits/chosen': -0.8749485015869141, 'logits/rejected': -0.702215850353241, 'epoch': 1.65}
{'loss': 0.0, 'grad_norm': 0.0013963989913463593, 'learning_rate': 2.2883682045119066e-07, 'rewards/chosen': 15.009808540344238, 'rewards/rejected': -1.170575737953186, 'rewards/accuracies': 1.0, 'rewards/margins': 16.18038558959961, 'logps/chosen': -48.16396713256836, 'logps/rejected': -84.02645874023438, 'logits/chosen': -0.8955982327461243, 'logits/rejected': -0.7358962297439575, 'epoch': 1.77}
{'loss': 0.0, 'grad_norm': 0.0001872618740890175, 'learning_rate': 6.349874889624963e-08, 'rewards/chosen': 14.727740287780762, 'rewards/rejected': -1.1412323713302612, 'rewards/accuracies': 1.0, 'rewards/margins': 15.868972778320312, 'logps/chosen': -46.811859130859375, 'logps/rejected': -82.57125854492188, 'logits/chosen': -0.9327837228775024, 'logits/rejected': -0.7925375699996948, 'epoch': 1.89}
{'loss': 0.0, 'grad_norm': 0.001355273649096489, 'learning_rate': 5.270012410216185e-10, 'rewards/chosen': 15.067658424377441, 'rewards/rejected': -1.1828739643096924, 'rewards/accuracies': 1.0, 'rewards/margins': 16.250532150268555, 'logps/chosen': -47.255558013916016, 'logps/rejected': -83.2221908569336, 'logits/chosen': -0.9007022976875305, 'logits/rejected': -0.749013364315033, 'epoch': 2.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 170/170 [17:45<00:00,  5.91s/it][INFO|trainer.py:4309] 2025-12-30 20:15:41,698 >> Saving model checkpoint to saves/vgpt2_v3/dpo\checkpoint-170
[INFO|configuration_utils.py:765] 2025-12-30 20:15:42,737 >> loading configuration file config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\config.json
[INFO|configuration_utils.py:839] 2025-12-30 20:15:42,737 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

[INFO|tokenization_utils_base.py:2421] 2025-12-30 20:15:44,226 >> chat template saved in saves/vgpt2_v3/dpo\checkpoint-170\chat_template.jinja
[INFO|tokenization_utils_base.py:2590] 2025-12-30 20:15:44,227 >> tokenizer config file saved in saves/vgpt2_v3/dpo\checkpoint-170\tokenizer_config.json
[INFO|tokenization_utils_base.py:2599] 2025-12-30 20:15:44,227 >> Special tokens file saved in saves/vgpt2_v3/dpo\checkpoint-170\special_tokens_map.json
[INFO|trainer.py:2810] 2025-12-30 20:15:47,926 >>

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 1071.4363, 'train_samples_per_second': 2.529, 'train_steps_per_second': 0.159, 'train_loss': 0.00015002657560484402, 'epoch': 2.0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 170/170 [17:51<00:00,  6.30s/it]
[INFO|trainer.py:4309] 2025-12-30 20:15:47,928 >> Saving model checkpoint to saves/vgpt2_v3/dpo
[INFO|configuration_utils.py:765] 2025-12-30 20:15:48,551 >> loading configuration file config.json from cache at C:\Users\olsen\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28\config.json
[INFO|configuration_utils.py:839] 2025-12-30 20:15:48,552 >> Model config Qwen2Config {
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}

[INFO|tokenization_utils_base.py:2421] 2025-12-30 20:15:49,889 >> chat template saved in saves/vgpt2_v3/dpo\chat_template.jinja
[INFO|tokenization_utils_base.py:2590] 2025-12-30 20:15:49,890 >> tokenizer config file saved in saves/vgpt2_v3/dpo\tokenizer_config.json
[INFO|tokenization_utils_base.py:2599] 2025-12-30 20:15:49,890 >> Special tokens file saved in saves/vgpt2_v3/dpo\special_tokens_map.json
***** train metrics *****
  epoch                    =        2.0
  total_flos               = 27146681GF
  train_loss               =     0.0002
  train_runtime            = 0:17:51.43
  train_samples_per_second =      2.529
  train_steps_per_second   =      0.159
Figure saved at: saves/vgpt2_v3/dpo\training_loss.png
Figure saved at: saves/vgpt2_v3/dpo\training_rewards_accuracies.png
[WARNING|2025-12-30 20:15:50] llamafactory.extras.ploting:148 >> No metric eval_loss to plot.
[INFO|trainer.py:4643] 2025-12-30 20:15:50,148 >>
***** Running Evaluation *****
[INFO|trainer.py:4645] 2025-12-30 20:15:50,148 >>   Num examples = 72
[INFO|trainer.py:4648] 2025-12-30 20:15:50,148 >>   Batch size = 1
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:11<00:00,  6.17it/s]
***** eval metrics *****
  epoch                   =        2.0
  eval_logits/chosen      =    -0.8272
  eval_logits/rejected    =    -0.6535
  eval_logps/chosen       =   -48.5414
  eval_logps/rejected     =   -90.8154
  eval_loss               =        0.0
  eval_rewards/accuracies =        1.0
  eval_rewards/chosen     =    14.5332
  eval_rewards/margins    =    16.5783
  eval_rewards/rejected   =    -2.0451
  eval_runtime            = 0:00:11.87
  eval_samples_per_second =      6.064
  eval_steps_per_second   =      6.064
[INFO|modelcard.py:456] 2025-12-30 20:16:02,032 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
(venv) PS C:\Github\LLM_fine-tuning>