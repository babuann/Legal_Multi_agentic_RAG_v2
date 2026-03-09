---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1502
- loss:DenoisingAutoEncoderLoss
- dataset_size:4500
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: As respondents explain, the an has no ongoing for does materially
    Ninth hold that 10b-5(b) liability be such
  sentences:
  - As respondents explain, the nondisclosure of an event that has no  ongoing or
    future  implications for the business does not render a risk statement materially
    misleading, and the Ninth Circuit did not hold that Rule 10b-5(b) liability can
    be premised on such an omission.
  - In re Burlington Coat Factory Securities Litigation, 114 F.3d 1410, 1432 (3d Cir.
  - 2011); see Restatement (Second) of Contracts   159 cmt.
- source_sentence: a certiorari filed on granted as to the first question
  sentences:
  - Aug. 16, 1979) .......... 19 SEC v. Tecumseh Holdings Corp., 765 F. Supp.
  - When used in this fashion, the words  could  and  may  are forward-looking in
    nature; they indicate the possibility that a future occurrence of the identified
    event will cause the risk to materialize into a real-world consequence.
  - The petition for a writ of certiorari was filed on March 4, 2024, and granted
    as to the first question presented on June 10, 2024.
- source_sentence: 229.105 ..................................................................
  sentences:
  - "229.105 .................................................................. 2a\
    \ \n\n (1a) 1."
  - Denny v. Barber, 576 F.2d 465, 470 (2d Cir.
  - denied, 544 U.S. 920 (2005)) (brackets omitted), cert.
- source_sentence: court of decision premised on understanding of Analytica misconduct
    was not filed 2016 10-K.
  sentences:
  - The strong financial incentive for public companies to settle securities class
    actions, even for substantial amounts, underscores the need to weed out meritless
    claims as early as possible.
  - Respondents also challenged separate statements in the  Risk Factors  section
    that Meta s business might suffer  [i]f people do not perceive [Meta s] products
    to be useful, reliable, and trustworthy.
  - And the court of appeals  decision was premised on that court s understanding
    that  the extent of Cambridge Analytica s misconduct was not yet public when Facebook
    filed its 2016 Form 10-K.  Pet.
- source_sentence: approach be traced common under which statement purposes of misrepresentation
    determined to effect under the the ordinary
  sentences:
  - That approach can be traced to the common law, under which the misleading nature
    of a statement for purposes of the tort of fraudulent misrepresentation was  determined
    according to the effect [it] would produce, under the circumstances, upon the
    ordinary mind.
  - But that is precisely what can make such a statement misleading.
  - The syllabus constitutes no part of the opinion of the Court but has been prepared
    by the Reporter of Decisions for the convenience of the reader.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'approach be traced common under which statement purposes of misrepresentation determined to effect under the the ordinary',
    'That approach can be traced to the common law, under which the misleading nature of a statement for purposes of the tort of fraudulent misrepresentation was  determined according to the effect [it] would produce, under the circumstances, upon the ordinary mind.',
    'The syllabus constitutes no part of the opinion of the Court but has been prepared by the Reporter of Decisions for the convenience of the reader.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9008, 0.6693],
#         [0.9008, 1.0000, 0.6739],
#         [0.6693, 0.6739, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,500 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 10 tokens</li><li>mean: 39.1 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 39.04 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                        | positive                                                                                                                                                                                                                                                                                                                                  |
  |:------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>23-980 In the Supreme Court of the United States FACEBOOK, INC., ET AL., PETITIONERS v. AMALGAMATED BANK, ET AL.</code> | <code>ON WRIT OF CERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT BRIEF FOR THE PETITIONERS JOSHUA S. LIPSHUTZ KATHERINE MORAN MEEKS TRENTON VAN OSS GIBSON, DUNN & CRUTCHER LLP 1050 Connecticut Avenue, N.W.</code>                                                                                               |
  | <code>23-980 In the Supreme Court of the United States FACEBOOK, INC., ET AL., PETITIONERS v. AMALGAMATED BANK, ET AL.</code> | <code>Washington, DC 20036 BRIAN M. LUTZ MICHAEL J. KAHN PATRICK J. FUSTER GIBSON, DUNN & CRUTCHER LLP One Embarcadero Center, Suite 2600 San Francisco, CA 94111 KANNON K. SHANMUGAM Counsel of Record WILLIAM T. MARKS MATTEO GODI JAKE L. KRAMER ANNA P. LIPIN PAUL, WEISS, RIFKIND, WHARTON & GARRISON LLP 2001 K Street, N.W.</code> |
  | <code>23-980 In the Supreme Court of the United States FACEBOOK, INC., ET AL., PETITIONERS v. AMALGAMATED BANK, ET AL.</code> | <code>(II) PARTIES TO THE PROCEEDING AND CORPORATE DISCLOSURE STATEMENT Petitioners are Facebook, Inc., now known as Meta Platforms, Inc.; Mark Zuckerberg; Sheryl Sandberg; and David M. Wehner.</code>                                                                                                                                  |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `learning_rate`: 2e-05
- `num_train_epochs`: 1
- `warmup_steps`: 100

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 100
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2660 | 50   | 9.9803        |
| 0.5319 | 100  | 7.0367        |
| 0.7979 | 150  | 5.8125        |
| 0.3546 | 100  | 2.4481        |
| 0.7092 | 200  | 1.9393        |


### Framework Versions
- Python: 3.9.6
- Sentence Transformers: 5.1.2
- Transformers: 4.57.6
- PyTorch: 2.8.0
- Accelerate: 1.10.1
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->