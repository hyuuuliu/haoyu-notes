## Yi: Open Foundation Models by 01.AI [[paper](https://arxiv.org/pdf/2403.04652.pdf)]

*01.AI*

*Published on 2024-03-07*

### Highlight

Model Scale Comparison:
| Model Name | Model Scale |
|------------|-------------|
| Yi-34B     | 34B         |
| Chinchilla | 70B         |
| LLaMA      | 70B         |


<mark>Since model scale is relative smaller, Yi uses a larger pretrain dataset (3.1T tokens) so that the **compute flops** can be compensated. (Chinchilla optimal regime, model-data scale balance [[paper](https://arxiv.org/pdf/2203.15556.pdf)])</mark> :paperclip:


Pretraining data cleaning system
| Pipeline      | Related Keywords                                                               |
|---------------|--------------------------------------------------------------------------------|
| Filtering     | language, heuristic textual features, perplexity, semantics, topic, and safety |
| Deduplication | paragraph, MinHash, and exact matching                                         |

<mark> For Youling crowdsourcing platform, shall we consider to make the above methods more frameworked. </mark>

Yi Data Clearning Framework:
<div align="center">
     <img src="assets/20240310_yi_1.png" width="1500" alt="Yi Data Clearning Framework"/>
</div>

A glance on the infra work:
<div align="center">
     <img src="assets/20240310_yi_2.png" width="1500" alt="Yi Data Clearning Framework"/>
</div>

### Pretraining

#### Data Processing

Yi's pretraining data mixture:
<div align="center">
     <img src="assets/20240310_yi_3.png" width="600" alt="Yi Data Clearning Framework"/>
</div>
   
   1. Heuristic Rule Filters
      * URL, domain, word blocklists and garbled text filters.
      * Document length, the ratio of special symbols, and the ratio of short, consecutive, or incomplete lines.
      * Repeated words, n-grams, or paragraphs.
      * Identify and anonymize Personal Identifiable Information (PII), such as email addresses and phone numbers.
  
   2. Learned Filters
      
      <div align="center">
        <img src="assets/20240310_yi_4.png" width="600" alt="Yi Data Clearning Framework"/>
      </div>

      * Perplexity Scorer, evaluates a vast array of web documents, discarding those with perplexity scores
      largely above average. **---based on CCNet** :paperclip:
      * Quality Scorer is a classifier trained to recognize and favor pages similar to Wikipedia in quality and assign scores accordingly. **---based on a trained classifier** :paperclip:
      * Document Coherence Scorer, identifies low-quality web documents that consist of disparate sentences or paragraphs, thus being incoherence. 
      * Safety Scorer identifies and removes web documents containing toxic content, such as violence, pornography, and political propaganda.
   3. Cluster-based Filters :writing_hand:
      <div align="center">
        <img src="assets/20240310_yi_5.png" width="600" alt="Yi Data Clearning Framework"/>
      </div>

   4. Deduplication :paperclip:
      <div align="center">
        <img src="assets/20240310_yi_6.png" width="600" alt="Yi Data Clearning Framework"/>
      </div>

#### Tokenization :paperclip:

   <div align="center">
     <img src="assets/20240310_yi_7.png" width="600" alt="Yi Data Clearning Framework"/>
   </div>

   * numbers into individual digits
   * allow handling on rare characters
   * employ the identity tokenizer
   * In LLMs prioritizing English, utilize dummy prefix in their tokenizers to generalize the same words at different positions. In Yi models, such utilization is dismissed

#### Model Architecture

Architectures for different Yi variations:
| Module                                | Model Size  | Differences               |
|---------------------------------------|-------------|---------------------------|
| Attention Mechanism                   | LLaMA-2-70B | Grouped-Query Attention   |
|                                       | LLaMA-2-7B  | full attention            |
|                                       | LLaMA-2-13B | full attention            |
|                                       | Yi-6B       | Grouped-Query Attention   |
|                                       | Yi-34B      | Grouped-Query Attention   |
| Activation Function                   | Yi-6B       | SwiGLU                    |
|                                       | Yi-34B      | SwiGLU                    |
| Positional Embedding and Long Context | Yi-6B       | Rotary Position Embedding |
|                                       | Yi-34B      | Rotary Position Embedding |

### Finetuning

<mark>An important insight: Finetuning method significantly emphasizes data quality over quantity. </mark>

   <div align="center">
     <img src="assets/20240310_yi_8.png" width="800" alt="Yi Data Clearning Framework"/>
   </div>

#### Data processing
   1. **Quality is all you need**
   
   | Model       | Finetuning dataset scale |
   |-------------|--------------------------|
   | Yi          | less than 10K            |
   | open-source | several hundred thousand |

   * prompt distribution: develope compound instructions and progressively evolved them to increase their complexity. **---based on WizardLM** :paperclip:
   * response formatting: the responses are structured in an introduction-body-conclusion format where the body is usually a list of bullet point. **---based on LIMA** :paperclip:
   * chain-of-thought data formatting: performing abstraction to formulate higher-level solutions before delving into reasoning about the original, more concrete questions. **---based on “Step-Back” pattern in paper Zheng et al.** :paperclip:
   * hallucination: examine and ensure that the knowledge in the responses is not contained within the model, and eliminate responses that might lead to memorization. :question:
   * repetition: **rewrite** the repetitive turns of the responses that usually exist but may be overlooked in the finetuning dats :writing_hand:
   
   2. **Diversity and Mixture**
   * Ensure  the **coverage of different capabilities**: a wide spectrum of open-source prompt, encompassing areas such as *question answering, creative writing, dialogue, reasoning, mathematics, coding, safety, bilingual capabilities, and others.* :writing_hand:
   * Ensure **fine-grained control of different directions of capabilities**: develop a instruction tagging system. **---based on InsTag** :paperclip:
   * Achieve **the optimal data ratio for balancing different directions of the capability**: an approximate grid search to determine our data mixture. **---based on Dong et al.** :paperclip:
   
   3. ChatML Format
   
   * Format of the data substantially influences the model's ultimate performance: need structured data. **---based on ChatML style format**

#### Training Method

   | Parameters          | Value                      |
   |---------------------|----------------------------|
   | Optimizer           | AdamW                      |
   | $\beta_1$           | 0.9                        |
   | $\beta_2$           | 0.999                      |
   | $\epsilon$          | $10^{-8}$                  |
   | Sequence Length     | 4096                       |
   | Batch Size          | 64                         |
   | Training Step       | 300                        |
   | Learning Rate       | $1 \times 10^{-5}$         |
   | Weight Decay        | $0.1$                      |
   | Gradient Clipping   | maximum threshold $1.0$    |
   | NEFTune noise scale | 45 for Yi-34B, 5 for Yi-6B |

### Infrastructure

   <div align="center">
     <img src="assets/20240310_yi_9.png" width="800" alt="Yi Data Clearning Framework"/>
   </div>

   1. Computing Resources Management
   * Manage training tasks of different priorities cross multiple cloud :writing_hand:
   * Elastic GPU utilization strategy :writing_hand:
   * Failure handling :writing_hand:
     * automated inspection, prediction, and labeling of nodes for different kinds of error categories;
     * safely remove failure nodes from resource pool
     * task queuing system with pre-checks
     * a user-friendly console, showing training tasks, hyper-parameters, and so on
  
   2. Performance and Cost Efficiency 
   
      <mark> *Memory* and *communication* restrictions are the two major technical challenges of large scale model training requiring integrated solutions **beyond adding more GPUs**. </mark>
    
   * **ZeRO-1** :paperclip:
   * **tensor parallel** :paperclip:
   * **kernel fusion** :paperclip:
   * topology-aware resource allocation :writing_hand:
   
   3. Finetuning Framework 
   
      <mark>Different from pretraining, finetuning LLMs may require the **orchestration of multiple models**, as is the practice of DPO and PPO.</mark> :writing_hand:

   4. Fast and Efficient Inference

      Primarily quantization, dynamic batching, and Paged Attention for improving decoding speed and memory usage.
      * 4-bit model quantization :paperclip:
      * 8-bit KV cache quantization :paperclip:
      * dynamic batching :paperclip:
      * pagedattention :paperclip:
   
   5. Long-context Window Support

      <div align="center">
        <img src="assets/20240310_yi_10.png" width="800" alt="Yi Data Clearning Framework"/>
      </div>

### Safety

   <div align="center">
     <img src="assets/20240310_yi_11.png" width="800" alt="Yi Data Clearning Framework"/>
   </div>

   1. Safety in Pretraining
   
      heuristic rules, keyword matching, and learned classifiers to remove text containing personal identifiers and private data, and reduce sexual, violent, and extremist content. **---based on existed data safety practices** :paperclip:

   2. Safety in Alignment

   * A comprehensive safety taxonomy :paperclip:
   
### Evaluation

#### Main Results

   1. Commonsense Reasoning :paperclip:
   2. Reading Comprehension :paperclip:
   3. Math :paperclip:
   4. Code :paperclip:
   5. Popular Aggregated Benchmark :paperclip:

   <div align="center">
     <img src="assets/20240310_yi_13.png" width="1800" alt="Yi Data Clearning Framework"/>
   </div>

#### Discussions

   1. Gain from model scale
   
      Obviously, Yi-34B is greatly superior to Yi-6B.

      <mark>Larger model size leads to higher performance gain on Code and Math benchmarks, compared to benchmarks focusing on Commonsense Reasoning, Reading Comprehension, or Knowledge. </mark>

      <div align="center">
        <img src="assets/20240310_yi_12.png" width="1800" alt="Yi Data Clearning Framework"/>
      </div>

   2. Data Quality:

      <div align="center">
        <img src="assets/20240310_yi_14.png" width="600" alt="Yi Data Clearning Framework"/>
      </div>

      <mark>Not quite sure if it's true.</mark>

   3. Gap between GPT-4 and Open-source LLMs

      <mark>A huge gap between GPT-4 and open-source models on reasoning-related benchmarks like BBH, code (HumanEval), and math (MATH).</mark>

### Final Takeaways

   * the full-stack development of the Yi language model family. 
   * Yi-34B achieves GPT-3.5 matching performance and is deployable (thank to the 4/8-bit quantization) on consumer-grade devices
   * Data quantity and quality: 
      1. training the model on a larger amount of data than the Chinchilla optimal delivers clear and consistent performance gain, which we highly recommend for all pretraining teams. 
      2. Our model is trained on 3.1T tokens, yet we belive with larger amount of data, we can continue improve the model performance (i.e., the model have not saturated at 3.1T); 
      3. when it comes to the pretraining data quality, we believe the most critical two factors are the source of the data (e.g., whether the text is produced for professional usage or for casual social media posting) and the details of the data cleaning (e.g., the strength of filtering and deduplication). 
      4. The key takeaways from the Yi finetuning procedure is to heavily iterate on a small amount of data (≤ 10K), case by case, over multiple iterations, directly by the machine learning engineer, and improved from real user feedback. 
   *  The reasoning capability, which we view as the core capability for real-world deployment of language models, is strongly correlated with model scale when the amount of pretraining data is fixed. 6. We believe that given our current results, continuing to scale up model parameters using thoroughly optimized data will lead to even stronger frontier models in our upcoming next versions.