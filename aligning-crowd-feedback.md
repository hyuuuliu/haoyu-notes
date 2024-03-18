## Aligning Crowd Feedback via Distributional Preference Reward Modeling [[paper](https://arxiv.org/pdf/2402.09764.pdf)]

*Dexun Li, Cong Zhang, Kuicai Dong, Derrick Goh Xin Deik, Ruiming Tang, Yong Liu*

Huawei Noah’s Ark Lab

### Problem
  1. Limited representation
  2. Evolving preferences
   <div style="text-align:center">
    <img src="assets/20240310_1.png" width="400" alt="problem"/>
   </div>

### Method
 1. human preference distrbution, beta distribution
 
    <img src="assets/20240310_2.png" width="1200" alt="DPRM framework"/>

    A,B,... are generated from one comercial LLM with various simulated personas.`

    <div style="text-align:center">
    <img src="assets/20240310_3.png" width="400" alt="ratings"/>
    </div>

    * Prior Distribution: $LLM_{API}$ directly outputs a distribution that captures a wide range of human perspectives.
    * Posterior Distribution: $LLM_{API}$ is instructed to emulate different human personas like "rigorous scientists", "impulsive teens", "eccentric artists", etc, and articulate their perfrences by selecting one of the six categories.
    * Label Smoothing: on posterior distriubtion, tempers absolute certainty by adjusting them towards near certainty with a marginal probability allocated to the next most likely label.

1. a novel reward model framework, Distributional Preference RM (DPRM)

   * Previously, use binary ranking label (chosen & rejected) for RM.
   * A direct idea is to use cross-entropy loss on the distributions:
   <div align="center">
   <img src="assets/20240310_4.png" width="400" alt="DPRM_1"/>
   </div>

   * However, such an idea results in equal distance between $[0.9, 0.1, 0, 0, 0, 0]$ with $[0.9, 0, 0.1, 0, 0, 0]$ or $[0.9, 0, 0, 0, 0, 0.1]$.
   * Need a ranking loss! Use optimal transport loss, which recognizes **geometry of label space**.
  
### Experiments
   1. Baselines
   <div align="center">
   <img src="assets/20240310_6.png" width="400" alt="baselines"/>
   </div>

   2. Comparison results:
   <div align="center">
     <img src="assets/20240310_5.png" width="1200" alt="DPRM_1"/>
   </div>