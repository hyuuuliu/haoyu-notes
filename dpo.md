<!-- % Title
% Author
% Date

# Slide 1 title

Some super quickly created demo slides

* Do not need anything else than markdown
    * Slides title starts with # (also starts a new slide)
    * Bullet points, newlines, empty lines: all standard markdown
* However, can also use other stuff, e.g.:
    * Some HTML (e.g. \<center\>)
    * When using pandoc beamer, can use latex commands (e.g. \\center, \\large, etc)\dots

# Slide 2 title

\center The slide syntax is so simple that you can quickly create a handful of slides on basically any device in any editor. E.g. on your mobile on the way to the meeting where you need the slides. Right before the meeting starts you use pandoc to create the actual slides from your source. -->

Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)]

*Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn*

Stanford University, Chan Zuckerberg Biohub
<!-- % Date -->

<!-- ## Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)]

*Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn*

Stanford University, Chan Zuckerberg Biohub -->

# Problem

   1. RLHF is a **complex** and often **unstable** procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model.

   ![From RLHF to DPO](assets/20240310_2_1.png)

# Method

   * Deriving the DPO objective

      RL Fine-Tuning Phase:

      $max_{\pi_\theta} \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_{\theta}(y|x)}[r_\phi(x, y) - \beta\mathbb{D}_{\text{KL}[\pi_\theta(y|x)||\pi_{\text{ref}}(y|x)]}]$

      The optimal solution of KL-constrained reward maximization objective: :paperclip:

      ![](assets/20240310_2_2.png){width=70%}

      $Z(x)$ is a partition function, which is hard to estimate.

      This makes it hard to utilize in practice.
      Take the logarithm of both sides, we have: 

      ![](assets/20240310_DPO_3.png){width=70%}

# Method

* Deriving the DPO objective
   
  Recall Bradley-Terry model:

  ![](assets/20240310_DPO_4.png)

  Substituting the reparameterized $r(x,y)$ into BT model, the optimal RLHF policy $\pi^*$ satisfy the preference model: 

  ![](assets/20240310_DPO_5.png)

# Method

* Deriving the DPO objective

  Now we have the probability of human preference data interms of the optimal policy rather than the reward model. To solve it, formualting a maximum likelihood objective for $\pi_\theta$:

  ![](assets/20240310_DPO_6.png)

# Method

* Deriving the DPO objective

  The gradient with respect to $\theta$:
      
  ![](assets/20240310_DPO_7.png)
      
  where ![](assets/20240310_DPO_8.png){width=30%}, 
      
  is the reward implicitly defined by the language model $\pi_\theta$ and reference model $\pi_{\text{ref}}$.

# Method
* Utilization
      1. Sample $y_1, y_2 \sim \pi_{\text{ref}}(.| x)$ for every prompt $x$, label them with human preferences.
      2. Optimize $\pi_{\theta}$ to minimize $\mathcal{L}_{\text{DPO}}$.