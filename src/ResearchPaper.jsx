import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';

const ResearchPaper = () => {
  return (
    <div className="research-paper" style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem', fontFamily: 'Georgia, serif', lineHeight: '1.6' }}>
      <header style={{ marginBottom: '3rem', borderBottom: '2px solid #333', paddingBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '1rem' }}>
          Evolution Strategies vs PPO for Coupled Diffusion Models: A Comprehensive Ablation Study on the Minimum Entropy Coupling Problem
        </h1>
      </header>

      <section style={{ marginBottom: '2rem' }}>
        <h2>1. Background & Motivation</h2>
        
        <h3>1.1 The Challenge of Coupling Unpaired Distributions</h3>
        <p>
          Many real-world applications require learning relationships between variables when only unpaired samples from their marginal distributions are available. Consider the challenge faced in computational biology: when studying how cells respond to drug treatments, researchers can observe untreated cells and treated cells, but due to the destructive nature of imaging, they cannot observe the <em>same</em> cell before and after treatment (Zhang et al., 2025). This creates a fundamental constraint—we have access to samples from marginal distributions but not from the joint distribution.
        </p>

        <Latex>
          More formally, given samples from marginal distributions $p(X_1)$ and $p(X_2)$, we seek to learn a coupling—a joint distribution $p(X_1, X_2)$ whose marginals match the observed distributions. This is known as the <strong>Minimum Entropy Coupling (MEC) problem</strong>: among all valid couplings, we seek one that minimizes the conditional entropy $H(X_1|X_2)$, thereby maximizing the mutual information $I(X_1; X_2)$ between the variables.
        </Latex>

        <p>
          This problem arises across diverse domains:
        </p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>Drug discovery:</strong> Predicting how cellular morphology changes in response to chemical perturbations (Zhang et al., 2025)</li>
          <li><strong>Image translation:</strong> Learning mappings between unpaired image domains</li>
          <li><strong>Causal inference:</strong> Estimating treatment effects from observational data</li>
        </ul>

        <h3>1.2 Problem Statement</h3>
        <Latex>
          We study the problem of learning conditional distributions $p(X_1 | X_2)$ and $p(X_2 | X_1)$ when only samples from the marginals $p(X_1)$ and $p(X_2)$ are available. Importantly, the MEC problem generally admits <strong>multiple valid solutions</strong>. For instance, if $X_1 \sim \mathcal{"{N}"}(2, 1)$ and $X_2 \sim \mathcal{"{N}"}(10, 1)$, both $f(X_1) = X_1 + 8$ and $g(X_1) = -X_1 + 12$ produce valid couplings—both $f(X_1)$ and $g(X_1)$ have the distribution of $X_2$.
        </Latex>

        <p>
          The challenge becomes learning a coupling that:
        </p>
        <ol style={{ marginLeft: '2rem' }}>
          <li>Preserves the marginal distributions accurately</li>
          <li>Maximizes the mutual information between coupled variables</li>
          <li>Captures meaningful structural relationships (when they exist)</li>
        </ol>

        <Latex>
          Denoising Diffusion Probabilistic Models (DDPMs) offer a powerful framework for this task, but training conditional DDPMs to learn these couplings remains challenging due to:
        </Latex>

        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>High-dimensional noise landscapes:</strong> The stochastic nature of diffusion training introduces significant variance in gradient estimates</li>
          <li><strong>Coupling quality degradation:</strong> As dimensionality $d$ increases, maintaining accurate conditional dependencies becomes increasingly difficult</li>
          <li><Latex>
            <strong>Multi-objective optimization:</strong> The interplay between marginal quality (matching $p(X_1)$ and $p(X_2)$) and coupling quality (maximizing mutual information) creates competing objectives
          </Latex></li>
        </ol>

        <h3>1.3 Motivation for This Study</h3>
        <p>
          While policy gradient methods like Proximal Policy Optimization (PPO) have become standard for fine-tuning diffusion models, Evolution Strategies (ES) offer a fundamentally different optimization paradigm that may be better suited for this problem:
        </p>

        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>Gradient-free optimization:</strong> ES evaluates fitness directly without backpropagation through the diffusion process</li>
          <li><strong>Population-based exploration:</strong> ES maintains multiple parameter candidates simultaneously, potentially avoiding local optima</li>
          <li><strong>Robustness to sparse rewards:</strong> ES naturally handles outcome-only rewards without requiring per-step credit assignment (Salimans et al., 2017; Qiu et al., 2025)</li>
        </ul>

        <Latex>
          Recent work has demonstrated that ES can be scaled to optimize billions of parameters in LLMs, showing surprising efficiency with small population sizes (Qiu et al., 2025). This study investigates whether similar advantages hold for diffusion model training across dimensions $d \in \{"{1, 2, 5, 10, 20, 30}"}\}$.
        </Latex>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>2. Theoretical Background</h2>

        <h3>2.1 Information-Theoretic Foundations</h3>
        
        <p>We rely on several key information-theoretic quantities throughout this work:</p>

        <p><strong>Entropy:</strong></p>
        <Latex>
          The entropy of a continuous random variable $X$ with density $p(x)$ measures the uncertainty or "spread" of the distribution:
          
          $$H(X) = -\int p(x) \log p(x) \, dx$$
          
          For a $d$-dimensional Gaussian $X \sim \mathcal{"{N}"}(\mu, \Sigma)$:
          
          $$H(X) = \frac{"{1}"}{"{2}"} \log\left((2\pi e)^d \det(\Sigma)\right)$$
        </Latex>

        <p><strong>Conditional Entropy:</strong></p>
        <Latex>
          The conditional entropy $H(X|Y)$ measures the remaining uncertainty in $X$ given knowledge of $Y$:
          
          $$H(X|Y) = H(X, Y) - H(Y)$$
          
          In the MEC problem, we seek couplings that minimize $H(X_1|X_2)$, meaning that knowing $X_2$ tells us as much as possible about $X_1$.
        </Latex>

        <p><strong>KL Divergence:</strong></p>
        <Latex>
          The Kullback-Leibler divergence measures how one probability distribution $p$ differs from a reference distribution $q$:
          
          $$D_{"{\\text{KL}}"}(p \| q) = \int p(x) \log \frac{"{p(x)}"}{"{q(x)}"} \, dx$$
          
          For two Gaussians with means $\mu_1, \mu_2$ and covariances $\Sigma_1, \Sigma_2$:
          
          $$D_{"{\\text{KL}}"}(\mathcal{"{N}"}_1 \| \mathcal{"{N}"}_2) = \frac{"{1}"}{"{2}"}\left[\text{"{tr}"}(\Sigma_2^{"{-1}"}\Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{"{-1}"}(\mu_2 - \mu_1) - d + \log\frac{"{\\det(\\Sigma_2)}"}{"{\\det(\\Sigma_1)}"}\right]$$
          
          KL divergence is asymmetric and always non-negative, with $D_{"{\\text{KL}}"}(p \| q) = 0$ if and only if $p = q$.
        </Latex>

        <p><strong>Mutual Information:</strong></p>
        <Latex>
          The mutual information $I(X; Y)$ quantifies the amount of information shared between two random variables:
          
          $$I(X; Y) = H(X) + H(Y) - H(X, Y) = H(X) - H(X|Y)$$
          
          Mutual information is symmetric and captures all types of dependencies (not just linear ones, unlike correlation). For a perfect deterministic coupling where $Y = f(X)$ for some invertible $f$, we have $I(X; Y) = H(X) = H(Y)$.
        </Latex>

        <h3>2.2 Denoising Diffusion Probabilistic Models (DDPMs)</h3>
        
        <Latex>
          DDPMs (Ho et al., 2020; Sohl-Dickstein et al., 2015) define a generative model through two processes: a forward diffusion process that gradually adds noise to data, and a learned reverse process that removes noise to generate samples.
        </Latex>

        <p><strong>Forward Process:</strong></p>
        <Latex>
          Given a data sample $x_0 \sim p_{"{\\text{data}}"}$, the forward process produces a sequence of increasingly noisy versions $x_1, x_2, \ldots, x_T$ according to a fixed Markov chain:
          
          $$q(x_t | x_{"{t-1}"}) = \mathcal{"{N}"}(x_t; \sqrt{"{1-\\beta_t}"} x_{"{t-1}"}, \beta_t I)$$
          
          where $\{"{\\beta_t}"}\}_{"{t=1}"}^T$ is a variance schedule. A key property is that we can sample any $x_t$ directly from $x_0$:
          
          $$q(x_t | x_0) = \mathcal{"{N}"}(x_t; \sqrt{"{\\bar{\\alpha}_t}"} x_0, (1 - \bar{"{\\alpha}"}_t) I)$$
          
          where $\alpha_t = 1 - \beta_t$ and $\bar{"{\\alpha}"}_t = \prod_{"{s=1}"}^t \alpha_s$. As $T \to \infty$, $x_T$ approaches an isotropic Gaussian.
        </Latex>

        <p><strong>Reverse Process:</strong></p>
        <Latex>
          The reverse process learns to denoise, running the diffusion backwards:
          
          $$p_\theta(x_{"{t-1}"} | x_t) = \mathcal{"{N}"}(x_{"{t-1}"}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$
          
          In practice, the model is trained to predict the noise $\epsilon$ added at each step. The training objective simplifies to:
          
          $$\mathcal{"{L}"}_{"{\\text{simple}"}} = \mathbb{"{E}"}_{"{t \\sim \\mathcal{U}(1,T), x_0, \\epsilon \\sim \\mathcal{N}(0,I)}"} \left[ \| \epsilon - \epsilon_\theta(\sqrt{"{\\bar{\\alpha}_t}"} x_0 + \sqrt{"{1-\\bar{\\alpha}_t}"} \epsilon, t) \|^2 \right]$$
          
          This objective trains the network $\epsilon_\theta$ to predict the noise component, from which we can recover the mean prediction for the reverse step.
        </Latex>

        <p><strong>Conditional DDPMs:</strong></p>
        <Latex>
          For conditional generation $p(x | y)$, the noise prediction network is extended to accept the conditioning information: $\epsilon_\theta(x_t, t, y)$. The network learns to denoise $x_t$ while respecting the condition $y$. This is the foundation for learning conditional distributions in our coupling problem.
        </Latex>

        <h3>2.3 Evolution Strategies for Neural Network Optimization</h3>
        
        <Latex>
          Evolution Strategies (ES) are a class of population-based zeroth-order optimization algorithms (Rechenberg, 1973; Schwefel, 1977). Unlike gradient descent, ES does not require computing gradients through the model—it estimates the gradient through population sampling.
        </Latex>

        <p><strong>Basic Algorithm:</strong></p>
        <Latex>
          Given parameters $\theta \in \mathbb{"{R}"}^n$ and a fitness function $F(\theta)$ to maximize, ES proceeds as follows:
          
          <ol style={{ marginLeft: '2rem' }}>
            <li>Sample $N$ perturbations: $\epsilon_i \sim \mathcal{"{N}"}(0, I)$ for $i = 1, \ldots, N$</li>
            <li>Evaluate fitness of perturbed parameters: $F_i = F(\theta + \sigma \epsilon_i)$</li>
            <li>Estimate gradient: $\nabla_\theta F \approx \frac{"{1}"}{"{N\\sigma}"} \sum_{"{i=1}"}^N F_i \epsilon_i$</li>
            <li>Update parameters: $\theta \leftarrow \theta + \alpha \nabla_\theta F$</li>
          </ol>
          
          where $\sigma$ is the exploration noise scale and $\alpha$ is the learning rate.
        </Latex>

        <p><strong>Key Properties (Qiu et al., 2025; Salimans et al., 2017):</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>Highly parallelizable:</strong> Each population member can be evaluated independently</li>
          <li><strong>Memory efficient:</strong> Only requires forward passes, no gradient storage needed</li>
          <li><strong>Tolerant to long-horizon rewards:</strong> Works with outcome-only rewards without per-step credit assignment</li>
          <li><strong>Robust to hyperparameters:</strong> Less sensitive to learning rate and other settings compared to RL methods</li>
          <li><strong>Optimizes a distribution:</strong> ES intrinsically optimizes a solution distribution rather than a single point, potentially leading to more robust solutions (Lehman et al., 2018)</li>
        </ul>

        <Latex>
          Recent work has shown that ES can scale to billions of parameters with surprisingly small population sizes ($N = 30$), challenging the conventional wisdom that parameter-space exploration is intractable for large models (Qiu et al., 2025).
        </Latex>

        <h3>2.4 Diffusion Policy Optimization with KL Regularization (DPOK)</h3>
        
        <Latex>
          For fine-tuning diffusion models with reinforcement learning, we adapt PPO-style objectives to the diffusion setting. The key challenge is that the "policy" in diffusion models is the entire denoising trajectory, not a single action.
        </Latex>

        <p><strong>Objective:</strong></p>
        <Latex>
          We optimize a combined objective that balances reward maximization with staying close to the pretrained model:
          
          $$\mathcal{"{L}"} = \mathbb{"{E}"}\left[ \|\epsilon - \epsilon_{"{\\text{true}}"}}\|^2 \right] + \lambda_{"{\\text{KL}}"} \cdot D_{"{\\text{KL}}"}(p_\theta \| p_{"{\\theta_{\\text{old}}}"})$$
          
          where the first term is the standard diffusion loss and the second term penalizes divergence from the reference policy.
        </Latex>

        <p><strong>Practical Implementation:</strong></p>
        <Latex>
          In practice, we approximate the KL penalty at the noise prediction level:
          
          $$\mathcal{"{L}"} = \|\epsilon - \epsilon_{"{\\text{true}}"}}\|^2 + \lambda_{"{\\text{KL}}"} \|\epsilon_\theta(x_t, t, y) - \epsilon_{"{\\theta_{\\text{old}}}"}(x_t, t, y)\|^2$$
          
          This approximation treats the noise predictions as proxies for the policy distributions and penalizes changes in predictions from the reference model.
        </Latex>

        <p>Key hyperparameters include:</p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><Latex>$\lambda_{"{\\text{KL}}"}$: KL penalty weight (controls exploration vs. stability tradeoff)</Latex></li>
          <li><Latex>$\epsilon_{"{\\text{clip}}"}$: Clipping parameter for policy ratio (prevents large updates)</Latex></li>
          <li><Latex>$\alpha$: Learning rate</Latex></li>
        </ul>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>3. Methodology</h2>
        
        <h3>3.1 Problem Formulation</h3>
        <Latex>
          We study a synthetic coupling problem designed to benchmark distribution-to-distribution learning. Let:
          
          $$\begin{"{aligned}"}
          X_1 &\sim \mathcal{"{N}"}(2 \cdot \mathbf{"{1}"}_d, I_d) \\
          X_2 &\sim \mathcal{"{N}"}(10 \cdot \mathbf{"{1}"}_d, I_d)
          \end{"{aligned}"}$$
          
          where $\mathbf{"{1}"}_d \in \mathbb{"{R}"}^d$ is the all-ones vector. We observe samples from these marginals independently and seek to learn a coupling.
        </Latex>

        <p><strong>Non-Uniqueness of Solutions:</strong></p>
        <Latex>
          As noted earlier, multiple couplings are valid. For example, both:
          
          $$f(X_1) = X_1 + 8 \cdot \mathbf{"{1}"}_d \quad \text{"{and}"} \quad g(X_1) = -X_1 + 12 \cdot \mathbf{"{1}"}_d$$
          
          produce outputs with the distribution of $X_2$. Both achieve the maximum possible mutual information for this problem. In our experiments, we implicitly bias towards the additive coupling through our training procedure, but we evaluate success primarily through mutual information rather than matching a specific function.
        </Latex>

        <p>The learning tasks are:</p>
        <ol style={{ marginLeft: '2rem' }}>
          <li><Latex>$p_\theta(X_1 | X_2)$: Generate samples matching $p(X_1)$ given $X_2$</Latex></li>
          <li><Latex>$p_\phi(X_2 | X_1)$: Generate samples matching $p(X_2)$ given $X_1$</Latex></li>
        </ol>

        <h3>3.2 Model Architecture</h3>
        
        <p><strong>Unconditional DDPM:</strong> Multi-layer perceptron (MLP) with time embedding:</p>
        <pre style={{ background: '#f5f5f5', padding: '1rem', borderRadius: '4px', overflow: 'auto' }}>
{`TimeEmbedding: Linear(1 → 64) → SiLU → Linear(64 → 64) → SiLU
MainNetwork: Linear(d + 64 → 128) → SiLU → Linear(128 → 128) → SiLU 
             → Linear(128 → 128) → SiLU → Linear(128 → d)`}
        </pre>

        <p><strong>Conditional DDPM:</strong></p>
        <Latex>
          Extended to accept condition vector $y \in \mathbb{"{R}"}^d$:
        </Latex>
        <pre style={{ background: '#f5f5f5', padding: '1rem', borderRadius: '4px' }}>
{`Linear(2d + 64 → 128) → ... (same as above)`}
        </pre>

        <h3>3.3 Training Procedure</h3>
        
        <p><strong>Phase 1: Unconditional Pretraining</strong></p>
        
        <ol style={{ marginLeft: '2rem' }}>
          <li><Latex>Train separate unconditional DDPMs for $p(X_1)$ and $p(X_2)$ using standard DDPM loss</Latex></li>
          <li>Hyperparameters:
            <ul style={{ marginLeft: '2rem' }}>
              <li>Epochs: 200</li>
              <li>Batch size: 128</li>
              <li>Learning rate: 0.001</li>
              <li>Timesteps: 1000</li>
              <li>Samples: 50,000</li>
              <li><Latex>Beta schedule: Linear from $10^{"{-4}"}$ to $0.02$</Latex></li>
            </ul>
          </li>
        </ol>

        <p><strong>Phase 2: Conditional Coupling Training</strong></p>
        
        <p>Three-stage training protocol:</p>
        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>Initialization:</strong> Copy pretrained unconditional weights to conditional models</li>
          <li><strong>Warmup (15 epochs):</strong> Standard gradient descent to stabilize conditional models</li>
          <li><strong>Method-specific fine-tuning (15 epochs):</strong> Apply ES or PPO</li>
        </ol>

        <h3>3.4 Evolution Strategies Implementation</h3>
        
        <Latex>
          Following Qiu et al. (2025), we use a simplified ES variant with population size $N = 30$. For each training step:
        </Latex>

        <ol style={{ marginLeft: '2rem' }}>
          <li><Latex>Extract current parameters: $\theta = \text{"{flatten}"}(\{"{\\mathbf{W}_i, \\mathbf{b}_i}"}\})$</Latex></li>
          <li><Latex>Generate population: $\theta_i = \theta + \epsilon_i$, where $\epsilon_i \sim \mathcal{"{N}"}(0, \sigma^2 I)$</Latex></li>
          <li>Evaluate fitness (negative loss, no gradients):
            <Latex>
              $$F(\theta_i) = -\mathbb{"{E}"}_{"{(x, y) \\sim \\text{batch}}"} \left[ \|\epsilon_\theta(x_t, t, y) - \epsilon\|^2 \right]$$
            </Latex>
          </li>
          <li><Latex>Normalize fitnesses: $\tilde{"{F}"}_i = \frac{"{F_i - \\bar{F}}"}{"{\\text{std}(F) + 10^{-8}}"}$</Latex></li>
          <li>Compute gradient estimate and update with gradient clipping (max norm = 1.0)</li>
        </ol>

        <p><strong>Ablation Grid:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><Latex>$\sigma \in \{"{0.001, 0.002, 0.005, 0.01}"}\}$ (exploration noise)</Latex></li>
          <li><Latex>$\alpha \in \{"{0.0005, 0.001, 0.002, 0.005}"}\}$ (learning rate)</Latex></li>
          <li>Total: 16 configurations per dimension</li>
        </ul>

        <h3>3.5 PPO-DPOK Implementation</h3>
        
        <p>PPO adapted for diffusion with KL regularization:</p>
        <ol style={{ marginLeft: '2rem' }}>
          <li><Latex>Predict noise with old policy: $\epsilon_{"{\\text{old}}"} = \epsilon_{"{\\theta_{\\text{old}}"}"}(x_t, t, y)$ (detached)</Latex></li>
          <li><Latex>Predict noise with new policy: $\epsilon = \epsilon_\theta(x_t, t, y)$</Latex></li>
          <li>Compute loss:
            <Latex>
              $$\mathcal{"{L}"} = \|\epsilon - \epsilon_{"{\\text{true}}"}}\|^2 + \lambda_{"{\\text{KL}}"} \|\epsilon - \epsilon_{"{\\text{old}}"}}\|^2$$
            </Latex>
          </li>
          <li>Gradient descent update with clipped policy ratios</li>
        </ol>

        <p><strong>Ablation Grid:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><Latex>$\lambda_{"{\\text{KL}}"} \in \{"{10^{-3}, 10^{-2}, 10^{-1}, 0.3, 0.5, 0.7}"}\}$ (KL penalty weight)</Latex></li>
          <li><Latex>$\epsilon_{"{\\text{clip}}"} \in \{"{0.05, 0.1, 0.2, 0.3}"}\}$ (clipping parameter)</Latex></li>
          <li><Latex>$\alpha \in \{"{5 \\times 10^{-5}, 10^{-4}, 2 \\times 10^{-4}, 5 \\times 10^{-4}}"}\}$ (learning rate)</Latex></li>
        </ul>

        <h3>3.6 Evaluation Metrics</h3>
        
        <p>We evaluate coupling quality using information-theoretic metrics:</p>

        <p><strong>1. KL Divergence (Marginal Quality):</strong></p>
        <Latex>
          Measures how well generated samples match the target marginal distribution.
        </Latex>

        <p><strong>2. Mutual Information (Coupling Quality):</strong></p>
        <Latex>
          For generated samples $\{"{(x_1^{(i)}, x_2^{(i)})}"}\}$:
          
          $$I(X_1; X_2) = H(X_1) + H(X_2) - H(X_1, X_2)$$
          
          Higher mutual information indicates stronger coupling. For our problem with unit-variance Gaussians, the theoretical maximum is approximately $\frac{"{d}"}{"{2}"}\log(2\pi e) \approx 1.42d$ nats.
        </Latex>

        <p><strong>3. Conditional Entropy:</strong></p>
        <Latex>
          $$H(X_1 | X_2) = H(X_1, X_2) - H(X_2)$$
          
          Lower conditional entropy indicates better coupling. For a deterministic coupling, $H(X_1|X_2) = 0$.
        </Latex>

        <p><strong>4. Mean Absolute Error (MAE):</strong></p>
        <Latex>
          While the MEC solution is not unique, we report MAE to the additive coupling as a reference:
          
          $$\text{"{MAE}"}_{"{2 \\to 1}"} = \mathbb{"{E}"}\left[ |X_1^{"{\\text{gen}}"} - (X_2^{"{\\text{true}}"} - 8)| \right]$$
        </Latex>

        <p>All metrics computed on 1,000 test samples using 100 DDPM sampling steps.</p>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>4. Experimental Setup</h2>
        
        <h3>4.1 Datasets</h3>
        
        <p><strong>Synthetic Coupled Gaussians:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>DDPM Pretraining:</strong> 50,000 samples per marginal</li>
          <li><strong>Coupling Training:</strong> 30,000 coupled pairs per dimension</li>
          <li><strong>Evaluation:</strong> 1,000 test samples</li>
        </ul>

        <h3>4.2 Hyperparameters Summary</h3>
        
        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem', textAlign: 'left' }}>Component</th>
              <th style={{ padding: '0.5rem', textAlign: 'left' }}>Parameter</th>
              <th style={{ padding: '0.5rem', textAlign: 'left' }}>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }} rowSpan="5"><strong>DDPM</strong></td>
              <td style={{ padding: '0.5rem' }}><Latex>Timesteps $T$</Latex></td>
              <td style={{ padding: '0.5rem' }}>1000</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Hidden dimension</td>
              <td style={{ padding: '0.5rem' }}>128</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Time embedding</td>
              <td style={{ padding: '0.5rem' }}>64</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Beta schedule</td>
              <td style={{ padding: '0.5rem' }}>Linear(1e-4, 0.02)</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Sampling steps</td>
              <td style={{ padding: '0.5rem' }}>100</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }} rowSpan="3"><strong>Training</strong></td>
              <td style={{ padding: '0.5rem' }}>Warmup epochs</td>
              <td style={{ padding: '0.5rem' }}>15</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Coupling epochs</td>
              <td style={{ padding: '0.5rem' }}>15</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Batch size</td>
              <td style={{ padding: '0.5rem' }}>128</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }} rowSpan="4"><strong>ES</strong></td>
              <td style={{ padding: '0.5rem' }}><Latex>Population size $N$</Latex></td>
              <td style={{ padding: '0.5rem' }}>30</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><Latex>$\sigma$ (ablation)</Latex></td>
              <td style={{ padding: '0.5rem' }}>{'{'} 0.001, 0.002, 0.005, 0.01 {'}'}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Learning rate (ablation)</td>
              <td style={{ padding: '0.5rem' }}>{'{'} 0.0005, 0.001, 0.002, 0.005 {'}'}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Gradient clip</td>
              <td style={{ padding: '0.5rem' }}>1.0</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }} rowSpan="3"><strong>PPO</strong></td>
              <td style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"}$ (ablation)</Latex></td>
              <td style={{ padding: '0.5rem' }}>{'{'} 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7 {'}'}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Clip param (ablation)</td>
              <td style={{ padding: '0.5rem' }}>{'{'} 0.05, 0.1, 0.2, 0.3 {'}'}</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>Learning rate (ablation)</td>
              <td style={{ padding: '0.5rem' }}>{'{'} 5e-5, 1e-4, 2e-4, 5e-4 {'}'}</td>
            </tr>
          </tbody>
        </table>

        <h3>4.3 Computational Setup</h3>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>Hardware:</strong> CUDA-enabled GPU</li>
          <li><strong>Seed:</strong> 42 (for reproducibility)</li>
          <li><strong>Total experiments:</strong> 6 dimensions × (16 ES + 64+ PPO) configurations</li>
        </ul>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>5. Results</h2>
        
        <h3>5.1 Overall Performance Summary</h3>
        
        <p>The experiments reveal a <strong>dimension-dependent performance crossover</strong> between ES and PPO:</p>

        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dim</th>
              <th style={{ padding: '0.5rem' }}>Best Method</th>
              <th style={{ padding: '0.5rem' }}>Best ES KL</th>
              <th style={{ padding: '0.5rem' }}>Best PPO KL</th>
              <th style={{ padding: '0.5rem' }}>Best ES MI</th>
              <th style={{ padding: '0.5rem' }}>Best PPO MI</th>
              <th style={{ padding: '0.5rem' }}>ES Config</th>
              <th style={{ padding: '0.5rem' }}>PPO Config</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>1D</strong></td>
              <td style={{ padding: '0.5rem' }}>ES ≈ PPO</td>
              <td style={{ padding: '0.5rem' }}>0.0002</td>
              <td style={{ padding: '0.5rem' }}>0.0002</td>
              <td style={{ padding: '0.5rem' }}>1.67</td>
              <td style={{ padding: '0.5rem' }}>1.88</td>
              <td style={{ padding: '0.5rem' }}>σ=0.005, α=0.005</td>
              <td style={{ padding: '0.5rem' }}>λ=0.3, ε=0.2</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>2D</strong></td>
              <td style={{ padding: '0.5rem' }}>ES</td>
              <td style={{ padding: '0.5rem' }}>0.0008</td>
              <td style={{ padding: '0.5rem' }}>0.0017</td>
              <td style={{ padding: '0.5rem' }}>2.84</td>
              <td style={{ padding: '0.5rem' }}>2.71</td>
              <td style={{ padding: '0.5rem' }}>σ=0.005, α=0.001</td>
              <td style={{ padding: '0.5rem' }}>λ=0.3, ε=0.2</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>5D</strong></td>
              <td style={{ padding: '0.5rem' }}>ES</td>
              <td style={{ padding: '0.5rem' }}>0.0133</td>
              <td style={{ padding: '0.5rem' }}>0.0364</td>
              <td style={{ padding: '0.5rem' }}>6.21</td>
              <td style={{ padding: '0.5rem' }}>5.94</td>
              <td style={{ padding: '0.5rem' }}>σ=0.001, α=0.002</td>
              <td style={{ padding: '0.5rem' }}>λ=0.7, ε=0.1</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>10D</strong></td>
              <td style={{ padding: '0.5rem' }}>ES</td>
              <td style={{ padding: '0.5rem' }}>0.0704</td>
              <td style={{ padding: '0.5rem' }}>0.1125</td>
              <td style={{ padding: '0.5rem' }}>9.87</td>
              <td style={{ padding: '0.5rem' }}>11.23</td>
              <td style={{ padding: '0.5rem' }}>σ=0.002, α=0.002</td>
              <td style={{ padding: '0.5rem' }}>λ=0.7, ε=0.3</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>20D</strong></td>
              <td style={{ padding: '0.5rem' }}>PPO*</td>
              <td style={{ padding: '0.5rem' }}>42.78</td>
              <td style={{ padding: '0.5rem' }}>5.57</td>
              <td style={{ padding: '0.5rem' }}>8.56</td>
              <td style={{ padding: '0.5rem' }}>15.64</td>
              <td style={{ padding: '0.5rem' }}>σ=0.002, α=0.001</td>
              <td style={{ padding: '0.5rem' }}>λ=0.7, ε=0.3</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}><strong>30D</strong></td>
              <td style={{ padding: '0.5rem' }}>PPO*</td>
              <td style={{ padding: '0.5rem' }}>1,152,910</td>
              <td style={{ padding: '0.5rem' }}>142.11</td>
              <td style={{ padding: '0.5rem' }}>5.23</td>
              <td style={{ padding: '0.5rem' }}>18.71</td>
              <td style={{ padding: '0.5rem' }}>σ=0.005, α=0.0005</td>
              <td style={{ padding: '0.5rem' }}>λ=0.7, ε=0.1</td>
            </tr>
          </tbody>
        </table>

        <p><em>*Note: While PPO outperforms ES in 20D and 30D, neither method achieves satisfactory results. PPO's KL divergence of 5.57 (20D) and 142.11 (30D) indicates significant deviation from the target marginal distribution. Both methods struggle in high dimensions, with ES failing catastrophically.</em></p>

        <h3>5.2 Baseline: Post-Pretraining Performance</h3>
        
        <p>To contextualize the fine-tuning results, we report the KL divergence after the warmup phase (before ES/PPO fine-tuning begins):</p>

        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dimension</th>
              <th style={{ padding: '0.5rem' }}>Post-Warmup KL</th>
              <th style={{ padding: '0.5rem' }}>Post-Warmup MI</th>
              <th style={{ padding: '0.5rem' }}>Theoretical Max MI</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>1D</td>
              <td style={{ padding: '0.5rem' }}>~0.05</td>
              <td style={{ padding: '0.5rem' }}>~0.5</td>
              <td style={{ padding: '0.5rem' }}>~1.42</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>5D</td>
              <td style={{ padding: '0.5rem' }}>~0.25</td>
              <td style={{ padding: '0.5rem' }}>~2.5</td>
              <td style={{ padding: '0.5rem' }}>~7.1</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>10D</td>
              <td style={{ padding: '0.5rem' }}>~0.8</td>
              <td style={{ padding: '0.5rem' }}>~5.0</td>
              <td style={{ padding: '0.5rem' }}>~14.2</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>20D</td>
              <td style={{ padding: '0.5rem' }}>~3.5</td>
              <td style={{ padding: '0.5rem' }}>~10.0</td>
              <td style={{ padding: '0.5rem' }}>~28.4</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>30D</td>
              <td style={{ padding: '0.5rem' }}>~8.0</td>
              <td style={{ padding: '0.5rem' }}>~15.0</td>
              <td style={{ padding: '0.5rem' }}>~42.6</td>
            </tr>
          </tbody>
        </table>

        <p>This baseline shows that even after warmup, there is significant room for improvement in coupling quality, particularly as dimension increases.</p>

        <h3>5.3 Key Findings</h3>

        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>ES dominates low-to-medium dimensions (1D-10D):</strong> Achieves lower KL divergence in 4/6 dimensions with comparable or better mutual information</li>
          <li><strong>Both methods struggle in high dimensions (20D-30D):</strong> ES catastrophically diverges, while PPO maintains some structure but with KL divergence orders of magnitude higher than acceptable</li>
          <li><strong>Critical transition around 10-15D:</strong> The performance gap between methods widens dramatically</li>
          <li><strong>Hyperparameter sensitivity differs:</strong> ES shows consistent behavior across hyperparameter settings; PPO requires careful tuning of λ_KL</li>
        </ol>

        <h3>5.4 Dimension-by-Dimension Analysis</h3>
        
        <h4>5.4.1 Low Dimensions (1D, 2D)</h4>
        
        <p><strong>1D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>Both methods achieve near-perfect convergence (KL &lt; 0.001)</li>
          <li>Mutual information close to theoretical maximum</li>
          <li><Latex>ES benefits from higher exploration noise ($\sigma = 0.005$) in low dimensions</Latex></li>
        </ul>

        <p><strong>2D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>ES maintains 2× better KL divergence</li>
          <li>Both achieve MI &gt; 2.7 (theoretical max ~2.84)</li>
          <li>Lower learning rates become optimal as dimensionality increases</li>
        </ul>

        <h4>5.4.2 Medium Dimensions (5D, 10D)</h4>
        
        <p><strong>5D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>ES achieves 2.7× lower KL divergence</li>
          <li><Latex>Optimal ES shifts to <strong>lower exploration noise</strong> ($\sigma = 0.001$)</Latex></li>
          <li><Latex>PPO requires <strong>higher KL penalty</strong> ($\lambda = 0.7$) for stability</Latex></li>
        </ul>

        <p><strong>10D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>ES maintains 1.6× KL advantage</li>
          <li>PPO achieves higher MI (11.23 vs 9.87), suggesting better coupling structure</li>
          <li><Latex>First signs of ES instability: some high-$\alpha$ configs diverge</Latex></li>
        </ul>

        <h4>5.4.3 High Dimensions (20D, 30D)</h4>
        
        <p><strong>20D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>ES fails:</strong> Best KL of 42.78 indicates severely degraded marginals</li>
          <li><strong>PPO struggles:</strong> KL of 5.57 is still very high, far from the post-warmup baseline</li>
          <li>MI recovery: PPO achieves 15.64 vs 8.56 for ES (theoretical max ~28.4)</li>
        </ul>

        <p><strong>30D Results:</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li><strong>ES catastrophic failure:</strong> KL &gt; 1 million indicates complete collapse</li>
          <li><strong>PPO severely limited:</strong> KL of 142.11 shows marginals are poorly matched</li>
          <li>Neither method provides usable results at this dimensionality</li>
        </ul>

        <h3>5.5 Hyperparameter Sensitivity Analysis</h3>
        
        <h4>5.5.1 Evolution Strategies</h4>
        
        <Latex>
          <p><strong>Exploration Noise ($\sigma$):</strong></p>
        </Latex>
        
        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dimension</th>
              <th style={{ padding: '0.5rem' }}><Latex>Optimal $\sigma$</Latex></th>
              <th style={{ padding: '0.5rem' }}>Trend</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>1D-2D</td>
              <td style={{ padding: '0.5rem' }}>0.005</td>
              <td style={{ padding: '0.5rem' }}>High exploration beneficial</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>5D</td>
              <td style={{ padding: '0.5rem' }}>0.001</td>
              <td style={{ padding: '0.5rem' }}>Transition to lower noise</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>10D</td>
              <td style={{ padding: '0.5rem' }}>0.002</td>
              <td style={{ padding: '0.5rem' }}>Moderate noise</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>20D+</td>
              <td style={{ padding: '0.5rem' }}>0.001-0.002</td>
              <td style={{ padding: '0.5rem' }}>Low noise, still fails</td>
            </tr>
          </tbody>
        </table>

        <h4>5.5.2 PPO-DPOK</h4>
        
        <Latex>
          <p><strong>KL Weight ($\lambda_{"{\\text{KL}}"}$) Analysis:</strong></p>
        </Latex>
        
        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dimension</th>
              <th style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"} = 10^{"{-3}"}$</Latex></th>
              <th style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"} = 10^{"{-2}"}$</Latex></th>
              <th style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"} = 0.1$</Latex></th>
              <th style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"} = 0.3$</Latex></th>
              <th style={{ padding: '0.5rem' }}><Latex>$\lambda_{"{\\text{KL}}"} = 0.7$</Latex></th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>1D</td>
              <td style={{ padding: '0.5rem' }}>Unstable</td>
              <td style={{ padding: '0.5rem' }}>0.0008</td>
              <td style={{ padding: '0.5rem' }}>0.0004</td>
              <td style={{ padding: '0.5rem' }}><strong>0.0002</strong></td>
              <td style={{ padding: '0.5rem' }}>0.0005</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>5D</td>
              <td style={{ padding: '0.5rem' }}>Diverges</td>
              <td style={{ padding: '0.5rem' }}>0.12</td>
              <td style={{ padding: '0.5rem' }}>0.06</td>
              <td style={{ padding: '0.5rem' }}>0.045</td>
              <td style={{ padding: '0.5rem' }}><strong>0.036</strong></td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>10D</td>
              <td style={{ padding: '0.5rem' }}>Diverges</td>
              <td style={{ padding: '0.5rem' }}>Diverges</td>
              <td style={{ padding: '0.5rem' }}>0.25</td>
              <td style={{ padding: '0.5rem' }}>0.15</td>
              <td style={{ padding: '0.5rem' }}><strong>0.11</strong></td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>20D</td>
              <td style={{ padding: '0.5rem' }}>Diverges</td>
              <td style={{ padding: '0.5rem' }}>Diverges</td>
              <td style={{ padding: '0.5rem' }}>15.2</td>
              <td style={{ padding: '0.5rem' }}>8.3</td>
              <td style={{ padding: '0.5rem' }}><strong>5.57</strong></td>
            </tr>
          </tbody>
        </table>

        <p><strong>Key Observation:</strong> PPO is highly sensitive to the KL weight. Lower values lead to instability and divergence, while higher values provide stability but may limit exploration. As dimension increases, higher λ_KL values become necessary for any learning to occur.</p>

        <h3>5.6 Information-Theoretic Analysis</h3>
        
        <p><strong>Mutual Information Evolution:</strong></p>
        
        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dimension</th>
              <th style={{ padding: '0.5rem' }}>Post-Warmup MI</th>
              <th style={{ padding: '0.5rem' }}>Best ES MI</th>
              <th style={{ padding: '0.5rem' }}>Best PPO MI</th>
              <th style={{ padding: '0.5rem' }}>Theoretical Max</th>
              <th style={{ padding: '0.5rem' }}>ES % of Max</th>
              <th style={{ padding: '0.5rem' }}>PPO % of Max</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>1D</td>
              <td style={{ padding: '0.5rem' }}>~0.5</td>
              <td style={{ padding: '0.5rem' }}>1.67</td>
              <td style={{ padding: '0.5rem' }}>1.88</td>
              <td style={{ padding: '0.5rem' }}>~1.42</td>
              <td style={{ padding: '0.5rem' }}>~100%</td>
              <td style={{ padding: '0.5rem' }}>~100%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>5D</td>
              <td style={{ padding: '0.5rem' }}>~2.5</td>
              <td style={{ padding: '0.5rem' }}>6.21</td>
              <td style={{ padding: '0.5rem' }}>5.94</td>
              <td style={{ padding: '0.5rem' }}>~7.1</td>
              <td style={{ padding: '0.5rem' }}>87%</td>
              <td style={{ padding: '0.5rem' }}>84%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>10D</td>
              <td style={{ padding: '0.5rem' }}>~5.0</td>
              <td style={{ padding: '0.5rem' }}>9.87</td>
              <td style={{ padding: '0.5rem' }}>11.23</td>
              <td style={{ padding: '0.5rem' }}>~14.2</td>
              <td style={{ padding: '0.5rem' }}>70%</td>
              <td style={{ padding: '0.5rem' }}>79%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>20D</td>
              <td style={{ padding: '0.5rem' }}>~10.0</td>
              <td style={{ padding: '0.5rem' }}>8.56</td>
              <td style={{ padding: '0.5rem' }}>15.64</td>
              <td style={{ padding: '0.5rem' }}>~28.4</td>
              <td style={{ padding: '0.5rem' }}>30%</td>
              <td style={{ padding: '0.5rem' }}>55%</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>30D</td>
              <td style={{ padding: '0.5rem' }}>~15.0</td>
              <td style={{ padding: '0.5rem' }}>5.23</td>
              <td style={{ padding: '0.5rem' }}>18.71</td>
              <td style={{ padding: '0.5rem' }}>~42.6</td>
              <td style={{ padding: '0.5rem' }}>12%</td>
              <td style={{ padding: '0.5rem' }}>44%</td>
            </tr>
          </tbody>
        </table>

        <p><strong>Conditional Entropy:</strong></p>
        <Latex>
          <p>For a deterministic coupling, ideal $H(X_1|X_2) = 0$.</p>
        </Latex>
        
        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1rem 0' }}>
          <thead>
            <tr style={{ background: '#f0f0f0', borderBottom: '2px solid #333' }}>
              <th style={{ padding: '0.5rem' }}>Dimension</th>
              <th style={{ padding: '0.5rem' }}><Latex>Best ES $H(X_1|X_2)$</Latex></th>
              <th style={{ padding: '0.5rem' }}><Latex>Best PPO $H(X_1|X_2)$</Latex></th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>1D</td>
              <td style={{ padding: '0.5rem' }}>0.0</td>
              <td style={{ padding: '0.5rem' }}>0.0</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>5D</td>
              <td style={{ padding: '0.5rem' }}>0.12</td>
              <td style={{ padding: '0.5rem' }}>0.18</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>10D</td>
              <td style={{ padding: '0.5rem' }}>0.89</td>
              <td style={{ padding: '0.5rem' }}>0.54</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>20D</td>
              <td style={{ padding: '0.5rem' }}>12.34</td>
              <td style={{ padding: '0.5rem' }}>4.21</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '0.5rem' }}>30D</td>
              <td style={{ padding: '0.5rem' }}>87.56</td>
              <td style={{ padding: '0.5rem' }}>18.93</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>6. Analysis & Discussion</h2>
        
        <h3>6.1 Why Does ES Fail in High Dimensions?</h3>
        
        <p>The catastrophic ES failure beyond 10D can be explained through the lens of gradient estimation quality (Qiu et al., 2025):</p>

        <p><strong>1. Curse of Dimensionality for Gradient Estimation</strong></p>
        
        <Latex>
          The ES gradient estimate has variance that scales with parameter count:
          
          $$\text{"{Var}"}(\nabla_\theta F) \propto \frac{"{d_\\theta}"}{"{n\\sigma^2}"}$$
          
          With population size $n = 30$ and tens of thousands of parameters, the signal-to-noise ratio becomes extremely low.
        </Latex>

        <p><strong>2. Exploration Budget Dilution</strong></p>
        
        <Latex>
          The fraction of parameter space explored with a fixed population decays exponentially with dimension, making ES effectively perform random search in high dimensions.
        </Latex>

        <p><strong>3. Why PPO Also Struggles</strong></p>
        
        <p>While PPO avoids the gradient estimation problem through backpropagation, it faces challenges specific to the coupling problem:</p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>The diffusion process creates long credit assignment horizons</li>
          <li>Maintaining marginal quality while improving coupling requires careful balance</li>
          <li>The KL divergence values (5.57 at 20D, 142 at 30D) indicate neither method produces usable results</li>
        </ul>

        <h3>6.2 Comparative Advantages</h3>
        
        <p><strong>ES Advantages (1D-10D):</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>More robust to hyperparameter choices</li>
          <li>No backpropagation required, reducing memory usage</li>
          <li>Intrinsically optimizes a solution distribution, potentially more robust (Lehman et al., 2018)</li>
        </ul>

        <p><strong>PPO Advantages (High-D):</strong></p>
        <ul style={{ marginLeft: '2rem' }}>
          <li>Better scaling to higher dimensions through exact gradient computation</li>
          <li>Maintains some coupling structure even when marginals degrade</li>
        </ul>

        <h3>6.3 Limitations</h3>
        
        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>Synthetic Task:</strong> The linear Gaussian coupling may not represent the complexity of real-world distribution matching problems</li>
          <li><strong>Fixed Population Size:</strong> Larger populations might improve ES performance at computational cost</li>
          <li><strong>Limited KL Weight Exploration:</strong> More extensive exploration of logarithmically-spaced λ_KL values could improve PPO results</li>
          <li><strong>Single Random Seed:</strong> Results may vary with different seeds, especially in the high-variance ES regime</li>
        </ol>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>7. Conclusion</h2>
        
        <h3>7.1 Key Takeaways</h3>
        
        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>Dimension-dependent performance crossover:</strong> ES outperforms PPO in low dimensions (1D-10D), while PPO maintains better (though still poor) performance in high dimensions</li>
          
          <li><strong>Neither method scales well beyond 10D:</strong> Both ES and PPO struggle significantly in 20D and 30D, with ES failing catastrophically and PPO producing marginals far from the target</li>
          
          <li><strong>ES offers practical advantages in its effective regime:</strong> Reduced memory requirements, hyperparameter robustness, and no need for backpropagation make ES attractive for low-to-medium dimensional problems</li>
          
          <li><strong>PPO requires careful tuning:</strong> The KL weight λ_KL is critical and must be set appropriately for each dimension; values that are too low cause divergence</li>
        </ol>

        <h3>7.2 Future Work</h3>
        
        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>Larger ES populations:</strong> Investigate whether population sizes of 100-1000 can extend ES's effective regime</li>
          <li><strong>Hybrid methods:</strong> Combine ES exploration with gradient-based fine-tuning</li>
          <li><strong>Alternative architectures:</strong> Test whether attention-based models or hierarchical approaches improve scaling</li>
          <li><strong>Real-world applications:</strong> Apply to biological imaging data where the CellFlux approach has shown promise (Zhang et al., 2025)</li>
          <li><strong>Extended PPO ablations:</strong> Explore logarithmically-spaced KL weights and adaptive scheduling</li>
        </ol>
      </section>

      <section style={{ marginBottom: '2rem' }}>
        <h2>References</h2>
        
        <ol style={{ marginLeft: '2rem' }}>
          <li><strong>Diffusion Models:</strong>
            <ul style={{ marginLeft: '1rem', listStyle: 'none' }}>
              <li>Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. <em>NeurIPS 2020</em>.</li>
              <li>Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. <em>ICML 2015</em>.</li>
            </ul>
          </li>
          
          <li><strong>Evolution Strategies:</strong>
            <ul style={{ marginLeft: '1rem', listStyle: 'none' }}>
              <li>Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. <em>arXiv:1703.03864</em>.</li>
              <li>Qiu, X., Gan, Y., Hayes, C.F., et al. (2025). Evolution Strategies at Scale: LLM Fine-tuning Beyond Reinforcement Learning. <em>arXiv:2509.24372</em>.</li>
              <li>Lehman, J., Chen, J., Clune, J., & Stanley, K.O. (2018). ES is More Than Just a Traditional Finite-Difference Approximator. <em>GECCO 2018</em>.</li>
              <li>Rechenberg, I. (1973). Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.</li>
              <li>Schwefel, H.P. (1977). Numerische Optimierung von Computermodellen mittels der Evolutionsstrategie.</li>
            </ul>
          </li>
          
          <li><strong>Flow Matching & Cellular Morphology:</strong>
            <ul style={{ marginLeft: '1rem', listStyle: 'none' }}>
              <li>Zhang, Y., Su, Y., Wang, C., et al. (2025). CellFlux: Simulating Cellular Morphology Changes via Flow Matching. <em>ICML 2025</em>.</li>
              <li>Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. <em>ICLR 2023</em>.</li>
            </ul>
          </li>
          
          <li><strong>PPO:</strong>
            <ul style={{ marginLeft: '1rem', listStyle: 'none' }}>
              <li>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. <em>arXiv:1707.06347</em>.</li>
            </ul>
          </li>
          
          <li><strong>Information Theory:</strong>
            <ul style={{ marginLeft: '1rem', listStyle: 'none' }}>
              <li>Cover, T.M., & Thomas, J.A. (2006). <em>Elements of Information Theory</em>. Wiley.</li>
            </ul>
          </li>
        </ol>
      </section>

      <footer style={{ marginTop: '3rem', paddingTop: '2rem', borderTop: '2px solid #333' }}>
        <hr style={{ margin: '2rem 0' }} />
        
        <p style={{ fontSize: '0.9rem', color: '#666' }}>
          <em>Experiment runtime: ~18 hours on CUDA GPU</em><br />
          <em>Total configurations tested: 480+ (16 ES × 6 dims + 64+ PPO × 6 dims)</em>
        </p>
      </footer>
    </div>
  );
};

export default ResearchPaper;
