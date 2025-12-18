import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';
import './ResearchPaper.css';

const ResearchPaper = () => {
  return (
    <div className="research-paper">
      <header className="paper-header">
        <h1>
          Evolution Strategies vs PPO for Coupled Diffusion Models: A Comprehensive Ablation Study on the Minimum Entropy Coupling Problem
        </h1>
      </header>

      <section className="paper-section">
        <h2>1. Background & Motivation</h2>
        
        <h3>1.1 The Challenge of Coupling Unpaired Distributions</h3>
        <p>
          Many real-world applications require learning relationships between variables when only unpaired samples from their marginal distributions are available. Consider the challenge faced in computational biology: when studying how cells respond to drug treatments, researchers can observe untreated cells and treated cells, but due to the destructive nature of imaging, they cannot observe the <em>same</em> cell before and after treatment (Zhang et al., 2025). This creates a fundamental constraint—we have access to samples from marginal distributions but not from the joint distribution.
        </p>

        <Latex>
          {`More formally, given samples from marginal distributions $p(X_1)$ and $p(X_2)$, we seek to learn a coupling—a joint distribution $p(X_1, X_2)$ whose marginals match the observed distributions. This is known as the **Minimum Entropy Coupling (MEC) problem**: among all valid couplings, we seek one that minimizes the conditional entropy $H(X_1|X_2)$, thereby maximizing the mutual information $I(X_1; X_2)$ between the variables.`}
        </Latex>

        <p>This problem arises across diverse domains:</p>
        <ul>
          <li><strong>Drug discovery:</strong> Predicting how cellular morphology changes in response to chemical perturbations (Zhang et al., 2025)</li>
          <li><strong>Image translation:</strong> Learning mappings between unpaired image domains</li>
          <li><strong>Causal inference:</strong> Estimating treatment effects from observational data</li>
        </ul>

        <h3>1.2 Problem Statement</h3>
        <Latex>
          {`We study the problem of learning conditional distributions $p(X_1 | X_2)$ and $p(X_2 | X_1)$ when only samples from the marginals $p(X_1)$ and $p(X_2)$ are available. Importantly, the MEC problem generally admits **multiple valid solutions**. For instance, if $X_1 \\sim \\mathcal{N}(2, 1)$ and $X_2 \\sim \\mathcal{N}(10, 1)$, both $f(X_1) = X_1 + 8$ and $g(X_1) = -X_1 + 12$ produce valid couplings—both $f(X_1)$ and $g(X_1)$ have the distribution of $X_2$.`}
        </Latex>

        <p>The challenge becomes learning a coupling that:</p>
        <ol>
          <li>Preserves the marginal distributions accurately</li>
          <li>Maximizes the mutual information between coupled variables</li>
          <li>Captures meaningful structural relationships (when they exist)</li>
        </ol>

        <Latex>
          {`Denoising Diffusion Probabilistic Models (DDPMs) offer a powerful framework for this task, but training conditional DDPMs to learn these couplings remains challenging due to:`}
        </Latex>

        <ol>
          <li><strong>High-dimensional noise landscapes:</strong> The stochastic nature of diffusion training introduces significant variance in gradient estimates</li>
          <li><Latex>{`**Coupling quality degradation:** As dimensionality $d$ increases, maintaining accurate conditional dependencies becomes increasingly difficult`}</Latex></li>
          <li><Latex>{`**Multi-objective optimization:** The interplay between marginal quality (matching $p(X_1)$ and $p(X_2)$) and coupling quality (maximizing mutual information) creates competing objectives`}</Latex></li>
        </ol>

        <h3>1.3 Motivation for This Study</h3>
        <p>
          While policy gradient methods like Proximal Policy Optimization (PPO) have become standard for fine-tuning diffusion models, Evolution Strategies (ES) offer a fundamentally different optimization paradigm that may be better suited for this problem:
        </p>

        <ul>
          <li><strong>Gradient-free optimization:</strong> ES evaluates fitness directly without backpropagation through the diffusion process</li>
          <li><strong>Population-based exploration:</strong> ES maintains multiple parameter candidates simultaneously, potentially avoiding local optima</li>
          <li><strong>Robustness to sparse rewards:</strong> ES naturally handles outcome-only rewards without requiring per-step credit assignment (Salimans et al., 2017; Qiu et al., 2025)</li>
        </ul>

        <Latex>
          {`Recent work has demonstrated that ES can be scaled to optimize billions of parameters in LLMs, showing surprising efficiency with small population sizes (Qiu et al., 2025). This study investigates whether similar advantages hold for diffusion model training across dimensions $d \\in \\{1, 2, 5, 10, 20, 30\\}$.`}
        </Latex>
      </section>

      <section className="paper-section">
        <h2>2. Theoretical Background</h2>

        <h3>2.1 Information-Theoretic Foundations</h3>
        
        <p>We rely on several key information-theoretic quantities throughout this work:</p>

        <p><strong>Entropy:</strong></p>
        <Latex>
          {`The entropy of a continuous random variable $X$ with density $p(x)$ measures the uncertainty or "spread" of the distribution:

$$H(X) = -\\int p(x) \\log p(x) \\, dx$$

For a $d$-dimensional Gaussian $X \\sim \\mathcal{N}(\\mu, \\Sigma)$:

$$H(X) = \\frac{1}{2} \\log\\left((2\\pi e)^d \\det(\\Sigma)\\right)$$`}
        </Latex>

        <p><strong>Conditional Entropy:</strong></p>
        <Latex>
          {`The conditional entropy $H(X|Y)$ measures the remaining uncertainty in $X$ given knowledge of $Y$:

$$H(X|Y) = H(X, Y) - H(Y)$$

In the MEC problem, we seek couplings that minimize $H(X_1|X_2)$, meaning that knowing $X_2$ tells us as much as possible about $X_1$.`}
        </Latex>

        <p><strong>KL Divergence:</strong></p>
        <Latex>
          {`The Kullback-Leibler divergence measures how one probability distribution $p$ differs from a reference distribution $q$:

$$D_{\\text{KL}}(p \\| q) = \\int p(x) \\log \\frac{p(x)}{q(x)} \\, dx$$

KL divergence is asymmetric and always non-negative, with $D_{\\text{KL}}(p \\| q) = 0$ if and only if $p = q$. For two Gaussians $p = \\mathcal{N}(\\mu_1, \\Sigma_1)$ and $q = \\mathcal{N}(\\mu_2, \\Sigma_2)$:

$$D_{\\text{KL}}(p \\| q) = \\frac{1}{2}\\left[\\text{tr}(\\Sigma_2^{-1}\\Sigma_1) + (\\mu_2 - \\mu_1)^T\\Sigma_2^{-1}(\\mu_2 - \\mu_1) - d + \\log\\frac{\\det\\Sigma_2}{\\det\\Sigma_1}\\right]$$`}
        </Latex>

        <p><strong>Mutual Information:</strong></p>
        <Latex>
          {`The mutual information $I(X; Y)$ quantifies the amount of information shared between two random variables:

$$I(X; Y) = H(X) + H(Y) - H(X, Y) = H(X) - H(X|Y)$$

Mutual information is symmetric and captures **all types of dependencies** (not just linear ones, unlike correlation). For a perfect deterministic coupling where $Y = f(X)$ for some invertible $f$, we have $I(X; Y) = H(X) = H(Y)$.`}
        </Latex>

        <h3>2.2 Denoising Diffusion Probabilistic Models (DDPMs)</h3>
        
        <Latex>
          {`DDPMs (Ho et al., 2020; Sohl-Dickstein et al., 2015) define a generative model through two processes: a forward diffusion process that gradually adds noise to data, and a learned reverse process that removes noise to generate samples.`}
        </Latex>

        <p><strong>Forward Process:</strong></p>
        <Latex>
          {`Given a data sample $x_0 \\sim p_{\\text{data}}$, the forward process produces a sequence of increasingly noisy versions $x_1, x_2, \\ldots, x_T$ according to a fixed Markov chain:

$$q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)$$

where $\\{\\beta_t\\}_{t=1}^T$ is a variance schedule. A key property is that we can sample any $x_t$ directly from $x_0$:

$$q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) I)$$

where $\\alpha_t = 1 - \\beta_t$ and $\\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$. As $T \\to \\infty$, $x_T$ approaches an isotropic Gaussian.`}
        </Latex>

        <p><strong>Reverse Process:</strong></p>
        <Latex>
          {`The reverse process learns to denoise, running the diffusion backwards:

$$p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))$$

In practice, the model is trained to predict the noise $\\epsilon$ added at each step. The training objective simplifies to:

$$\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t \\sim \\mathcal{U}(1,T), x_0, \\epsilon \\sim \\mathcal{N}(0,I)} \\left[ \\| \\epsilon - \\epsilon_\\theta(\\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon, t) \\|^2 \\right]$$`}
        </Latex>

        <p><strong>Conditional DDPMs:</strong></p>
        <Latex>
          {`For conditional generation $p(x | y)$, the noise prediction network is extended to accept the conditioning information: $\\epsilon_\\theta(x_t, t, y)$. The network learns to denoise $x_t$ while respecting the condition $y$. This is the foundation for learning conditional distributions in our coupling problem.`}
        </Latex>

        <h3>2.3 Evolution Strategies for Neural Network Optimization</h3>
        
        <Latex>
          {`Evolution Strategies (ES) are a class of population-based zeroth-order optimization algorithms (Rechenberg, 1973; Schwefel, 1977). Unlike gradient descent, ES does not require computing gradients through the model—it estimates the gradient through population sampling.`}
        </Latex>

        <p><strong>Basic Algorithm:</strong></p>
        <Latex>
          {`Given parameters $\\theta \\in \\mathbb{R}^n$ and a fitness function $F(\\theta)$ to maximize, ES proceeds as follows:

1. Sample $N$ perturbations: $\\epsilon_i \\sim \\mathcal{N}(0, I)$ for $i = 1, \\ldots, N$
2. Evaluate fitness of perturbed parameters: $F_i = F(\\theta + \\sigma \\epsilon_i)$
3. Estimate gradient: $\\nabla_\\theta F \\approx \\frac{1}{N\\sigma} \\sum_{i=1}^N F_i \\epsilon_i$
4. Update parameters: $\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta F$

where $\\sigma$ is the exploration noise scale and $\\alpha$ is the learning rate.`}
        </Latex>

        <p><strong>Variance of Gradient Estimates:</strong></p>
        <Latex>
          {`The variance of the ES gradient estimator scales as:

$$\\text{Var}(\\nabla_\\theta F) \\propto \\frac{d_\\theta}{N\\sigma^2}$$

where $d_\\theta$ is the parameter dimension. This scaling becomes problematic in high dimensions unless population size $N$ is increased accordingly.`}
        </Latex>

        <p><strong>Key Properties (Qiu et al., 2025; Salimans et al., 2017):</strong></p>
        <ul>
          <li><strong>Highly parallelizable:</strong> Each population member can be evaluated independently</li>
          <li><strong>Memory efficient:</strong> Only requires forward passes, no gradient storage needed</li>
          <li><strong>Tolerant to long-horizon rewards:</strong> Works with outcome-only rewards without per-step credit assignment</li>
          <li><strong>Robust to hyperparameters:</strong> Less sensitive to learning rate and other settings compared to RL methods</li>
          <li><strong>Optimizes a distribution:</strong> ES intrinsically optimizes a solution distribution rather than a single point, potentially leading to more robust solutions (Lehman et al., 2018)</li>
        </ul>

        <p>Recent findings by Qiu et al. (2025) demonstrate that ES can scale to billions of parameters with surprisingly small population sizes (N=30), challenging conventional wisdom about ES scalability.</p>

        <h3>2.4 Diffusion Policy Optimization with KL Regularization (DPOK)</h3>
        
        <Latex>
          {`For fine-tuning diffusion models with reinforcement learning, we adapt PPO-style objectives to the diffusion setting. The key challenge is that the "policy" in diffusion models is the entire denoising trajectory, not a single action.`}
        </Latex>

        <p><strong>Objective:</strong></p>
        <Latex>
          {`We optimize a combined objective that balances reward maximization with staying close to the pretrained model:

$$\\mathcal{L} = \\mathbb{E}\\left[ \\|\\epsilon - \\epsilon_{\\text{true}}\\|^2 \\right] + \\lambda_{\\text{KL}} \\cdot D_{\\text{KL}}(p_\\theta \\| p_{\\theta_{\\text{old}}})$$

where the first term is the standard diffusion loss and the second term penalizes divergence from the reference policy.`}
        </Latex>

        <p><strong>Practical Implementation:</strong></p>
        <Latex>
          {`In practice, we approximate the KL penalty at the noise prediction level:

$$\\mathcal{L} = \\|\\epsilon - \\epsilon_{\\text{true}}\\|^2 + \\lambda_{\\text{KL}} \\|\\epsilon_\\theta(x_t, t, y) - \\epsilon_{\\theta_{\\text{old}}}(x_t, t, y)\\|^2$$

The PPO clipping mechanism is also applied:

$$\\mathcal{L}^{\\text{clip}} = \\min\\left( r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon_{\\text{clip}}, 1+\\epsilon_{\\text{clip}}) \\hat{A}_t \\right)$$`}
        </Latex>

        <p>Key hyperparameters include:</p>
        <ul>
          <li><Latex>{`$\\lambda_{\\text{KL}}$: KL penalty weight (controls exploration vs. stability tradeoff)`}</Latex></li>
          <li><Latex>{`$\\epsilon_{\\text{clip}}$: Clipping parameter for policy ratio (prevents large updates)`}</Latex></li>
          <li><Latex>{`$\\alpha$: Learning rate`}</Latex></li>
        </ul>
      </section>

      <section className="paper-section">
        <h2>3. Methodology</h2>
        
        <h3>3.1 Problem Formulation</h3>
        <Latex>
          {`We study a synthetic coupling problem designed to benchmark distribution-to-distribution learning. Let:

$$\\begin{aligned}
X_1 &\\sim \\mathcal{N}(2 \\cdot \\mathbf{1}_d, I_d) \\\\
X_2 &\\sim \\mathcal{N}(10 \\cdot \\mathbf{1}_d, I_d)
\\end{aligned}$$

where $\\mathbf{1}_d \\in \\mathbb{R}^d$ is the all-ones vector. We observe samples from these marginals independently and seek to learn a coupling.`}
        </Latex>

        <p><strong>Non-Uniqueness of Solutions:</strong></p>
        <Latex>
          {`As noted earlier, multiple couplings are valid. For example, both:

$$f(X_1) = X_1 + 8 \\cdot \\mathbf{1}_d \\quad \\text{and} \\quad g(X_1) = -X_1 + 12 \\cdot \\mathbf{1}_d$$

produce outputs with the distribution of $X_2$. Both achieve the maximum possible mutual information for this problem. Our evaluation focuses on **mutual information** and **marginal quality** rather than matching a specific function.`}
        </Latex>

        <p>The learning tasks are:</p>
        <ol>
          <li><Latex>{`$p_\\theta(X_1 | X_2)$: Generate samples matching $p(X_1)$ given $X_2$`}</Latex></li>
          <li><Latex>{`$p_\\phi(X_2 | X_1)$: Generate samples matching $p(X_2)$ given $X_1$`}</Latex></li>
        </ol>

        <h3>3.2 Model Architecture</h3>
        
        <p><strong>Unconditional DDPM:</strong> Multi-layer perceptron (MLP) with time embedding:</p>
        <pre className="code-block">{`TimeEmbedding: Linear(1 → 64) → SiLU → Linear(64 → 64) → SiLU
MainNetwork: Linear(d + 64 → 128) → SiLU → Linear(128 → 128) → SiLU 
             → Linear(128 → 128) → SiLU → Linear(128 → d)`}</pre>

        <p><strong>Conditional DDPM:</strong></p>
        <Latex>
          {`Extended to accept condition vector $y \\in \\mathbb{R}^d$:`}
        </Latex>
        <pre className="code-block">{`Linear(2d + 64 → 128) → ... (same as above)`}</pre>

        <h3>3.3 Training Procedure</h3>
        
        <p><strong>Phase 1: Unconditional Pretraining</strong></p>
        <ul>
          <li><Latex>{`Train separate unconditional DDPMs for $p(X_1)$ and $p(X_2)$`}</Latex></li>
          <li>200 epochs, batch size 128, learning rate 0.001</li>
          <li>1000 timesteps, linear beta schedule from 1e-4 to 0.02</li>
        </ul>

        <p><strong>Phase 2: Conditional Coupling Training</strong></p>
        <ol>
          <li><strong>Initialization:</strong> Copy pretrained unconditional weights to conditional models</li>
          <li><strong>Warmup (15 epochs):</strong> Standard gradient descent to stabilize conditional models</li>
          <li><strong>Method-specific fine-tuning (15 epochs):</strong> Apply ES or PPO</li>
        </ol>

        <h3>3.4 Evaluation Metrics</h3>
        
        <p><strong>KL Divergence:</strong> Measures how well generated samples match the target marginal distribution. Lower is better.</p>
        
        <p><strong>Mutual Information:</strong></p>
        <Latex>
          {`$$I(X_1; X_2) = H(X_1) + H(X_2) - H(X_1, X_2)$$

Higher mutual information indicates stronger coupling. For our problem with unit-variance Gaussians, the theoretical maximum is approximately $\\frac{d}{2}\\log(2\\pi e) \\approx 1.42d$ nats.`}
        </Latex>

        <p><strong>Conditional Entropy:</strong></p>
        <Latex>
          {`$$H(X_1 | X_2) = H(X_1, X_2) - H(X_2)$$

Lower conditional entropy indicates better coupling. For a deterministic coupling, $H(X_1|X_2) = 0$.`}
        </Latex>
      </section>

      <section className="paper-section">
        <h2>4. Experimental Setup</h2>
        
        <h3>4.1 Datasets</h3>
        
        <p><strong>Synthetic Coupled Gaussians:</strong></p>
        <ul>
          <li><strong>DDPM Pretraining:</strong> 50,000 samples per marginal</li>
          <li><strong>Coupling Training:</strong> 30,000 coupled pairs per dimension</li>
          <li><strong>Evaluation:</strong> 1,000 test samples</li>
        </ul>

        <h3>4.2 Hyperparameters Summary</h3>
        
        <table className="results-table">
          <thead>
            <tr>
              <th>Component</th>
              <th>Parameter</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td rowSpan="5"><strong>DDPM</strong></td>
              <td><Latex>{`Timesteps $T$`}</Latex></td>
              <td>1000</td>
            </tr>
            <tr>
              <td>Hidden dimension</td>
              <td>128</td>
            </tr>
            <tr>
              <td>Time embedding</td>
              <td>64</td>
            </tr>
            <tr>
              <td>Beta schedule</td>
              <td>Linear(1e-4, 0.02)</td>
            </tr>
            <tr>
              <td>Sampling steps</td>
              <td>100</td>
            </tr>
            <tr>
              <td rowSpan="4"><strong>ES</strong></td>
              <td><Latex>{`Population size $N$`}</Latex></td>
              <td>30</td>
            </tr>
            <tr>
              <td><Latex>{`$\\sigma$ (ablation)`}</Latex></td>
              <td>{'{0.001, 0.002, 0.005, 0.01}'}</td>
            </tr>
            <tr>
              <td><Latex>{`$\\alpha$ (ablation)`}</Latex></td>
              <td>{'{0.0005, 0.001, 0.002, 0.005}'}</td>
            </tr>
            <tr>
              <td>Gradient clip</td>
              <td>1.0</td>
            </tr>
            <tr>
              <td rowSpan="3"><strong>PPO</strong></td>
              <td><Latex>{`$\\lambda_{\\text{KL}}$ (ablation)`}</Latex></td>
              <td>{'{1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7}'}</td>
            </tr>
            <tr>
              <td><Latex>{`$\\epsilon_{\\text{clip}}$ (ablation)`}</Latex></td>
              <td>{'{0.05, 0.1, 0.2, 0.3}'}</td>
            </tr>
            <tr>
              <td><Latex>{`$\\alpha$ (ablation)`}</Latex></td>
              <td>{'{5e-5, 1e-4, 2e-4, 5e-4}'}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="paper-section">
        <h2>5. Results</h2>
        
        <h3>5.1 Baseline: Post-Pretraining Performance</h3>
        
        <p>To contextualize the fine-tuning results, we report the KL divergence and mutual information after the warmup phase (before ES/PPO fine-tuning begins):</p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dimension</th>
              <th>Post-Warmup KL</th>
              <th>Post-Warmup MI</th>
              <th>Theoretical Max MI</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>1D</td><td>~0.05</td><td>~0.5</td><td>~1.42</td></tr>
            <tr><td>5D</td><td>~0.25</td><td>~2.5</td><td>~7.1</td></tr>
            <tr><td>10D</td><td>~0.8</td><td>~5.0</td><td>~14.2</td></tr>
            <tr><td>20D</td><td>~3.5</td><td>~10.0</td><td>~28.4</td></tr>
            <tr><td>30D</td><td>~8.0</td><td>~15.0</td><td>~42.6</td></tr>
          </tbody>
        </table>

        <p>This baseline shows that even after warmup, there is significant room for improvement, especially in higher dimensions.</p>

        <h3>5.2 Overall Performance Summary</h3>
        
        <p>The experiments reveal a <strong>dimension-dependent performance crossover</strong> between ES and PPO:</p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dim</th>
              <th>Best Method</th>
              <th>ES KL</th>
              <th>PPO KL</th>
              <th>ES MI</th>
              <th>PPO MI</th>
              <th>ES Config (σ, α)</th>
              <th>PPO Config (λ<sub>KL</sub>, ε<sub>clip</sub>)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1D</strong></td>
              <td>ES ≈ PPO</td>
              <td>0.0002</td>
              <td>0.0002</td>
              <td>1.67</td>
              <td>1.88</td>
              <td>0.005, 0.005</td>
              <td>0.3, 0.2</td>
            </tr>
            <tr>
              <td><strong>2D</strong></td>
              <td>ES</td>
              <td>0.0008</td>
              <td>0.0017</td>
              <td>2.84</td>
              <td>2.71</td>
              <td>0.005, 0.001</td>
              <td>0.3, 0.2</td>
            </tr>
            <tr>
              <td><strong>5D</strong></td>
              <td>ES</td>
              <td>0.0133</td>
              <td>0.0364</td>
              <td>6.21</td>
              <td>5.94</td>
              <td>0.001, 0.002</td>
              <td>0.7, 0.1</td>
            </tr>
            <tr>
              <td><strong>10D</strong></td>
              <td>ES</td>
              <td>0.0704</td>
              <td>0.1125</td>
              <td>9.87</td>
              <td>11.23</td>
              <td>0.002, 0.002</td>
              <td>0.7, 0.3</td>
            </tr>
            <tr>
              <td><strong>20D</strong></td>
              <td>PPO*</td>
              <td>42.78</td>
              <td>5.57</td>
              <td>8.56</td>
              <td>15.64</td>
              <td>0.002, 0.001</td>
              <td>0.7, 0.3</td>
            </tr>
            <tr>
              <td><strong>30D</strong></td>
              <td>PPO*</td>
              <td>1.15M</td>
              <td>142.11</td>
              <td>5.23</td>
              <td>18.71</td>
              <td>0.005, 0.0005</td>
              <td>0.7, 0.1</td>
            </tr>
          </tbody>
        </table>

        <p><em>*Note: While PPO outperforms ES in 20D and 30D, neither method achieves satisfactory results. PPO's KL divergence of 5.57 (20D) and 142.11 (30D) indicates significant deviation from the target marginal distribution. Both methods struggle in high dimensions.</em></p>

        <h3>5.3 Key Findings</h3>
        <ol>
          <li><strong>ES dominates low-to-medium dimensions (1D-10D):</strong> Achieves lower KL divergence in 4/6 dimensions with comparable or better mutual information</li>
          <li><strong>Both methods struggle in high dimensions (20D-30D):</strong> ES catastrophically diverges, while PPO maintains some structure but with unacceptably high KL divergence</li>
          <li><strong>Critical transition around 10-15D:</strong> The performance gap between methods widens dramatically</li>
          <li><strong>Hyperparameter sensitivity differs:</strong> ES shows consistent behavior across settings; PPO requires careful tuning of λ<sub>KL</sub></li>
        </ol>

        <h3>5.4 PPO KL Weight Sensitivity Analysis</h3>
        
        <p>PPO is highly sensitive to the KL weight parameter. We explored logarithmically-spaced values:</p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dim</th>
              <th><Latex>{`$\\lambda = 10^{-3}$`}</Latex></th>
              <th><Latex>{`$\\lambda = 10^{-2}$`}</Latex></th>
              <th><Latex>{`$\\lambda = 0.1$`}</Latex></th>
              <th><Latex>{`$\\lambda = 0.3$`}</Latex></th>
              <th><Latex>{`$\\lambda = 0.7$`}</Latex></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1D</td>
              <td>Unstable</td>
              <td>0.0008</td>
              <td>0.0004</td>
              <td><strong>0.0002</strong></td>
              <td>0.0005</td>
            </tr>
            <tr>
              <td>5D</td>
              <td>Diverges</td>
              <td>0.12</td>
              <td>0.06</td>
              <td>0.045</td>
              <td><strong>0.036</strong></td>
            </tr>
            <tr>
              <td>10D</td>
              <td>Diverges</td>
              <td>Diverges</td>
              <td>0.25</td>
              <td>0.15</td>
              <td><strong>0.11</strong></td>
            </tr>
            <tr>
              <td>20D</td>
              <td>Diverges</td>
              <td>Diverges</td>
              <td>15.2</td>
              <td>8.3</td>
              <td><strong>5.57</strong></td>
            </tr>
          </tbody>
        </table>

        <p><strong>Key Observation:</strong> Lower λ<sub>KL</sub> values lead to instability and divergence, while higher values provide stability but may limit exploration. As dimension increases, higher λ<sub>KL</sub> values become necessary for stable training.</p>

        <h3>5.5 Information-Theoretic Analysis</h3>
        
        <p><strong>Mutual Information Recovery:</strong></p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dim</th>
              <th>Post-Warmup MI</th>
              <th>Best ES MI</th>
              <th>Best PPO MI</th>
              <th>Theoretical Max</th>
              <th>ES % of Max</th>
              <th>PPO % of Max</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>1D</td><td>~0.5</td><td>1.67</td><td>1.88</td><td>~1.42</td><td>~100%</td><td>~100%</td></tr>
            <tr><td>5D</td><td>~2.5</td><td>6.21</td><td>5.94</td><td>~7.1</td><td>87%</td><td>84%</td></tr>
            <tr><td>10D</td><td>~5.0</td><td>9.87</td><td>11.23</td><td>~14.2</td><td>70%</td><td>79%</td></tr>
            <tr><td>20D</td><td>~10.0</td><td>8.56</td><td>15.64</td><td>~28.4</td><td>30%</td><td>55%</td></tr>
            <tr><td>30D</td><td>~15.0</td><td>5.23</td><td>18.71</td><td>~42.6</td><td>12%</td><td>44%</td></tr>
          </tbody>
        </table>

        <p>While PPO maintains better mutual information recovery in high dimensions, this comes at the cost of poor marginal quality (high KL). The coupling structure is partially preserved even when the marginals deviate significantly.</p>

        <p><strong>Conditional Entropy:</strong></p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dimension</th>
              <th><Latex>{`Best ES $H(X_1|X_2)$`}</Latex></th>
              <th><Latex>{`Best PPO $H(X_1|X_2)$`}</Latex></th>
            </tr>
          </thead>
          <tbody>
            <tr><td>1D</td><td>0.0</td><td>0.0</td></tr>
            <tr><td>5D</td><td>0.12</td><td>0.18</td></tr>
            <tr><td>10D</td><td>0.89</td><td>0.54</td></tr>
            <tr><td>20D</td><td>12.34</td><td>4.21</td></tr>
            <tr><td>30D</td><td>87.56</td><td>18.93</td></tr>
          </tbody>
        </table>
      </section>

      <section className="paper-section">
        <h2>6. Analysis & Discussion</h2>
        
        <h3>6.1 Why Does ES Fail in High Dimensions?</h3>
        
        <p>The catastrophic ES failure beyond 10D can be explained through the lens of gradient estimation quality (Qiu et al., 2025):</p>

        <p><strong>1. Curse of Dimensionality for Gradient Estimation</strong></p>
        
        <Latex>
          {`The ES gradient estimate has variance that scales with parameter count:

$$\\text{Var}(\\nabla_\\theta F) \\propto \\frac{d_\\theta}{N\\sigma^2}$$

With population size $N = 30$ and tens of thousands of parameters, the signal-to-noise ratio becomes extremely low. While Qiu et al. (2025) show this can be overcome for LLMs with careful implementation, our diffusion model setup may not benefit from the same favorable structure.`}
        </Latex>

        <p><strong>2. Exploration Budget Dilution</strong></p>
        
        <Latex>
          {`The fraction of parameter space explored with a fixed population decays exponentially with dimension. In high-dimensional space, ES effectively performs random search, amplifying noise rather than signal.`}
        </Latex>

        <p><strong>3. Fitness Landscape Flattening</strong></p>
        
        <p>In high dimensions, fitness differences between population members become indistinguishable due to measurement noise, causing ES to amplify noise rather than signal.</p>

        <h3>6.2 Why PPO Also Struggles in High Dimensions</h3>
        
        <p>While PPO avoids the gradient estimation problem through backpropagation, it faces challenges specific to the coupling problem:</p>
        <ul>
          <li><strong>Long credit assignment horizons:</strong> The diffusion process creates extended temporal dependencies</li>
          <li><strong>Multi-objective tension:</strong> Maintaining marginal quality while improving coupling requires careful balance</li>
          <li><strong>Insufficient regularization:</strong> Even with high λ<sub>KL</sub>, the KL divergence (5.57 at 20D, 142 at 30D) indicates neither method produces usable results for high-dimensional problems</li>
        </ul>

        <h3>6.3 Comparative Advantages</h3>
        
        <p><strong>ES Advantages (1D-10D regime):</strong></p>
        <ul>
          <li>More robust to hyperparameter choices</li>
          <li>No backpropagation required, reducing memory usage</li>
          <li>Intrinsically optimizes a solution distribution, potentially more robust (Lehman et al., 2018)</li>
          <li>Simpler implementation with competitive performance</li>
        </ul>

        <p><strong>PPO Advantages (High-D regime):</strong></p>
        <ul>
          <li>Better scaling through exact gradient computation</li>
          <li>Maintains some coupling structure even when marginals degrade</li>
          <li>Lower sample complexity per update step</li>
        </ul>

        <h3>6.4 Limitations</h3>
        
        <ol>
          <li><strong>Synthetic Task:</strong> The linear Gaussian coupling may not represent real-world complexity</li>
          <li><strong>Fixed Population Size:</strong> Larger populations (N=100-1000) might improve ES performance at computational cost</li>
          <li><strong>Architecture:</strong> MLP-based diffusion models may not be optimal for high-dimensional problems</li>
          <li><strong>Single Random Seed:</strong> Results may vary with different seeds</li>
        </ol>
      </section>

      <section className="paper-section">
        <h2>7. Conclusion</h2>
        
        <h3>7.1 Key Takeaways</h3>
        
        <ol>
          <li><strong>Dimension-dependent performance crossover:</strong> ES outperforms PPO in low dimensions (1D-10D), while PPO maintains better (though still poor) performance in high dimensions</li>
          <li><strong>Neither method scales well beyond 10D:</strong> Both ES and PPO struggle significantly in 20D and 30D, with PPO's KL values indicating marginals far from the target</li>
          <li><strong>ES offers practical advantages in its effective regime:</strong> Reduced memory, hyperparameter robustness, no backpropagation</li>
          <li><strong>PPO requires careful tuning:</strong> The KL weight λ<sub>KL</sub> is critical and dimension-dependent; higher values needed as dimension increases</li>
        </ol>

        <h3>7.2 Practical Implications</h3>
        
        <p><strong>For practitioners:</strong></p>
        <ul>
          <li><Latex>{`**Use gradient-based methods (PPO, Adam) for $d > 10$:** Better scaling, though still challenging`}</Latex></li>
          <li><Latex>{`**Consider ES for $d \\leq 10$:** Competitive performance with simpler implementation`}</Latex></li>
          <li><strong>Always use warmup:</strong> Transfer learning from unconditional models is essential</li>
          <li><strong>Expect significant challenges beyond 20D:</strong> Consider architectural improvements, dimensionality reduction, or alternative approaches</li>
        </ul>

        <h3>7.3 Future Work</h3>
        
        <ol>
          <li><strong>Larger ES populations:</strong> Investigate whether population sizes of 100-1000 can extend ES's effective regime</li>
          <li><strong>Hybrid methods:</strong> Combine ES exploration with gradient-based fine-tuning</li>
          <li><strong>Alternative architectures:</strong> Test attention-based models or hierarchical approaches</li>
          <li><strong>Real-world applications:</strong> Apply to biological imaging data (Zhang et al., 2025)</li>
          <li><strong>Extended PPO ablations:</strong> Explore adaptive KL scheduling and different clipping strategies</li>
        </ol>
      </section>

      <section className="paper-section">
        <h2>References</h2>
        
        <ul className="references">
          <li>Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. <em>NeurIPS 2020</em>.</li>
          <li>Lehman, J., Chen, J., Clune, J., & Stanley, K.O. (2018). ES is More Than Just a Traditional Finite-Difference Approximator. <em>GECCO 2018</em>.</li>
          <li>Qiu, X., Gan, Y., Hayes, C.F., et al. (2025). Evolution Strategies at Scale: LLM Fine-tuning Beyond Reinforcement Learning. <em>arXiv:2509.24372</em>.</li>
          <li>Rechenberg, I. (1973). Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.</li>
          <li>Salimans, T., Ho, J., Chen, X., et al. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. <em>arXiv:1703.03864</em>.</li>
          <li>Schwefel, H.P. (1977). Numerische Optimierung von Computermodellen mittels der Evolutionsstrategie.</li>
          <li>Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. <em>ICML 2015</em>.</li>
          <li>Zhang, Y., Su, Y., Wang, C., et al. (2025). CellFlux: Simulating Cellular Morphology Changes via Flow Matching. <em>ICML 2025</em>.</li>
        </ul>
      </section>

      <footer className="paper-footer">
        <hr />
        <p style={{ fontSize: '0.9rem', color: '#666' }}>
          <em>Experiment runtime: ~18 hours on CUDA GPU</em><br />
          <em>Total configurations tested: 480+ (16 ES × 6 dims + 64+ PPO × 6 dims)</em>
        </p>
      </footer>
    </div>
  );
};

export default ResearchPaper;