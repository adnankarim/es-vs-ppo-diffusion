import React from 'react';
import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';
import './ResearchPaper.css';

const ResearchPaper = () => {
  return (
    <div className="research-paper">
      <header className="paper-header">
        <h1>
          Evolution Strategies vs PPO for Coupled Diffusion Models: A Comprehensive Ablation Study Across Dimensions
        </h1>
        {/* <p><strong>Author:</strong> Research Team</p>
        <p><strong>Date:</strong> December 13, 2024</p>
        <p><strong>Experiment ID:</strong> run_20251211_215609</p> */}
      </header>

      <section className="paper-section">
        <h2>1. Background & Motivation</h2>
        
        <h3>1.1 Problem Statement</h3>
        <Latex>
          {`Denoising Diffusion Probabilistic Models (DDPMs) have demonstrated remarkable capabilities in generative modeling, but training conditional DDPMs to learn complex joint distributions remains challenging. Specifically, when we need to learn coupled conditional distributions $p(X_1 | X_2)$ and $p(X_2 | X_1)$ where $X_1, X_2 \\in \\mathbb{R}^d$ are related random variables, gradient-based optimization methods may struggle due to:`}
        </Latex>

        <ol>
          <li><strong>High-dimensional noise landscapes:</strong> The stochastic nature of diffusion training introduces significant variance in gradient estimates</li>
          <li><strong>Coupling quality degradation:</strong> As dimensionality $d$ increases, maintaining accurate conditional dependencies becomes increasingly difficult</li>
          <li><Latex>
            {`**Optimization landscape complexity:** The interplay between marginal quality (matching $p(X_1)$ and $p(X_2)$) and coupling quality (preserving mutual information) creates a multi-objective optimization challenge`}
          </Latex></li>
        </ol>

        <h3>1.2 Motivation for This Study</h3>
        <p>
          While gradient descent with policy-based methods like Proximal Policy Optimization (PPO) has become standard in training conditional diffusion models, Evolution Strategies (ES) offer a fundamentally different optimization paradigm:
        </p>

        <ul>
          <li><strong>Gradient-free optimization:</strong> ES evaluates fitness directly without backpropagation through the diffusion process</li>
          <li><strong>Population-based exploration:</strong> ES maintains multiple parameter candidates simultaneously, potentially avoiding local optima</li>
          <li><strong>Robustness to noise:</strong> ES may be less sensitive to stochastic gradient variance</li>
        </ul>

        <Latex>
          {`However, ES's performance in the context of conditional diffusion model training remains under-explored, particularly as problem dimensionality scales. This study conducts a systematic ablation across dimensions $d \\in \\{1, 2, 5, 10, 20, 30\\}$ to understand:`}
        </Latex>

        <ol>
          <li><Latex>{`**Hyperparameter sensitivity:** How do key hyperparameters (ES: $\\sigma, \\alpha$; PPO: KL weight, clip parameter, learning rate) affect training across dimensions?`}</Latex></li>
          <li><strong>Scalability:</strong> At what dimensionality does each method break down?</li>
          <li><Latex>{`**Information-theoretic coupling quality:** How well does each method preserve mutual information $I(X_1; X_2)$ while learning accurate marginals?`}</Latex></li>
        </ol>

        <h3>1.3 Theoretical Context</h3>
        
        <p><strong>Diffusion Models:</strong></p>
        <Latex>
          {`A DDPM defines a forward diffusion process that progressively adds Gaussian noise to data $x_0$:

$$q(x_t | x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) I)$$

where $\\bar{\\alpha}_t = \\prod_{s=1}^t (1 - \\beta_s)$ with variance schedule $\\{\\beta_t\\}_{t=1}^T$. The reverse process learns to denoise:

$$p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\Sigma_\\theta(x_t, t))$$

Training objective (simplified):

$$\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\| \\epsilon - \\epsilon_\\theta(x_t, t) \\|^2 \\right]$$`}
        </Latex>

        <p><strong>Conditional Extension:</strong></p>
        <Latex>
          {`For conditional generation $p(x | y)$, the model becomes $\\epsilon_\\theta(x_t, t, y)$.`}
        </Latex>

        <p><strong>Evolution Strategies:</strong></p>
        <Latex>
          {`ES optimizes parameters $\\theta$ by sampling perturbations $\\epsilon_i \\sim \\mathcal{N}(0, \\sigma^2 I)$ and updating:

$$\\theta_{k+1} = \\theta_k + \\alpha \\frac{1}{n\\sigma} \\sum_{i=1}^n F(\\theta_k + \\epsilon_i) \\epsilon_i$$

where $F(\\cdot)$ is fitness (negative loss), $\\alpha$ is learning rate, $\\sigma$ is exploration noise, and $n$ is population size.`}
        </Latex>

        <p><strong>PPO for Diffusion:</strong></p>
        <Latex>
          {`While PPO was originally designed for reinforcement learning, it can be adapted for diffusion training by treating the denoising process as a sequential decision problem. The PPO objective includes:

$$\\mathcal{L}^{\\text{PPO}} = \\mathbb{E}_t \\left[ \\min\\left( r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) - \\lambda_{\\text{KL}} D_{\\text{KL}}(p_{\\theta_{\\text{old}}} \\| p_\\theta) \\right]$$

where $r_t(\\theta) = \\frac{p_\\theta(x_t)}{p_{\\theta_{\\text{old}}}(x_t)}$ is the probability ratio, $\\hat{A}_t$ is an advantage estimate, $\\lambda_{\\text{KL}}$ is the KL penalty weight, and $\\epsilon$ is the clipping parameter.`}
        </Latex>
      </section>

      <section className="paper-section">
        <h2>2. Methodology</h2>
        
        <h3>2.1 Problem Formulation</h3>
        <Latex>
          {`We consider a synthetic coupled Gaussian distribution designed to test conditional generation:

$$\\begin{aligned}
X_1 &\\sim \\mathcal{N}(2 \\cdot \\mathbf{1}_d, I_d) \\\\
X_2 &= X_1 + 8 \\cdot \\mathbf{1}_d
\\end{aligned}$$

where $\\mathbf{1}_d \\in \\mathbb{R}^d$ is the all-ones vector. This creates a deterministic coupling with known ground truth: $X_2 = X_1 + 8$. The task is to learn:`}
        </Latex>

        <ol>
          <li><Latex>{`$p_\\theta(X_1 | X_2)$: Should generate $X_1 \\approx X_2 - 8$`}</Latex></li>
          <li><Latex>{`$p_\\phi(X_2 | X_1)$: Should generate $X_2 \\approx X_1 + 8$`}</Latex></li>
        </ol>

        <h3>2.2 Model Architecture</h3>
        
        <p><strong>Unconditional DDPM:</strong> Multi-layer perceptron (MLP) with time embedding:</p>
        <pre className="code-block">{`TimeEmbedding: Linear(1 → 64) → SiLU → Linear(64 → 64) → SiLU
MainNetwork: Linear(d + 64 → 128) → SiLU → Linear(128 → 128) → SiLU 
             → Linear(128 → 128) → SiLU → Linear(128 → d)`}</pre>

        <p><strong>Conditional DDPM:</strong></p>
        <Latex>
          {`Extended to accept condition vector $y \\in \\mathbb{R}^d$:`}
        </Latex>
        <pre className="code-block">{`Linear(2d + 64 → 128) → ... (same as above)`}</pre>

        <Latex>
          {`The network predicts noise $\\epsilon_\\theta(x_t, t, y)$ at each timestep $t$.`}
        </Latex>
      </section>

      <section className="paper-section">
        <h2>3. Experimental Setup</h2>
        
        <h3>3.1 Datasets</h3>
        
        <p><strong>Synthetic Coupled Gaussians:</strong> Generated on-the-fly during training.</p>
        <ul>
          <li><strong>DDPM Pretraining:</strong> 50,000 samples per marginal</li>
          <li><strong>Coupling Training:</strong> 30,000 coupled pairs per dimension</li>
          <li><strong>Evaluation:</strong> 1,000 test samples</li>
        </ul>

        <h3>3.2 Hyperparameters Summary</h3>
        
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
              <td><Latex>Timesteps $T$</Latex></td>
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
              <td rowSpan="3"><strong>Training</strong></td>
              <td>Warmup epochs</td>
              <td>15</td>
            </tr>
            <tr>
              <td>Coupling epochs</td>
              <td>15</td>
            </tr>
            <tr>
              <td>Batch size</td>
              <td>128</td>
            </tr>
            <tr>
              <td rowSpan="4"><strong>ES</strong></td>
              <td><Latex>Population size $n$</Latex></td>
              <td>30</td>
            </tr>
            <tr>
              <td><Latex>$\\sigma$ (ablation)</Latex></td>
              <td>{'{0.001, 0.002, 0.005, 0.01}'}</td>
            </tr>
            <tr>
              <td>Learning rate (ablation)</td>
              <td>{'{0.0005, 0.001, 0.002, 0.005}'}</td>
            </tr>
            <tr>
              <td>Gradient clip</td>
              <td>1.0</td>
            </tr>
            <tr>
              <td rowSpan="3"><strong>PPO</strong></td>
              <td>KL weight (ablation)</td>
              <td>{'{0.1, 0.3, 0.5, 0.7}'}</td>
            </tr>
            <tr>
              <td>Clip param (ablation)</td>
              <td>{'{0.05, 0.1, 0.2, 0.3}'}</td>
            </tr>
            <tr>
              <td>Learning rate (ablation)</td>
              <td>{'{5e-5, 1e-4, 2e-4, 5e-4}'}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="paper-section">
        <h2>4. Results</h2>
        
        <h3>4.1 Overall Performance Summary</h3>
        
        <p>The experiments reveal a <strong>dimension-dependent performance crossover</strong> between ES and PPO:</p>

        <table className="results-table">
          <thead>
            <tr>
              <th>Dimension</th>
              <th>ES Winner</th>
              <th>PPO Winner</th>
              <th>Best ES KL</th>
              <th>Best PPO KL</th>
              <th>ES Corr.</th>
              <th>PPO Corr.</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1D</strong></td>
              <td>✓</td>
              <td></td>
              <td>0.0002</td>
              <td>0.0002</td>
              <td>0.9813</td>
              <td>0.9953</td>
            </tr>
            <tr>
              <td><strong>2D</strong></td>
              <td>✓</td>
              <td></td>
              <td>0.0008</td>
              <td>0.0017</td>
              <td>0.9896</td>
              <td>0.9842</td>
            </tr>
            <tr>
              <td><strong>5D</strong></td>
              <td>✓</td>
              <td></td>
              <td>0.0133</td>
              <td>0.0364</td>
              <td>0.9841</td>
              <td>0.9838</td>
            </tr>
            <tr>
              <td><strong>10D</strong></td>
              <td>✓</td>
              <td></td>
              <td>0.0704</td>
              <td>0.1125</td>
              <td>0.9533</td>
              <td>0.9678</td>
            </tr>
            <tr>
              <td><strong>20D</strong></td>
              <td></td>
              <td>✓</td>
              <td>42.78</td>
              <td>5.57</td>
              <td>0.6617</td>
              <td>0.7898</td>
            </tr>
            <tr>
              <td><strong>30D</strong></td>
              <td></td>
              <td>✓</td>
              <td>1,152,910</td>
              <td>142.11</td>
              <td>0.4206</td>
              <td>0.5619</td>
            </tr>
          </tbody>
        </table>

        <p><strong>Key Findings:</strong></p>
        <ol>
          <li><strong>ES dominates low-to-medium dimensions (1D-10D):</strong> Achieves lower KL divergence in 4/6 dimensions</li>
          <li><strong>PPO dominates high dimensions (20D-30D):</strong> ES catastrophically diverges beyond 10D</li>
          <li><strong>Critical transition at ~15D:</strong> Performance gap widens dramatically at 20D</li>
          <li><strong>Overall winner: ES (4/6 dimensions)</strong></li>
        </ol>
      </section>

      <section className="paper-section">
        <h2>5. Analysis & Discussion</h2>
        
        <h3>5.1 Why Does ES Fail in High Dimensions?</h3>
        
        <p>The catastrophic ES failure beyond 10D can be explained through multiple lenses:</p>

        <p><strong>1. Curse of Dimensionality for Gradient Estimation</strong></p>
        
        <Latex>
          {`ES gradient estimate:

$$\\nabla_\\theta F \\approx \\frac{1}{n\\sigma} \\sum_{i=1}^n \\tilde{F}_i \\epsilon_i$$

As $d_\\theta$ (parameter count) grows, the variance of this estimator scales as:

$$\\text{Var}(\\nabla_\\theta F) \\propto \\frac{d_\\theta}{n\\sigma^2}$$

For our 30D conditional MLP, the signal-to-noise ratio becomes negligibly small, making gradient estimates pure noise.`}
        </Latex>

        <p><strong>2. Exploration Budget Dilution</strong></p>
        
        <Latex>
          {`In $d_\\theta$-dimensional space, a fixed population of $n = 30$ samples explores only a tiny fraction of the space. The "volume" of explored region relative to total parameter space decays exponentially, meaning ES effectively performs random search in high dimensions.`}
        </Latex>

        <p><strong>3. Fitness Landscape Flattening</strong></p>
        
        <p>In high dimensions, fitness differences between population members become indistinguishable due to measurement noise, causing ES to amplify noise rather than signal.</p>

        <p><strong>4. Lack of Gradient Signal Reuse</strong></p>
        
        <p>PPO (and gradient descent) computes exact gradients via backpropagation, which scale well with dimension. ES must re-estimate gradients from scratch at each step using expensive function evaluations, without any information reuse.</p>

        <h3>5.2 Why Does PPO Succeed?</h3>
        
        <p>PPO's robustness in high dimensions stems from:</p>

        <p><strong>1. Exact Gradient Computation</strong></p>
        
        <Latex>
          {`Backpropagation provides unbiased, low-variance gradients:

$$\\nabla_\\theta \\mathcal{L} = \\frac{\\partial \\mathcal{L}}{\\partial \\theta}$$

Variance is controlled by mini-batch size, not dimensionality.`}
        </Latex>

        <p><strong>2. Adaptive Regularization</strong></p>
        
        <Latex>
          {`The KL penalty term acts as an adaptive trust region, preventing large policy shifts that could cause instability. In high dimensions, this regularization becomes critical.`}
        </Latex>

        <p><strong>3. Lower Sample Complexity Per Update</strong></p>
        
        <p>PPO requires only 1 forward + 1 backward pass per batch, vs. ES's 30 forward passes (population size). This allows PPO to "see" more data during training.</p>
      </section>

      <section className="paper-section">
        <h2>6. Conclusion</h2>
        
        <h3>6.1 Key Takeaways</h3>
        
        <ol>
          <li><strong>Dimension-dependent performance crossover:</strong> ES wins in low dimensions (1D-10D), PPO dominates in high dimensions (20D+)</li>
          <li><strong>ES breakdown at ~15D:</strong> Evolution Strategies suffer catastrophic failure beyond 10D</li>
          <li><strong>PPO's robust scaling:</strong> Gradient-based optimization with adaptive regularization maintains trainability even in 30D</li>
          <li><strong>Hyperparameter trends:</strong> ES requires decreasing exploration noise as dimension grows; PPO requires increasing regularization</li>
          <li><strong>Information-theoretic perspective:</strong> Mutual information preservation is the critical challenge</li>
        </ol>

        <h3>6.2 Practical Implications</h3>
        
        <p><strong>For practitioners:</strong></p>
        <ul>
          <li><Latex>{`**Use gradient-based methods (PPO, Adam) for $d > 10$:** The computational and performance advantages are overwhelming`}</Latex></li>
          <li><Latex>{`**Consider ES for $d \\leq 5$:** Competitive performance with simpler implementation`}</Latex></li>
          <li><strong>Always use warmup:</strong> Transfer learning from unconditional models is essential</li>
          <li><strong>Expect degradation beyond 20D:</strong> Consider architectural improvements or dimensionality reduction</li>
        </ul>
      </section>

      <footer className="paper-footer">
        <p><strong>Acknowledgments:</strong> This work was conducted as part of research into gradient-free optimization for generative models.</p>
        
        <p><strong>Code:</strong> Complete implementation available in <code>run_ablation_study.py</code> (2091 lines, documented).</p>
        
        <hr />
        
        <p style={{ fontSize: '0.9rem', color: '#666' }}>
          {/* <em>Last updated: December 13, 2024</em><br /> */}
          <em>Experiment runtime: ~18 hours on CUDA GPU</em><br />
          <em>Total configurations tested: 480 (16 ES × 6 dims + 64 PPO × 6 dims)</em>
        </p>
      </footer>
    </div>
  );
};

export default ResearchPaper;

