import React, { useState, useEffect } from 'react';
import Latex from 'react-latex-next';

// --- Theme & Styles ---
const colors = {
    primary: '#0f766e', // Teal 700
    secondary: '#0d9488', // Teal 600
    accent: '#f59e0b', // Amber 500
    text: '#1e293b', // Slate 800
    textLight: '#475569', // Slate 600
    bg: '#f8fafc', // Slate 50
    slideBg: '#ffffff',
    success: '#16a34a',
    danger: '#dc2626',
    theta: '#2563eb',
    phi: '#dc2626'
};

const Slide = ({ children, title, subtitle }) => (
    <div style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 40px',
        boxSizing: 'border-box',
        overflow: 'hidden',
        fontFamily: "'Inter', sans-serif"
    }}>
        {title && (
            <div style={{ marginBottom: '10px', borderBottom: `2px solid ${colors.accent}`, paddingBottom: '5px', flexShrink: 0 }}>
                <h2 style={{
                    fontSize: '2rem',
                    color: colors.primary,
                    margin: 0,
                    fontWeight: 700
                }}>
                    {title}
                </h2>
                {subtitle && <div style={{ fontSize: '1.2rem', color: colors.textLight, marginTop: '4px', fontWeight: 300 }}>{subtitle}</div>}
            </div>
        )}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', fontSize: '1.2rem', color: colors.text, overflow: 'hidden' }}>
            {children}
        </div>
    </div>
);

const PresentationMode = ({ onExit }) => {
    const [currentSlide, setCurrentSlide] = useState(0);

    const slides = [
        // Slide 1: Title
        <Slide>
            <div style={{ textAlign: 'center' }}>
                <h1 style={{ fontSize: '3.5rem', fontWeight: 800, color: colors.primary, marginBottom: '24px' }}>
                    Evolution Strategies vs PPO<br />
                    <span style={{ fontSize: '2.5rem', fontWeight: 400, color: colors.textLight }}>for Diffusion-Based MEC</span>
                </h1>
                <div style={{ fontSize: '1.6rem', color: colors.text, marginBottom: '40px' }}>
                    From synthetic Gaussian couplings (1D→30D) to BBBC021 cellular morphology
                </div>
                <div style={{ fontSize: '1.4rem', fontStyle: 'italic', color: colors.textLight, marginBottom: '60px' }}>
                    Theme: RL fine-tuning for minimum-entropy coupling (MEC) with diffusion “policies”
                </div>
            </div>
        </Slide>,

        // Slide 2: Roadmap
        <Slide title="Roadmap">
            <ol style={{ lineHeight: 1.8, fontSize: '1.5rem', marginLeft: '40px' }}>
                <li><strong>Synthetic experiments</strong>: Gaussian coupling ablation (1D→30D)</li>
                <li><strong>Methodology</strong>: How PPO & ES update diffusion models (loss / reward / KL)</li>
                <li><strong>Biology</strong>: BBBC021 + Conditional DDPM&MEC + ES/PPO fine-tuning</li>
                <li><strong>Approaches</strong>: Unconditional DDPM + PPO, Flux Matching, SD + LoRA</li>
                <li><strong>Conclusion</strong>: Results + image references</li>
            </ol>
        </Slide>,

        // Slide 3: Why Synthetic?
        <Slide title="Why a synthetic benchmark first?">
            <ul style={{ lineHeight: 1.8, fontSize: '1.5rem', marginLeft: '40px' }}>
                <li>Diffusion RL fine-tuning can <strong>drift</strong> from the pretrained prior (bad samples / collapse)</li>
                <li>High-dimensional latents make optimization noisy and deceptive</li>
                <li>Synthetic Gaussians give:
                    <ul style={{ marginTop: '10px' }}>
                        <li>ground-truth structure</li>
                        <li>controlled dimensionality (<Latex>{`$d = 1\\dots30$`}</Latex>)</li>
                        <li>clean metrics (KL, MAE, MI, entropy)</li>
                    </ul>
                </li>
            </ul>
        </Slide>,

        // Slide 4: Synthetic Task
        <Slide title="Synthetic task: learn a coupling" subtitle="From unpaired marginals">
            <div style={{ lineHeight: 1.6 }}>
                <p>We observe <em>unpaired</em> datasets:</p>
                <ul style={{ marginLeft: '40px', marginBottom: '20px' }}>
                    <li><Latex>{`$x_1 \\sim \\mathcal{N}(2, 0.99)$`}</Latex></li>
                    <li><Latex>{`$x_2 \\sim \\mathcal{N}(10, 1.00)$`}</Latex></li>
                </ul>
                <p>Target coupling manifold:</p>
                <ul style={{ marginLeft: '40px', marginBottom: '20px' }}>
                    <li><strong>shift constraint:</strong> <Latex>{`$x_2 \\approx x_1 + 8$`}</Latex></li>
                </ul>
                <p>Dimensional version:</p>
                <ul style={{ marginLeft: '40px' }}>
                    <li><Latex>{`$x_1, x_2 \\in \\mathbb{R}^d$`}</Latex>, with <Latex>{`$d \\in \\{1,2,\\dots,30\\}$`}</Latex></li>
                </ul>
            </div>
        </Slide>,

        // Slide 5: Synthetic Architecture
        <Slide title="Synthetic Architecture" subtitle="Conditional Multi-Dimensional MLP">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', alignItems: 'center' }}>
                <div>
                    <p><strong>Backbone:</strong> Conditional MLP taking <Latex>{`$\\mathbf{x}_t, t, \\mathbf{c}$`}</Latex>.</p>
                    <ul style={{ lineHeight: 1.5 }}>
                        <li><strong>Time Embedding:</strong> Gaussian Fourier <Latex>{`$\\sin(2\\pi t f)$`}</Latex></li>
                        <li><strong>Input:</strong> Concat <Latex>{`$[\\mathbf{x}_t, \\mathbf{c}, \\text{emb}(t)]$`}</Latex></li>
                        <li><strong>Body:</strong> 3x Residual Blocks (Linears + SiLU)</li>
                        <li><strong>Output:</strong> Noise <Latex>{`$\\epsilon_\\theta \\in \\mathbb{R}^d$`}</Latex></li>
                    </ul>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#ffffff', padding: '10px', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                    <img src="/synth/block.jpg" style={{ maxHeight: '60vh', maxWidth: '100%', objectFit: 'contain' }} alt="Synthetic Block Diagram" />
                </div>
            </div>
        </Slide >,

        // Slide 5: Diffusion Basics
        <Slide title="Diffusion basics for Gaussians" subtitle="Pretraining Step">
            <p>Forward noising (DDPM):</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$x_t = \\sqrt{\\bar{\\alpha}_t}\\,x_0 + \\sqrt{1-\\bar{\\alpha}_t}\\,\\epsilon,\\quad \\epsilon\\sim\\mathcal{N}(0,I)$`}</Latex>
            </div>
            <p>Noise-prediction objective:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$\\mathcal{L}_{\\text{DDPM}}(\\theta)=\\mathbb{E}\\left[\\|\\epsilon - \\epsilon_\\theta(x_t,t,\\text{cond})\\|^2\\right]$`}</Latex>
            </div>
            <p><strong>Pretraining goal:</strong> learn correct <em>marginals</em> before RL coupling.</p>
            <div style={{ padding: '20px', background: '#f1f5f9', borderRadius: '8px', textAlign: 'center', color: colors.textLight, marginTop: '20px' }}>
                [Images removed. Pretraining successfully learns independent marginals.]
            </div>
        </Slide>,

        // Slide 6: Two Conditionals
        <Slide title="Two conditional diffusion models" subtitle="Specular Setup">
            <p>We train <strong>two</strong> conditionals (a "pair of translators"):</p>
            <ul style={{ marginLeft: '40px', marginBottom: '20px' }}>
                <li>Forward: <Latex>{`$p_\\theta(x_2 \\mid x_1)$`}</Latex></li>
                <li>Backward: <Latex>{`$p_\\phi(x_1 \\mid x_2)$`}</Latex></li>
            </ul>
            <p>Key idea: enforce <strong>cycle consistency</strong> <em>without paired data</em>:</p>
            <ul style={{ marginLeft: '40px' }}>
                <li>sample <Latex>{`$\\hat{x}_2 \\sim p_\\theta(\\cdot \\mid x_1)$`}</Latex></li>
                <li>score with the partner: <Latex>{`$p_\\phi(x_1 \\mid \\hat{x}_2)$`}</Latex></li>
            </ul>
        </Slide>,

        // Slide 7: MEC Objective
        <Slide title="MEC-style objective" subtitle="Likelihood + Trust Region">
            <p>Reward using partner model (“scorer”):</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$R_\\theta(x_1,\\hat{x}_2)\\;\\approx\\;\\log p_\\phi(x_1\\mid \\hat{x}_2)$`}</Latex>
            </div>
            <p>Anchor / trust region to preserve the prior:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$J(\\theta) = \\mathbb{E}_{\\tau\\sim\\pi_\\theta}\\left[R(\\tau)\\right] - \\beta\\,D_{KL}\\left(\\pi_\\theta \\,::\\, \\pi_{\\text{anchor}}\\right)$`}</Latex>
            </div>
            <p><strong>Interpretation:</strong> maximize coupling quality <strong>while staying close</strong> to the pretrained diffusion behavior.</p>
        </Slide>,

        // Slide 8: Diffusion as Policy
        <Slide title="Diffusion as a policy" subtitle="What PPO Optimizes">
            <p>Treat reverse diffusion as an MDP:</p>
            <ul style={{ marginLeft: '40px', marginBottom: '20px' }}>
                <li>State: <Latex>{`$s_t = x_t$`}</Latex></li>
                <li>Action: <Latex>{`$a_t = x_{t-1}$`}</Latex></li>
                <li>Policy: <Latex>{`$\\pi_\\theta(a_t|s_t) = p_\\theta(x_{t-1}\\mid x_t, \\text{cond})$`}</Latex></li>
            </ul>
            <p>For each step, policy is Gaussian:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$p_\\theta(x_{t-1}\\mid x_t)=\\mathcal{N}\\big(\\mu_\\theta(x_t,t,\\text{cond}),\\sigma_t^2 I\\big)$`}</Latex>
            </div>
            <p>So we can compute <Latex>{`$\\log \\pi_\\theta(a_t|s_t)$`}</Latex> and KL analytically.</p>
        </Slide>,

        // Slide 9: PPO Update
        <Slide title="PPO Update" subtitle="Clipped Objective + Anchor Penalty">
            <p>Define probability ratio:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$r_t(\\theta)=\\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$`}</Latex>
            </div>
            <p>Clipped PPO objective:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$\\mathcal{L}_{\\text{PPO}}(\\theta)=\\mathbb{E}\\left[\\min\\left(r_tA_t,\\;\\text{clip}(r_t,1-\\epsilon,1+\\epsilon)A_t\\right)\\right]$`}</Latex>
            </div>
            <p><strong>Update:</strong> gradient-based backprop through <Latex>{`$\\mathcal{L}$`}</Latex>.</p>
        </Slide>,

        // Slide 10: ES Update
        <Slide title="ES Update" subtitle="Finite-Difference Gradient">
            <p>Sample perturbations <Latex>{`$\\epsilon_i \\sim \\mathcal{N}(0,I)$`}</Latex> and evaluate fitness <Latex>{`$F(\\theta+\\sigma\\epsilon_i)$`}</Latex>.</p>
            <p>Gradient estimate:</p>
            <div style={{ textAlign: 'center', margin: '20px 0' }}>
                <Latex>{`$\\nabla_\\theta J \\approx \\frac{1}{n\\sigma}\\sum_{i=1}^n F(\\theta+\\sigma\\epsilon_i)\\,\\epsilon_i$`}</Latex>
            </div>
            <p>Variance reduction: antithetic pairs <Latex>{`$\\epsilon, -\\epsilon$`}</Latex>.</p>
            <p><strong>Update:</strong> <Latex>{`$\\theta \\leftarrow \\theta + \\alpha \\cdot \\text{Adam}(\\nabla_\\theta J)$`}</Latex></p>
        </Slide>,

        // Slide 11: PPO vs ES
        <Slide title="How PPO vs ES Update Loss">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
                <div>
                    <h3>Shared Ingredients</h3>
                    <ol>
                        <li>Rollout trajectory <Latex>{`$\\hat{x}$`}</Latex></li>
                        <li>Compute <strong>Reward</strong> (Partner Likelihood)</li>
                        <li>Apply <strong>Trust Region</strong> (KL to Anchor)</li>
                    </ol>
                </div>
                <div>
                    <h3>Key Difference</h3>
                    <ul>
                        <li style={{ marginBottom: '15px' }}><strong>PPO</strong>: Differentiates <em>through</em> log-probs. Lower variance, but can overfit ("hack") the reward.</li>
                        <li><strong>ES</strong>: Treats model as black-box. Robust exploration, often better at <strong>prior preservation</strong>.</li>
                    </ul>
                </div>
            </div>
        </Slide>,

        // Slide 13: Optimization Details
        <Slide title="Synthetic Optimization Details">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
                <div style={{ background: '#f0f9ff', padding: '30px', borderRadius: '12px' }}>
                    <h3>PPO Grid</h3>
                    <ul>
                        <li>KL Weight <Latex>{`$\\beta$`}</Latex>: <Latex>{`$\\{10^{-4}, 10^{-3}, 3\\cdot10^{-3}\\}$`}</Latex></li>
                        <li>LR: <Latex>{`$\\{10^{-5}, 10^{-4}\\}$`}</Latex></li>
                        <li>Clip <Latex>{`$\\epsilon$`}</Latex>: <Latex>{`$\\{0.02, 0.1\\}$`}</Latex></li>
                    </ul>
                </div>
                <div style={{ background: '#fff7ed', padding: '30px', borderRadius: '12px' }}>
                    <h3>ES Grid</h3>
                    <ul>
                        <li>Noise Scale <Latex>{`$\\sigma$`}</Latex>: <Latex>{`$\\{0.001, 0.01\\}$`}</Latex></li>
                        <li>LR: <Latex>{`$\\{10^{-4}, 10^{-3}\\}$`}</Latex></li>
                        <li>Population: 15 (Antithetic)</li>
                    </ul>
                </div>
            </div>
        </Slide>,

        // Slide 15: Synthetic Results Table
        <Slide title="Synthetic Results (1D → 30D)" subtitle="ES vs PPO">
            <table style={{ width: '100%', fontSize: '1.2rem', borderCollapse: 'collapse', textAlign: 'center' }}>
                <thead style={{ background: colors.primary, color: 'white' }}>
                    <tr>
                        <th style={{ padding: '15px' }}>Dim</th>
                        <th style={{ padding: '15px' }}>KL (ES) ↓</th>
                        <th style={{ padding: '15px' }}>KL (PPO) ↓</th>
                        <th style={{ padding: '15px' }}>MAE (ES)</th>
                        <th style={{ padding: '15px' }}>MAE (PPO)</th>
                        <th style={{ padding: '15px', background: '#0d9488' }}>WINNER</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style={{ borderBottom: '1px solid #ccc' }}>
                        <td>1</td> <td>0.001</td> <td>0.003</td> <td>1.122</td> <td>1.125</td> <td>ES</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ccc' }}>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ccc' }}>
                        <td>5</td> <td>0.001</td> <td>0.001</td> <td>1.116</td> <td>1.122</td> <td>ES</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ccc' }}>
                        <td>10</td> <td>0.001</td> <td>0.002</td> <td>1.122</td> <td>1.125</td> <td>ES</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ccc' }}>
                        <td>20</td> <td>0.001</td> <td>0.005</td> <td>1.120</td> <td>1.122</td> <td>ES </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ccc', background: '#f0fdf4', border: `2px solid ${colors.success}` }}>
                        <td style={{ fontWeight: 'bold' }}>30</td>
                        <td style={{ color: colors.success, fontWeight: 'bold' }}>0.002</td>
                        <td style={{ color: colors.danger }}>0.015</td>
                        <td>1.122</td>
                        <td>1.121</td>
                        <td style={{ fontWeight: 'bold', color: colors.success }}>ES (Significant)</td>
                    </tr>
                </tbody>
            </table>
            <div style={{ marginTop: '20px', textAlign: 'center', fontStyle: 'italic' }}>
                ES maintains ~10× lower KL than PPO in 30D.
            </div>
        </Slide>,

        // Slide 16: Synthetic Takeaway
        <Slide title="Synthetic Takeaway" subtitle="Why ES can be preferable">
            <ul style={{ lineHeight: 1.8, fontSize: '1.5rem', marginLeft: '40px' }}>
                <li>PPO is a strong “gradient hammer,” but can:
                    <ul>
                        <li>chase reward shortcuts</li>
                        <li>drift from the anchor prior in high-d settings</li>
                    </ul>
                </li>
                <li>ES often:
                    <ul>
                        <li>preserves the prior better (lower marginal KL)</li>
                        <li>behaves more stably with noisy likelihood-based rewards</li>
                    </ul>
                </li>
            </ul>
            <div style={{ background: '#fff7ed', padding: '20px', borderRadius: '12px', marginTop: '30px', borderLeft: `6px solid ${colors.accent}` }}>
                <strong>Implication for Biology:</strong> Prior preservation matters → avoids “biologically implausible” morphologies.
            </div>
        </Slide>,

        // Slide 17: Biological Experiments
        <Slide title="Biological Experiments (BBBC021)" subtitle="Counterfactual Morphology Prediction">
            <ul style={{ lineHeight: 1.8, fontSize: '1.5rem', marginLeft: '40px' }}>
                <li><strong>Control cells:</strong> DMSO (healthy baseline)</li>
                <li><strong>Treated cells:</strong> Compound exposure (Taxol, etc.)</li>
                <li><strong>Challenge:</strong> Unpaired snapshots (no true per-cell counterfactual pairs)</li>
            </ul>
        </Slide>,

        // Slide 18: BBBC021 Dataset
        <Slide title="BBBC021 Dataset" subtitle="Structure & Preprocessing">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', height: '100%', overflow: 'hidden' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <p style={{ fontSize: '1.1rem' }}>~97,504 single-cell crops, 35 compounds. Batch-disjoint split.</p>
                    <img src="/top_compounds.png" style={{ width: '100%', maxHeight: '35vh', objectFit: 'contain' }} />
                    <img src="/dmso_vs_treated.png" style={{ width: '100%', maxHeight: '35vh', objectFit: 'contain' }} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                    <img src="/split_counts.png" style={{ width: '100%', maxHeight: '35vh', objectFit: 'contain' }} />
                    <img src="/batch_size_hist.png" style={{ width: '100%', maxHeight: '35vh', objectFit: 'contain' }} />
                    <p style={{ fontSize: '1rem', marginTop: '5px' }}>
                        <strong>Preprocessing:</strong> Crop 96x96x3, Normalize [-1, 1].
                    </p>
                </div>
            </div>
        </Slide>,

        // Slide 19: FID & cFID Explained
        <Slide title="Evaluation Metrics" subtitle="How we measure success">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
                <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '12px', border: `1px solid ${colors.secondary}` }}>
                    <h3 style={{ color: colors.primary, marginTop: 0 }}>FID (Fréchet Inception Distance)</h3>
                    <p style={{ fontSize: '1.1rem' }}>Measures <strong>overall realism</strong>. Distance between Gaussian stats of InceptionV3 features of Real vs Generated images.</p>
                    <div style={{ textAlign: 'center', margin: '15px 0' }}>
                        <Latex>{`$\\text{FID} = \\|\\mu_r-\\mu_g\\|^2 + \\text{Tr}(\\Sigma_r+\\Sigma_g-2(\\Sigma_r\\Sigma_g)^{1/2})$`}</Latex>
                    </div>
                    <div style={{ fontStyle: 'italic', fontSize: '0.9rem', textAlign: 'center' }}>Lower is better.</div>
                </div>
                <div style={{ background: '#fff7ed', padding: '20px', borderRadius: '12px', border: `1px solid ${colors.accent}` }}>
                    <h3 style={{ color: colors.theta, marginTop: 0 }}>cFID (Conditional FID)</h3>
                    <p style={{ fontSize: '1.1rem' }}>Measures <strong>conditional consistency</strong>. Computes FID <em>per compound</em> and averages them.</p>
                    <div style={{ textAlign: 'center', margin: '15px 0' }}>
                        <Latex>{`$\\text{cFID} = \\mathbb{E}_{d}[\\text{FID}(\\text{Real}_d, \\text{Gen}_d)]$`}</Latex>
                    </div>
                    <div style={{ fontStyle: 'italic', fontSize: '0.9rem', textAlign: 'center' }}>Does "Taxol" look like "Taxol"?</div>
                </div>
            </div>
        </Slide>,

        // Slide 20: Conditional DDPM Architecture
        <Slide title="Conditional DDPM Architecture" subtitle="Modified UNet2DModel">
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '30px', alignItems: 'center' }}>
                <div>
                    <p>Standard UNet adapted for biological conditioning:</p>
                    <ul style={{ lineHeight: 1.6 }}>
                        <li><strong>Dual-Image Input (6 Channels):</strong>
                            <ul>
                                <li><strong>Ch 1-3:</strong> Noisy Target <Latex>{`$x_t$`}</Latex></li>
                                <li><strong>Ch 4-6:</strong> Clean Conditioning <Latex>{`$x_0^c$`}</Latex> (Control)</li>
                            </ul>
                        </li>
                        <li><strong>Chemical Injection:</strong>
                            <ul>
                                <li>1024-dim Morgan Fingerprint.</li>
                                <li>Projected via MLP to time-embed dim.</li>
                                <li>Injected via <code>class_labels</code> embedding.</li>
                            </ul>
                        </li>
                    </ul>
                </div>
                <div style={{ background: '#f0fdf4', padding: '20px', borderRadius: '12px', border: '1px solid #bbf7d0', textAlign: 'center' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#166534' }}>Input Tensor</div>
                    <div style={{ fontSize: '0.9rem', color: '#166534', fontFamily: 'monospace' }}>[Batch, 6, 96, 96]</div>
                    <hr style={{ margin: '15px 0', borderColor: '#bbf7d0' }} />
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#166534' }}>Conditioning</div>
                    <div style={{ fontSize: '0.9rem', color: '#166534' }}>ClassEmb(Fingerprint)</div>
                </div>
            </div>
        </Slide>,

        // Slide 20: Cond DDPM Setup
        <Slide title="Conditional DDPM + MEC" subtitle="DDMEC Setup">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '40px' }}>
                <div>
                    <h3>Conditioning Inputs:</h3>
                    <ul>
                        <li>Control Image <Latex>{`$x_0^c$`}</Latex></li>
                        <li>Drug Fingerprint <Latex>{`$d \\in \\{0,1\\}^{1024}$`}</Latex></li>
                    </ul>
                    <h3>UNet Trick:</h3>
                    <ul>
                        <li>Concatenate <Latex>{`$[x_t \\;\\Vert\\; x_0^c]$`}</Latex> (6 Channels)</li>
                        <li>Inject fingerprint via Class Embedding</li>
                    </ul>
                </div>
            </div>
        </Slide>,

        // Slide 21: Conditional Baseline Results
        <Slide title="Conditional Baseline Results">
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '20px' }}>
                <table style={{ width: '100%', fontSize: '1.2rem', borderCollapse: 'collapse' }}>
                    <thead style={{ background: colors.primary, color: 'white' }}>
                        <tr>
                            <th style={{ padding: '10px' }}>Direction</th>
                            <th style={{ padding: '10px' }}>Initial FID</th>
                            <th style={{ padding: '10px' }}>Final FID (Epoch 100)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style={{ borderBottom: '1px solid #ddd' }}>
                            <td style={{ padding: '15px' }}>Forward (Control → Treated)</td>
                            <td style={{ padding: '15px' }}>~160</td>
                            <td style={{ padding: '15px', fontWeight: 'bold' }}>24.49</td>
                        </tr>
                        <tr style={{ borderBottom: '1px solid #ddd' }}>
                            <td style={{ padding: '15px' }}>Inverse (Treated → Control)</td>
                            <td style={{ padding: '15px' }}>~160</td>
                            <td style={{ padding: '15px', fontWeight: 'bold' }}>43.97</td>
                        </tr>
                    </tbody>
                </table>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', flex: 1, overflow: 'hidden' }}>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.9rem', marginBottom: '5px' }}>Phi (Control) Epoch 100</div>
                        <img src="/bio/phi/phi_epoch_100.png" style={{ maxHeight: '35vh', maxWidth: '100%', objectFit: 'contain' }} />
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '0.9rem', marginBottom: '5px' }}>Theta (Treated) Epoch 100</div>
                        <img src="/bio/theta/theta_epoch_100.png" style={{ maxHeight: '35vh', maxWidth: '100%', objectFit: 'contain' }} />
                    </div>
                </div>
            </div>
        </Slide>,

        // Slide 22: ES-DDMEC Fine-tuning
        <Slide title="ES-DDMEC Fine-tuning" subtitle="Co-Evolution Loop">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', alignItems: 'center', height: '100%' }}>
                <div>
                    <p>Core idea: alternate <strong>ES reward maximization</strong> and <strong>supervised diffusion regression</strong>.</p>
                    <ol style={{ lineHeight: 1.8, fontSize: '1.2rem', marginLeft: '20px' }}>
                        <li><strong>Phase A:</strong> ES update <Latex>{`$\\theta$`}</Latex> (Likelihood Reward)</li>
                        <li><strong>Phase B:</strong> Supervised update <Latex>{`$\\phi$`}</Latex> (Denoising)</li>
                        <li><strong>Phase C:</strong> ES update <Latex>{`$\\phi$`}</Latex> (Likelihood Reward)</li>
                        <li><strong>Phase D:</strong> Supervised update <Latex>{`$\\theta$`}</Latex> (Denoising)</li>
                    </ol>
                </div>
                <div style={{ textAlign: 'center', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img src="/finetune/es/block.jpg" style={{ maxHeight: '50vh', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                </div>
            </div>
        </Slide>,

        // Slide 23: ES-DDMEC Results
        <Slide title="ES-DDMEC Results (5K Eval)">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', height: '100%' }}>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <table style={{ width: '100%', fontSize: '1.4rem', borderCollapse: 'collapse', marginBottom: '20px' }}>
                        <thead style={{ background: colors.primary, color: 'white' }}>
                            <tr>
                                <th style={{ padding: '15px' }}>Direction</th>
                                <th style={{ padding: '15px' }}>FID ↓</th>
                                <th style={{ padding: '15px' }}>cFID ↓</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style={{ borderBottom: '1px solid #ddd' }}>
                                <td style={{ padding: '20px' }}>Phi (Control)</td>
                                <td style={{ padding: '20px' }}>148.15</td>
                                <td style={{ padding: '20px' }}>151.50</td>
                            </tr>
                            <tr style={{ borderBottom: '1px solid #ddd', background: '#f0fdf4' }}>
                                <td style={{ padding: '20px' }}>Theta (Treated)</td>
                                <td style={{ padding: '20px', fontWeight: 'bold' }}>60.36</td>
                                <td style={{ padding: '20px', fontWeight: 'bold' }}>71.39</td>
                            </tr>
                        </tbody>
                    </table>
                    <div style={{ fontSize: '1.2rem', fontStyle: 'italic', background: '#fff7ed', padding: '15px', borderRadius: '8px' }}>
                        ES improves cycle-consistency reward (Theta), but inverse mapping (Phi) remains challenging.
                    </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: colors.secondary }}>Theta Evolution (ES)</div>
                    <video src="/bio/theta/theta_video_100.mp4" autoPlay loop muted playsInline style={{ maxHeight: '70vh', maxWidth: '100%', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                </div>
            </div>
        </Slide>,

        // Slide 24: PPO-DDMEC Fine-tuning
        <Slide title="PPO-DDMEC Fine-tuning" subtitle="Diffusion as Policy Optimization">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', alignItems: 'center' }}>
                <div>
                    <p>PPO treats denoising as a trajectory:</p>
                    <ul>
                        <li>Reward from partner likelihood proxy</li>
                        <li>With trust region to anchor / reference</li>
                    </ul>
                    <div style={{ textAlign: 'center', margin: '20px 0', padding: '15px', background: '#f8fafc', borderRadius: '12px' }}>
                        <Latex>{`$\\mathcal{L} = \\mathbb{E}\\left[\\min(r_tA_t,\\text{clip}A_t)\\right] - \\beta\\,\\text{KL}$`}</Latex>
                    </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <img src="/finetune/ppo/block.jpg" style={{ maxHeight: '50vh', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px' }} />
                </div>
            </div>
        </Slide>,

        // Slide 25: PPO-DDMEC Results
        <Slide title="PPO-DDMEC Results">
            <table style={{ width: '100%', fontSize: '1.4rem', borderCollapse: 'collapse' }}>
                <thead style={{ background: colors.primary, color: 'white' }}>
                    <tr>
                        <th style={{ padding: '15px' }}>Direction</th>
                        <th style={{ padding: '15px' }}>FID ↓</th>
                        <th style={{ padding: '15px' }}>cFID ↓</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style={{ borderBottom: '1px solid #ddd' }}>
                        <td style={{ padding: '20px' }}>Phi (Control)</td>
                        <td style={{ padding: '20px' }}>200.12</td>
                        <td style={{ padding: '20px' }}>251.50</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #ddd' }}>
                        <td style={{ padding: '20px' }}>Theta (Treated)</td>
                        <td style={{ padding: '20px' }}>80.82</td>
                        <td style={{ padding: '20px' }}>96.39</td>
                    </tr>
                </tbody>
            </table>
            <div style={{ marginTop: '30px', fontSize: '1.2rem', fontStyle: 'italic', color: colors.danger }}>
                Takeaway: PPO shows more drift and worse FID than ES here, despite stable per-step updates.
            </div>
        </Slide>,

        // Slide 28: Unconditional MEC Architecture
        <Slide title="Unconditional MEC Architecture" subtitle="Independent Experts">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', height: '100%' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <h3>Dual Independent UNets</h3>
                    <ul style={{ lineHeight: 1.6 }}>
                        <li><strong>Control Model:</strong> <Latex>{`$p(x|\\text{DMSO})$`}</Latex>.</li>
                        <li><strong>Perturbed Model:</strong> <Latex>{`$p(x|\\text{Drug})$`}</Latex>.</li>
                        <li><strong>Weights:</strong> CIFAR-10 initialized.</li>
                    </ul>
                    <img src="/uncond/finetune/block diaghram.jpg" style={{ maxHeight: '40vh', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px', border: '1px solid #ccc' }} />
                </div>
                <div style={{ background: '#eff6ff', padding: '20px', borderRadius: '12px', border: '1px solid #dbeafe', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <h3 style={{ marginTop: 0, color: colors.primary }}>Why split?</h3>
                    <p>Ensures marginals are learned perfectly <em>before</em> any coupling is attempted.</p>
                    <div style={{ marginTop: '20px', fontWeight: 'bold', textAlign: 'center', color: colors.theta, background: 'white', padding: '10px', borderRadius: '8px' }}>
                        No shared weights = No leakage.
                    </div>
                </div>
            </div>
        </Slide>,

        // Slide 29: Unconditional DDPM Pretraining
        <Slide title="Unconditional DDPM Pretraining">
            <p>Train two independent unconditional DDPMs:</p>
            <ul>
                <li><strong>Control DDPM</strong> (DMSO Only)</li>
                <li><strong>Perturbed DDPM</strong> (All Treated)</li>
            </ul>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
                <img src="/uncond/pretrain/controls_samples.png" style={{ width: '100%', maxHeight: '40vh', objectFit: 'contain' }} />
                <img src="/uncond/pretrain/perturbed_samples.png" style={{ width: '100%', maxHeight: '40vh', objectFit: 'contain' }} />
            </div>
        </Slide>,

        // Slide 27: Uncond + PPO-DDMEC Results
        <Slide title="Uncond + PPO-DDMEC Results">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
                <div>
                    <img src="/uncond/finetune/plots ddmec uncond ddpms.png" style={{ width: '100%', maxHeight: '78vh', objectFit: 'contain' }} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <ul>
                        <li>Rapid FID drop early (~3k iters).</li>
                        <li>Phi stabilizes lower than Theta.</li>
                        <li>Low KL (<Latex>{`0.01 - 0.04`}</Latex>) indicates conservative updates.</li>
                    </ul>
                </div>
            </div>
        </Slide>,

        // Slide 28: Flux Matching
        <Slide title="Flux Matching" subtitle="Flux.1-Dev + LoRA + ControlNet">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', height: '100%', alignItems: 'center' }}>
                <div>
                    <p>Flow matching models probability paths directly.</p>
                    <ul style={{ marginBottom: '20px' }}>
                        <li>Frozen Flux backbone (12B Params)</li>
                        <li>Trainable ControlNet</li>
                        <li>LoRA Adapters</li>
                    </ul>
                    <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                        <h2 style={{ color: colors.success, margin: '10px 0' }}>FID: 77.49 | cFID: 152.79</h2>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <img src="/flux/block.png" style={{ maxHeight: '45vh', maxWidth: '100%', objectFit: 'contain' }} />
                    </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: colors.success }}>Generation Process</div>
                    <video src="/flux/video_step_4000.mp4" autoPlay loop muted playsInline style={{ maxHeight: '70vh', maxWidth: '100%', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                </div>
            </div>
        </Slide>,

        // Slide 29: SD + LoRA + ControlNet
        <Slide title="Stable Diffusion + LoRA" subtitle="Gold Standard">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', height: '100%', alignItems: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: colors.primary }}>Evaluation Samples</div>
                    <video src="/stable/video_eval_latest.mp4" autoPlay loop muted playsInline style={{ maxHeight: '70vh', maxWidth: '100%', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                </div>
                <div>
                    <p>Why it helps:</p>
                    <ul>
                        <li>Frozen SD backbone preserves strong prior.</li>
                        <li>LoRA learns microscopy "style".</li>
                        <li><strong>Drug Projector:</strong> Fingerprint → 4 Tokens.</li>
                    </ul>
                    <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                        <h2 style={{ color: colors.primary, margin: '10px 0' }}>FID: 66.49 | cFID: 132.79</h2>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                        <img src="/stable/block.png" style={{ maxHeight: '45vh', maxWidth: '100%', objectFit: 'contain' }} />
                    </div>
                </div>
            </div>
        </Slide>,

        // Slide 30: Final Comparison
        <Slide title="Final Comparison & Conclusions">
            <table style={{ width: '100%', fontSize: '1.2rem', borderCollapse: 'collapse' }}>
                <thead style={{ background: colors.text, color: 'white' }}>
                    <tr>
                        <th style={{ padding: '10px' }}>Method</th>
                        <th style={{ padding: '10px' }}>FID ↓</th>
                        <th style={{ padding: '10px' }}>cFID ↓</th>
                        <th style={{ padding: '10px' }}>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style={{ background: '#e0f2fe' }}>
                        <td style={{ padding: '10px' }}>Pretrained Cond DDPM</td>
                        <td style={{ padding: '10px' }}>11.94</td>
                        <td style={{ padding: '10px' }}>75.2</td>
                        <td style={{ padding: '10px' }}>Best FID, strong baseline</td>
                    </tr>
                    <tr>
                        <td style={{ padding: '10px' }}>ES Fine-tune (Theta)</td>
                        <td style={{ padding: '10px' }}>60.36</td>
                        <td style={{ padding: '10px' }}>71.39</td>
                        <td style={{ padding: '10px' }}>Better than PPO</td>
                    </tr>
                    <tr>
                        <td style={{ padding: '10px' }}>PPO Fine-tune (Theta)</td>
                        <td style={{ padding: '10px' }}>80.82</td>
                        <td style={{ padding: '10px' }}>96.39</td>
                        <td style={{ padding: '10px' }}>Worse FID</td>
                    </tr>
                    <tr>
                        <td style={{ padding: '10px' }}>Flux Matching</td>
                        <td style={{ padding: '10px' }}>77.49</td>
                        <td style={{ padding: '10px' }}>152.79</td>
                        <td style={{ padding: '10px' }}>Strong prior, flow objective</td>
                    </tr>
                    <tr style={{ background: '#f0fdf4' }}>
                        <td style={{ padding: '10px' }}>SD + LoRA + ControlNet</td>
                        <td style={{ padding: '10px' }}>66.49</td>
                        <td style={{ padding: '10px' }}>132.79</td>
                        <td style={{ padding: '10px' }}>Pragmatic stability boost</td>
                    </tr>
                </tbody>
            </table>
            <div style={{ marginTop: '30px', padding: '20px', background: '#fff7ed', borderRadius: '12px' }}>
                <strong>Key Takeaway:</strong> Synthetic ablation suggests ES better preserves priors in high-dims. In Biology, Supervised Conditional Diffusion is strongest, but <strong>ES &gt; PPO</strong> for RL fine-tuning.
            </div>
        </Slide>
    ];

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'ArrowRight' || e.key === 'Space') {
                setCurrentSlide(prev => Math.min(prev + 1, slides.length - 1));
            } else if (e.key === 'ArrowLeft') {
                setCurrentSlide(prev => Math.max(prev - 1, 0));
            } else if (e.key === 'Escape') {
                onExit();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [slides.length, onExit]);

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: colors.slideBg,
            zIndex: 9999,
            color: colors.text,
            fontFamily: "'Inter', sans-serif"
        }}>
            {/* Progress Bar */}
            <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                height: '8px',
                background: colors.secondary,
                width: `${((currentSlide + 1) / slides.length) * 100}%`,
                transition: 'width 0.3s ease',
                zIndex: 10001
            }} />

            {/* Slide Content */}
            {slides[currentSlide]}

            {/* Controls */}
            <div style={{
                position: 'fixed',
                bottom: '30px',
                right: '40px',
                display: 'flex',
                gap: '16px',
                alignItems: 'center',
                zIndex: 10000
            }}>
                <span style={{ fontSize: '1.2rem', color: colors.textLight, userSelect: 'none', marginRight: '20px' }}>
                    {currentSlide + 1} / {slides.length}
                </span>
                <button
                    onClick={onExit}
                    style={{
                        padding: '10px 20px',
                        background: 'white',
                        border: `2px solid ${colors.textLight}`,
                        borderRadius: '30px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        color: colors.textLight,
                        boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
                    }}
                >
                    Exit
                </button>
            </div>

            {/* Navigation Hints */}
            <div style={{
                position: 'fixed',
                bottom: '30px',
                left: '40px',
                color: colors.textLight,
                opacity: 0.6,
                fontSize: '0.9rem'
            }}>
                Use Arrow Keys to Navigate
            </div>
        </div>
    );
};

export default PresentationMode;
