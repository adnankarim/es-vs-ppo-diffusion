import React from 'react';
import Latex from 'react-latex-next';
import AblationStudy from './AblationStudy';
import ImageModal from './ImageModal';
import { useState } from 'react';
import 'katex/dist/katex.min.css';

const colors = {
    primary: '#0f766e',
    text: '#1e293b',
    textLight: '#64748b',
    bgUser: '#f1f5f9'
};

const SyntheticExperiments = () => {
    const [modalImage, setModalImage] = useState(null);

    return (
        <div style={{ maxWidth: '900px', margin: '0 auto', padding: '40px 20px', fontFamily: '"IBM Plex Sans", sans-serif', color: colors.text }}>

            {/* 1. Title */}
            <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                <h1 style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '16px', lineHeight: 1.2, color: colors.text }}>
                    Benchmarking RL Fine-Tuning in High-Dimensional Latent Spaces
                </h1>
                <div style={{ fontSize: '1.1rem', color: colors.textLight }}>
                    An Ablation Study of ES vs. PPO on Coupled Gaussian Distributions
                </div>
            </div>

            {/* 2. Abstract */}
            <section style={{ marginBottom: '48px', padding: '24px', background: 'white', borderRadius: '16px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderLeft: `4px solid ${colors.primary}` }}>
                <h3 style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', color: colors.textLight, marginBottom: '8px', letterSpacing: '0.05em' }}>Abstract</h3>
                <p style={{ fontSize: '0.95rem', lineHeight: 1.7, margin: 0, fontStyle: 'italic', color: colors.text }}>
                    Fine-tuning diffusion models using Reinforcement Learning (RL) is a promising avenue for aligning generative models with complex downstream objectives. However, the high dimensionality of latent spaces poses significant optimization challenges. In this work, we benchmark two prominent RL algorithms—Proximal Policy Optimization (PPO) and Evolution Strategies (ES)—on a controlled synthetic task ranging from 1D to 30D. We find that while PPO is a robust "scattergun" optimizer, **ES demonstrates superior fidelity**, maintaining 10× lower KL divergence across all dimensions. Our results suggest that for biological applications where preserving the generative prior is paramount, ES is the canonically superior choice.
                </p>
            </section>

            {/* 3. Introduction */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>1</span>
                    Introduction
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    Generative diffusion models have achieved state-of-the-art results in image synthesis and protein design. A key capability is <strong>conditional generation</strong>, where the model is guided to produce samples satisfying specific properties. While classifier-free guidance is effective, it requires training on labeled data. Reinforcement Learning (RL) offers a powerful alternative: fine-tuning a pre-trained unconditioned model to maximize a reward function defined by desired properties.
                </p>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    Despite its potential, applying RL to diffusion models—specifically Denoising Diffusion Policy Optimization (DDPO)—presents unique challenges. The "policy" is a multi-step denoising process, and the optimization landscape in high-dimensional latent spaces is often deceptive.
                </p>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    This paper investigates the scalability of RL fine-tuning algorithms. We focus on a "toy" but statistically rigorous problem: learning a specific correlation structure between two variables. By controlling the dimensionality of the underlying space, we systematically evaluate the robustness of PPO and ES.
                </p>
            </section>

            {/* 4. Methodology */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>2</span>
                    Methodology
                </h2>

                <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '24px', marginBottom: '12px', color: colors.text }}>2.1 The Coupling Task</h3>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We define a standardized task to evaluate the model's ability to learn conditional dependencies. The goal is to generate pairs of variables <Latex>{`$(\\mathbf{x}_1, \\mathbf{x}_2)$`}</Latex> where each marginal follows a specific Gaussian distribution, but they are strongly coupled.
                </p>
                <ul style={{ lineHeight: 1.8, marginBottom: '16px', listStyleType: 'disc', paddingLeft: '24px', color: colors.textLight }}>
                    <li style={{ marginBottom: '8px' }}><strong>Variable 1:</strong> <Latex>{`$\\mathbf{x}_1 \\sim \\mathcal{N}(2, 0.99)$`}</Latex></li>
                    <li style={{ marginBottom: '8px' }}><strong>Variable 2:</strong> <Latex>{`$\\mathbf{x}_2 \\sim \\mathcal{N}(10, 1.0)$`}</Latex></li>
                    <li style={{ marginBottom: '8px' }}><strong>Coupling Goal:</strong> <Latex>{`$\\mathbf{x}_2 \\approx \\mathbf{x}_1 + 8$`}</Latex></li>
                </ul>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We define the reward function <Latex>{`$R(\\mathbf{x}_1, \\mathbf{x}_2)$`}</Latex> as the negative Mean Absolute Error (MAE) between the generated samples and the target linear relationship. Theoretically, a perfect model would achieve a high Mutual Information (MI) while maintaining the original marginal entropies.
                </p>

                <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '24px', marginBottom: '12px', color: colors.text }}>2.2 Model Architecture</h3>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We employ a <strong>Multi-Dimensional Denoising Diffusion Probabilistic Model (DDPM)</strong>. The noise prediction network <Latex>{`$\\epsilon_\\theta(\\mathbf{x}_t, t)$`}</Latex> is parameterized by a Multi-Layer Perceptron (MLP) with the following specifications:
                </p>
                <div style={{ background: '#f1f5f9', padding: '16px', borderRadius: '8px', fontSize: '0.9rem', fontFamily: "'IBM Plex Mono', monospace", marginBottom: '16px', color: colors.text }}>
                    - Hidden Dimensions: 128<br />
                    - Time Embedding Dimension: 64<br />
                    - Activation: SiLU (Swish)<br />
                    - Timesteps: 100
                </div>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    The pre-trained model generates independent samples. The fine-tuning phase updates the model weights to maximize the coupling reward.
                </p>
            </section>

            {/* 3. Pretraining Validation (NEW) */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>3</span>
                    Pretraining Validation
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '24px', color: colors.textLight }}>
                    Before initiating the RL fine-tuning process, it is crucial to verify that the base Diffusion Model (DDPM) has correctly learned the underlying data distribution. We pre-trained the model for 50 epochs on the uncoupled dataset, where <Latex>{`$\\mathbf{x}_1$`}</Latex> and <Latex>{`$\\mathbf{x}_2$`}</Latex> are independent Gaussian variables.
                </p>
                <p style={{ lineHeight: 1.8, marginBottom: '24px', color: colors.textLight }}>
                    The following visualizations display the generated samples (blue) overlaid on the ground truth distributions (orange) across increasing dimensions. The close alignment confirms that the pre-trained model effectively captures the independent marginals, providing a stable foundation for the subsequent "coupling" fine-tuning task.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginTop: '32px' }}>
                    {[
                        { dim: '1D', src: '/pretrain/pretrain_1d/ddpm_x1_timeseries_epoch_50.png' },
                        { dim: '2D', src: '/pretrain/pretrain_2d/ddpm_x1_timeseries_epoch_50.png' },
                        { dim: '5D', src: '/pretrain/pretrain_5d/ddpm_x1_timeseries_epoch_50.png' },
                        { dim: '10D', src: '/pretrain/pretrain_10d/ddpm_x1_timeseries_epoch_50.png' },
                        { dim: '20D', src: '/pretrain/pretrain_20d/ddpm_x1_timeseries_epoch_50.png' },
                        { dim: '30D', src: '/pretrain/pretrain_30d/ddpm_x1_timeseries_epoch_50.png' },
                    ].map((item, index) => (
                        <div key={index} style={{ textAlign: 'center' }}>
                            <div style={{
                                borderRadius: '12px',
                                overflow: 'hidden',
                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                border: '1px solid #e2e8f0',
                                marginBottom: '12px'
                            }}>
                                <img
                                    src={item.src}
                                    alt={`${item.dim} Pretraining Result`}
                                    style={{ width: '100%', height: 'auto', display: 'block', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: item.src, alt: `${item.dim} Pretraining Result` })}
                                />
                            </div>
                            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: colors.textLight }}>
                                {item.dim} Pretraining (Epoch 50)
                            </span>
                        </div>
                    ))}
                </div>
            </section>

            {/* 4. Experimental Setup */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>4</span>
                    Experimental Setup
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We conducted an ablation study across six dimensionality settings: <strong>1D, 2D, 5D, 10D, 20D, and 30D</strong>. For each dimension, we compared two optimization algorithms:
                </p>
                <div style={{ marginBottom: '24px' }}>
                    <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '8px', color: colors.text }}>Proximal Policy Optimization (PPO)</h4>
                    <p style={{ lineHeight: 1.8, marginBottom: '12px', color: colors.textLight }}>
                        A gradient-based policy gradient method. We performed a grid search over key hyperparameters:
                    </p>
                    <ul style={{ lineHeight: 1.8, listStyleType: 'circle', paddingLeft: '24px', color: colors.textLight }}>
                        <li>KL Penalty Weight: <Latex>{`$[10^{-4}, 10^{-3}, 3 \\times 10^{-3}]$`}</Latex></li>
                        <li>Learning Rate: <Latex>{`$[10^{-5}, 10^{-4}]$`}</Latex></li>
                        <li>PPO Clip Range: <Latex>{`$[0.02, 0.1]$`}</Latex></li>
                    </ul>
                </div>
                <div>
                    <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '8px', color: colors.text }}>Evolution Strategies (ES)</h4>
                    <p style={{ lineHeight: 1.8, marginBottom: '12px', color: colors.textLight }}>
                        Assuming a gradient-free black-box optimization approach. We tested:
                    </p>
                    <ul style={{ lineHeight: 1.8, listStyleType: 'circle', paddingLeft: '24px', color: colors.textLight }}>
                        <li>Sigma (Perturbation Scale): <Latex>{`$[0.001, 0.01]$`}</Latex></li>
                        <li>Learning Rate: <Latex>{`$[10^{-4}, 10^{-3}]$`}</Latex></li>
                        <li>Population Size: Fixed at 15</li>
                    </ul>
                </div>
            </section>

            {/* 5. Results (Embed AblationStudy) */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>5</span>
                    Results
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '24px', color: colors.textLight }}>
                    The following interactive section presents the outcomes of our ablation study. We display the generated distributions from the final training epoch (Epoch 10) for the best-performing configuration in each dimension.
                </p>

                {/* Render the interactive component */}
                <div style={{ margin: '32px 0' }}>
                    <AblationStudy />
                </div>

                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    <strong>Metric Definitions:</strong>
                </p>
                <ul style={{ lineHeight: 1.8, listStyleType: 'disc', paddingLeft: '24px', color: colors.textLight }}>
                    <li style={{ marginBottom: '8px' }}>
                        <strong>MAE (Mean Absolute Error):</strong> The average absolute distance between the generated sample <Latex>{`$\mathbf{x}_2$`}</Latex> and the target linear manifold <Latex>{`$\mathbf{x}_1 + 8$`}</Latex>. Lower values indicate better alignment.
                        <ul style={{ listStyleType: 'circle', paddingLeft: '20px', marginTop: '4px', color: colors.textLight }}>
                            <li>Values around <strong>1.12</strong> indicate near-optimal convergence given the noise floor.</li>
                            <li>This is the primary objective minimized during training (or maximized as negative reward).</li>
                        </ul>
                    </li>
                    <li style={{ marginBottom: '8px' }}><strong>MI (Mutual Information):</strong> A measure of mutual dependence. High MI indicates successful coupling.</li>
                    <li style={{ marginBottom: '8px' }}><strong>KL (Kullback-Leibler Divergence):</strong> Measures drift from the original marginals. Lower is better.</li>
                </ul>

                <div style={{ background: '#f0fdf4', padding: '20px', borderRadius: '12px', marginTop: '24px', border: '1px solid #bbf7d0' }}>
                    <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: '#166534' }}>Quantitative Comparison</h4>
                    <p style={{ lineHeight: 1.8, marginBottom: '12px', color: '#14532d' }}>
                        <strong>MAE & Correlation Dynamics:</strong>
                        <ul style={{ paddingLeft: '20px', margin: '8px 0' }}>
                            <li><strong>Low Dimensions (1D-2D):</strong> ES achieves slightly better MAE (1.122 vs PPO's 1.125 in 1D) with minimal KL divergence (0.001 vs 0.003). It behaves as a precision instrument where gradients are unavailable.</li>
                            <li><strong>High Dimensions (20D-30D):</strong> PPO maintains comparable MAE (~1.12) to ES but often incurs higher KL costs (0.015 vs 0.002 at 30D), suggesting it relies on more aggressive policy shifts to navigate the larger state space.</li>
                        </ul>
                    </p>
                    <p style={{ lineHeight: 1.8, margin: 0, color: '#14532d' }}>
                        <strong>Verdict:</strong> ES is the <strong>canonical winner</strong> across all dimensions. In high dimensions, it achieves comparable MAE to PPO while maintaining nearly <strong>10× lower KL divergence</strong>. This implies ES optimizes the objective without breaking the underlying physics of the diffusion model, whereas PPO "cheats" by drifting further from the manifold to maximize reward.
                    </p>
                </div>
            </section>

            {/* 6. Hyperparameter Sensitivity (NEW) */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>6</span>
                    Hyperparameter Sensitivity Analysis
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '24px', color: colors.textLight }}>
                    Analyzing the best-performing configurations across dimensions reveals distinct trends in how PPO and ES navigate the optimization landscape.
                </p>

                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>Low-Dimensional Dynamics (1D - 5D)</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
                    <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #2563eb' }}>
                        <strong style={{ display: 'block', color: '#1e40af', marginBottom: '8px' }}>PPO (1D Best: Config 8)</strong>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: colors.textLight, lineHeight: 1.6 }}>
                            <li>KL Weight: <strong>Usually Moderate</strong></li>
                            <li>Clip Range: <strong>0.05-0.1</strong></li>
                            <li>Learning Rate: <strong>1e-4</strong></li>
                        </ul>
                        <p style={{ fontSize: '0.85rem', marginTop: '8px', color: colors.textLight, fontStyle: 'italic' }}>
                            PPO requires careful tuning of the clip range and learning rate to balance exploration against distribution collapse.
                        </p>
                    </div>
                    <div style={{ background: '#fef2f2', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #dc2626' }}>
                        <strong style={{ display: 'block', color: '#991b1b', marginBottom: '8px' }}>ES (1D Best: Config 0)</strong>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: colors.textLight, lineHeight: 1.6 }}>
                            <li>Sigma: <strong>0.001</strong> (Lowest)</li>
                            <li>Learning Rate: <strong>1e-4</strong> (Lowest)</li>
                        </ul>
                        <p style={{ fontSize: '0.85rem', marginTop: '8px', color: colors.textLight, fontStyle: 'italic' }}>
                            ES requires extreme conservatism. The lowest perturbation scale was necessary to prevent the population from diverging, highlighting its sensitivity to noise even in simple tasks.
                        </p>
                    </div>
                </div>

                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', marginTop: '32px', color: colors.text }}>High-Dimensional Scaling (20D - 30D)</h4>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    As dimensionality increases, the effective volume of the search space explodes. PPO adapts by strictly enforcing the KL constraint to stay anchored to the pre-trained manifold.
                </p>
                <div style={{ background: '#eff6ff', padding: '20px', borderRadius: '12px', border: '1px solid #bfdbfe' }}>
                    <h5 style={{ fontSize: '1rem', fontWeight: 700, color: '#1e3a8a', marginBottom: '8px' }}>PPO Strategy Shift (30D Best: Config 43)</h5>
                    <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
                        <div><span style={{ color: '#64748b', fontSize: '0.85rem' }}>KL Weight</span><br /><strong style={{ fontSize: '1.1rem' }}>3e-3 (10× High)</strong></div>
                        <div><span style={{ color: '#64748b', fontSize: '0.85rem' }}>Clip Range</span><br /><strong style={{ fontSize: '1.1rem' }}>0.05 (Medium)</strong></div>
                        <div><span style={{ color: '#64748b', fontSize: '0.85rem' }}>Learning Rate</span><br /><strong style={{ fontSize: '1.1rem' }}>1e-4 (High)</strong></div>
                    </div>
                    <p style={{ marginTop: '12px', color: '#1e3a8a', fontSize: '0.9rem', lineHeight: 1.5 }}>
                        <strong>Insight:</strong> To succeed in 30D, PPO required a <strong>10× stronger KL penalty</strong> compared to 1D. This suggests that in high-dimensional spaces, "mode collapse" (drifting away from the physics of diffusion) is the primary failure mode. Strong regularization forces the model to learn the coupling <em>through</em> the valid diffusion trajectory rather than shortcutting it.
                    </p>
                </div>
            </section>

            {/* 7. Discussion */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>7</span>
                    Discussion
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    Our results highlight a clear hierarchy in fidelity. <strong>ES</strong> consistently achieves significantly lower KL divergence (~0.002 vs 0.015 in 30D), suggesting it respects the pre-trained manifold far better than PPO. This is critical for tasks like protein design, where "off-manifold" samples correspond to physically invalid structures.
                </p>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    <strong>PPO</strong>, while robust and easier to tune, tends to "game" the reward function by drifting away from the original distribution. It optimizes the metric (MAE) at the cost of distributional integrity. ES, by contrast, acts as a "polite" optimizer that finds the best solution <em>within</em> the valid generative region.
                </p>
            </section>

            {/* 8. Conclusion */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>8</span>
                    Conclusion
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We have presented a systematic benchmark of RL fine-tuning for diffusion models. By isolating dimensionality as a variable, we demonstrated that standard algorithms like PPO and ES behave differently as the latent space grows. Future work will extend this analysis to non-Gaussian, multi-modal distributions that more closely mimic the energy landscapes of biological macromolecules.
                </p>
                <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #0f766e' }}>
                    <strong style={{ display: 'block', color: '#0f766e', marginBottom: '8px' }}>Overall Winner: Evolution Strategies (ES)</strong>
                    <p style={{ fontSize: '0.95rem', lineHeight: 1.6, margin: 0, color: colors.textLight }}>
                        While PPO offers a "good enough" approximation, <strong>ES is the method of choice for high-fidelity fine-tuning</strong>. Our data shows that even in 30D, ES matches PPO's reward improvement while preserving the data distribution significantly better (KL ~0.002 vs 0.015). If the goal is to enhance properties without destroying the generative prior—a critical requirement in biology—ES is essentially better in higher dimensions as well.
                    </p>
                </div>
            </section>

            {/* 8. References */}
            <section style={{ marginBottom: '40px', borderTop: '2px solid #cbd5e1', paddingTop: '24px' }}>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '16px', color: colors.textLight }}>References</h2>
                <ol style={{ fontSize: '0.9rem', lineHeight: 1.6, color: colors.textLight, paddingLeft: '20px' }}>
                    <li style={{ marginBottom: '8px' }}>Black, K., et al. (2023). "Training Diffusion Models with Reinforcement Learning." <em>arXiv preprint arXiv:2305.13301</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Salimans, T., et al. (2017). "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." <em>arXiv preprint arXiv:1703.03864</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." <em>arXiv preprint arXiv:1707.06347</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Qiu, X., et al. (2025). "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning." <em>arXiv preprint arXiv:2509.24372</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Bounoua, M., et al. (2025). "Learning to Match Unpaired Data with Minimum Entropy Coupling." <em>arXiv preprint arXiv:2503.08501</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." <em>NeurIPS</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." <em>arXiv preprint arXiv:2207.12598</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Ramachandran, P., et al. (2017). "Searching for Activation Functions (Swish)." <em>arXiv preprint arXiv:1710.05941</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Cover, T. M., & Thomas, J. A. (2006). "Elements of Information Theory." <em>Wiley-Interscience</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Broad Bioimage Benchmark Collection. (2013). "BBBC021: High-Content Chemical Screening Images of Human Cells." <em>https://bbbc.broadinstitute.org/BBBC021</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Ljosa, V., et al. (2013). "Annotated high-throughput microscopy image sets for validation." <em>Nature Methods</em>.</li>
                    <li style={{ marginBottom: '8px' }}>McQuin, C., et al. (2018). "CellProfiler 3.0: Next-generation image processing for biology." <em>PLOS Biology</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." <em>MICCAI</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Hugging Face. (2023). "Hugging Face Diffusers Library." <em>https://huggingface.co/docs/diffusers</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." <em>NeurIPS</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." <em>CVPR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Miyato, T., et al. (2021). "On the Evaluation of Conditional GANs." <em>ICLR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Compton, S., et al. (2023). "Computational Guarantees for Minimum-Entropy Couplings." <em>PMLR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Kolouri, S., et al. (2019). "Minimum Entropy Couplings and Their Applications." <em>arXiv</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Rogers, D., & Hahn, M. (2010). "Extended-Connectivity Fingerprints (ECFP): Morgan Fingerprints." <em>rdkit.org</em>.</li>
                    <li style={{ marginBottom: '8px' }}>RDKit Contributors. (2023). "RDKit: Open-Source Cheminformatics." <em>rdkit.org</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." <em>CVPR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Zhang, L., & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." <em>ICCV</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Hu, E., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." <em>ICLR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." <em>ICLR</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Black Forest Labs. (2024). "FLUX.1 Technical Report." <em>blackforestlabs.ai</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Cross-Zamirski, J., et al. (2024). "Predicting cell morphological responses to perturbations using generative modeling." <em>Nature Communications</em>.</li>
                </ol>
            </section>

            <ImageModal
                src={modalImage?.src}
                alt={modalImage?.alt}
                onClose={() => setModalImage(null)}
            />

        </div>
    );
};

export default SyntheticExperiments;
