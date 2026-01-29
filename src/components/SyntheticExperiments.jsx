import React from 'react';
import Latex from 'react-latex-next';
import AblationStudy from './AblationStudy';
import 'katex/dist/katex.min.css';

const colors = {
    primary: '#0f766e',
    text: '#1e293b',
    textLight: '#64748b',
    bgUser: '#f1f5f9'
};

const SyntheticExperiments = () => {
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
                    Fine-tuning diffusion models using Reinforcement Learning (RL) is a promising avenue for aligning generative models with complex downstream objectives. However, the high dimensionality of latent spaces poses significant optimization challenges. In this work, we benchmark two prominent RL algorithms—Proximal Policy Optimization (PPO) and Evolution Strategies (ES)—on a controlled synthetic task: coupling independent Gaussian marginals across dimensions ranging from 1D to 30D. We find that while both methods perform comparably in lower dimensions, their stability and sample efficiency diverge as dimensionality increases. PPO demonstrates superior consistency in maintaining distribution metrics (KL divergence), whereas ES exhibits higher variance but competitive reward maximization in specific high-dimensional settings. Our results provide a foundational baseline for applying RL fine-tuning to large-scale diffusion models in biology and chemistry.
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

            {/* 3. Experimental Setup */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>3</span>
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

            {/* 4. Results (Embed AblationStudy) */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>4</span>
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
                    <li style={{ marginBottom: '8px' }}><strong>Score:</strong> The mean reward obtained (higher is better). Reflects alignment with the <Latex>{`$\\mathbf{x}_2 = \\mathbf{x}_1 + 8$`}</Latex> target.</li>
                    <li style={{ marginBottom: '8px' }}><strong>MI (Mutual Information):</strong> A measure of mutual dependence between the two variables. High MI indicates successful coupling.</li>
                    <li style={{ marginBottom: '8px' }}><strong>KL (Kullback-Leibler Divergence):</strong> Measures deviation from the original Gaussian marginals. A lower KL indicates the model preserved the original distribution structure while learning the coupling.</li>
                </ul>
            </section>

            {/* 5. Hyperparameter Sensitivity (NEW) */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>5</span>
                    Hyperparameter Sensitivity Analysis
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '24px', color: colors.textLight }}>
                    Analyzing the best-performing configurations across dimensions reveals distinct trends in how PPO and ES navigate the optimization landscape.
                </p>

                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>Low-Dimensional Dynamics (1D - 5D)</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
                    <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', borderLeft: '4px solid #2563eb' }}>
                        <strong style={{ display: 'block', color: '#1e40af', marginBottom: '8px' }}>PPO (1D Best: Config 23)</strong>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.9rem', color: colors.textLight, lineHeight: 1.6 }}>
                            <li>KL Weight: <strong>3e-4</strong> (Low-Mid)</li>
                            <li>Clip Range: <strong>0.1</strong> (High)</li>
                            <li>Learning Rate: <strong>1e-4</strong> (High)</li>
                        </ul>
                        <p style={{ fontSize: '0.85rem', marginTop: '8px', color: colors.textLight, fontStyle: 'italic' }}>
                            In low dimensions, PPO benefits from aggressive exploration (High Clip/LR) and lower regularization constraint, allowing it to quickly adapt to the target distribution.
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

            {/* 6. Discussion */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>6</span>
                    Discussion
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    Our results highlight a trade-off between stability and exploration. <strong>PPO</strong> consistently achieves lower KL divergence, suggesting it stays closer to the pre-trained manifold. This is advantageous for tasks like protein design where deviating from "naturalness" yields invalid structures.
                </p>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    <strong>ES</strong>, while showing higher variance in outcomes, proved competitive in specific high-dimensional settings when carefully tuned. Its gradient-free nature allows it to traverse the optimization landscape without relying on the specific curvature information that might mislead gradient-based methods in sparse reward environments. However, our sensitivity analysis confirms it is significantly more brittle to hyperparameter choices (sigma) than PPO.
                </p>
            </section>

            {/* 7. Conclusion */}
            <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
                    <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>7</span>
                    Conclusion
                </h2>
                <p style={{ lineHeight: 1.8, marginBottom: '16px', color: colors.textLight }}>
                    We have presented a systematic benchmark of RL fine-tuning for diffusion models. By isolating dimensionality as a variable, we demonstrated that standard algorithms like PPO and ES behave differently as the latent space grows. Future work will extend this analysis to non-Gaussian, multi-modal distributions that more closely mimic the energy landscapes of biological macromolecules.
                </p>
            </section>

            {/* 8. References */}
            <section style={{ marginBottom: '40px', borderTop: '2px solid #cbd5e1', paddingTop: '24px' }}>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '16px', color: colors.textLight }}>References</h2>
                <ol style={{ fontSize: '0.9rem', lineHeight: 1.6, color: colors.textLight, paddingLeft: '20px' }}>
                    <li style={{ marginBottom: '8px' }}>Black, K., et al. (2023). "Training Diffusion Models with Reinforcement Learning." <em>arXiv preprint arXiv:2305.13301</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Salimans, T., et al. (2017). "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." <em>arXiv preprint arXiv:1703.03864</em>.</li>
                    <li style={{ marginBottom: '8px' }}>Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." <em>arXiv preprint arXiv:1707.06347</em>.</li>
                </ol>
            </section>

        </div>
    );
};

export default SyntheticExperiments;
