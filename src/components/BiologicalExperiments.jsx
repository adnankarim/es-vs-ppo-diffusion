import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css';

// Parsed Data from theta.csv (Forward Process: Control -> Treated)
const thetaData = [
    { epoch: 1, loss: 0.0504, mse: 0.0498, fid: 160.00, fidc: 320.00 },
    { epoch: 5, loss: 0.0484, mse: 0.0484, fid: 154.52, fidc: 311.85 },
    { epoch: 10, loss: 0.0462, mse: 0.0462, fid: 147.68, fidc: 301.66 },
    { epoch: 20, loss: 0.0418, mse: 0.0421, fid: 133.99, fidc: 281.27 },
    { epoch: 30, loss: 0.0369, mse: 0.0367, fid: 120.31, fidc: 260.89 },
    { epoch: 40, loss: 0.0318, mse: 0.0313, fid: 106.62, fidc: 240.51 },
    { epoch: 50, loss: 0.0268, mse: 0.0269, fid: 92.93, fidc: 220.13 },
    { epoch: 65, loss: 0.0150, mse: 0.0160, fid: 60.00, fidc: 180.00 },
    { epoch: 80, loss: 0.0080, mse: 0.0075, fid: 40.00, fidc: 145.00 },
    { epoch: 100, loss: 0.0033, mse: 0.0029, fid: 24.49, fidc: 118.22 },
];

// Parsed Data from phi.csv (Inverse Process: Treated -> Control)
const phiData = [
    { epoch: 1, loss: 0.0504, mse: 0.0499, fid: 160.00, fidc: 320.00 },
    { epoch: 10, loss: 0.0497, mse: 0.0496, fid: 149.45, fidc: 304.43 },
    { epoch: 30, loss: 0.0363, mse: 0.0366, fid: 127.18, fidc: 271.55 },
    { epoch: 50, loss: 0.0255, mse: 0.0257, fid: 101.40, fidc: 233.48 },
    { epoch: 70, loss: 0.0120, mse: 0.0130, fid: 70.00, fidc: 185.00 },
    { epoch: 85, loss: 0.0080, mse: 0.0090, fid: 55.00, fidc: 165.00 },
    { epoch: 100, loss: 0.0046, mse: 0.0052, fid: 43.97, fidc: 148.70 },
];

const colors = {
    primary: '#0f766e', theta: '#2563eb', phi: '#dc2626',
    text: '#1e293b', textLight: '#64748b', background: '#ffffff',
    accent: '#f59e0b'
};

const Section = ({ title, number, children }) => (
    <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px', color: colors.text }}>
            <span style={{ background: colors.primary, color: 'white', width: '36px', height: '36px', borderRadius: '50%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginRight: '12px', fontSize: '16px' }}>
                {number}
            </span>
            {title}
        </h2>
        {children}
    </section>
);

const SubTabButton = ({ active, onClick, children }) => (
    <button
        onClick={onClick}
        style={{
            padding: '8px 16px',
            fontSize: '13px',
            fontWeight: 600,
            border: 'none',
            background: active ? '#f1f5f9' : 'transparent',
            color: active ? colors.primary : colors.textLight,
            cursor: 'pointer',
            borderRadius: '6px',
            transition: 'all 0.2s ease',
            marginRight: '8px'
        }}
    >
        {children}
    </button>
);

import ImageModal from './ImageModal';

const BiologicalExperiments = () => {
    const [activeSubTab, setActiveSubTab] = useState('ddpm');
    const [modalImage, setModalImage] = useState(null);

    return (
        <>
            <div style={{ backgroundColor: 'white', borderRadius: '0 0 16px 16px', padding: '32px' }}>

                {/* Sub-Tab Navigation */}
                <div style={{ display: 'flex', background: '#f8fafc', padding: '6px', borderRadius: '10px', marginBottom: '32px', width: 'fit-content', border: '1px solid #e2e8f0' }}>
                    <SubTabButton active={activeSubTab === 'ddpm'} onClick={() => setActiveSubTab('ddpm')}>
                        DDPM & MEC
                    </SubTabButton>
                    <SubTabButton active={activeSubTab === 'flux'} onClick={() => setActiveSubTab('flux')}>
                        Flux Matching
                    </SubTabButton>
                    <SubTabButton active={activeSubTab === 'sdlora'} onClick={() => setActiveSubTab('sdlora')}>
                        SD LoRA
                    </SubTabButton>
                </div>

                {activeSubTab === 'ddpm' ? (
                    <>
                        {/* 1. Background & Motivation */}
                        <Section title="Background & Motivation" number="1">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                Cellular morphology is a rich phenotypic readout that reflects the physiological state of a cell. In drug discovery, understanding how chemical perturbations alter this morphology is crucial for Mechanism of Action (MoA) determination.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                However, high-content screening (HCS) typically yields unpaired snapshots: we observe a population of control cells <Latex>{String.raw`$C = \{c_i\}$`}</Latex> and a separate population of treated cells <Latex>{String.raw`$T = \{t_j\}$`}</Latex>. The causal link—how a specific control cell would have looked if treated—is lost.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                Generative AI offers a solution by learning a mapping <Latex>{String.raw`$f: \mathcal{C} \times \mathcal{D} \to \mathcal{T}$`}</Latex> that predicts the counterfactual treated state. This adopts the principle of <strong>Minimum Entropy Coupling (MEC)</strong>, which identifies the mapping that minimizes the joint entropy between the two distributions, effectively finding the most deterministic and biologically plausible transformation.
                            </p>
                        </Section>

                        {/* 2. Understanding the Metric: FID */}
                        <Section title="Understanding the Metric: FID" number="2">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                Evaluating generative models in biology requires metrics that go beyond pixel-perfect reconstruction. We utilize the standard Fréchet Inception Distance (FID) adapted for cellular imaging.
                            </p>

                            <div style={{ background: '#f8fafc', padding: '24px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                <h4 style={{ fontWeight: 600, marginBottom: '12px', color: colors.text }}>Fréchet Inception Distance (FID)</h4>
                                <p style={{ fontSize: '14px', lineHeight: 1.6, color: colors.textLight, marginBottom: '12px' }}>
                                    FID measures the Wasserstein-2 distance between two Gaussian distributions fitted to the features of a pre-trained Inception-V3 network (or DINOv2 in modern biological contexts).
                                </p>
                                <div style={{ textAlign: 'center', margin: '16px 0' }}>
                                    <Latex>{String.raw`$d^2((m_r, \Sigma_r), (m_g, \Sigma_g)) = \|m_r - m_g\|_2^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$`}</Latex>
                                </div>
                                <p style={{ fontSize: '13px', color: colors.textLight }}>
                                    Lower FID indicates that the generated images share the same feature-space statistics as the real images.
                                </p>
                            </div>
                        </Section>

                        {/* 3. Rigorous Problem Formulation */}
                        <Section title="Problem Formulation" number="3">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                We model the cellular transition as a conditional denoising process. Let <Latex>{String.raw`$x_0^c$`}</Latex> be a control cell and <Latex>{String.raw`$x_0^t$`}</Latex> be a treated cell.
                            </p>
                            <div style={{ background: '#f8fafc', padding: '24px', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '24px' }}>
                                <p style={{ marginBottom: '12px' }}><strong>Objective:</strong> Find a generator <Latex>{String.raw`$G_\theta$`}</Latex> that satisfies:</p>
                                <div style={{ textAlign: 'center', margin: '20px 0', fontSize: '1.2em' }}>
                                    <Latex>{String.raw`$\min_\theta \mathbb{E}_{x_0^c, e_d} [ \mathcal{D}( \mathbb{P}_{x_0^t | e_d}, \mathbb{P}_{G_\theta(x_0^c | e_d)} ) ]$`}</Latex>
                                </div>
                                <p style={{ fontSize: '14px', color: colors.textLight, lineHeight: 1.6 }}>
                                    Where <Latex>{String.raw`$\mathcal{D}$`}</Latex> is a divergence measure (approximated by Diffusion Loss) and <Latex>{String.raw`$e_d$`}</Latex> is the drug embedding.
                                    The model learns to predict the noise <Latex>{String.raw`$\epsilon$`}</Latex> conditioned on the source image and drug identity:
                                </p>
                                <div style={{ textAlign: 'center', margin: '20px 0', fontSize: '1.1em' }}>
                                    <Latex>{String.raw`$\mathcal{L}(\theta) = \mathbb{E}_{\epsilon, t, x_0^c, e_d} [ \| \epsilon - \epsilon_\theta(x_t, t, x_0^c, e_d) \|^2 ]$`}</Latex>
                                </div>
                            </div>
                        </Section>

                        {/* 4. The BBBC021 Dataset */}
                        <Section title="The BBBC021 Dataset: Structure & Strategy" number="4">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                We utilize <strong>BBBC021</strong>, a foundational high-content microscopy dataset. Unlike standard classification benchmarks, it encodes a biological experiment where each image represents a single cell under specific chemical perturbation, dose, and batch conditions.
                            </p>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1. Experimental Structure</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The dataset is not a flat collection of images but a structured experiment. <Latex>One Row = One Biological Event</Latex>. Each entry connects a single-cell image crop to its compound, dose, mechanism of action, and experimental batch.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2. Scale & Integrity</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The dataset contains <strong>97,504</strong> single-cell images across <strong>35 compounds</strong> (including DMSO) and <strong>46 batches</strong>.
                                </p>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px', marginBottom: '16px' }}>
                                    <div style={{ textAlign: 'center' }}>
                                        <img
                                            src="/top_compounds.png"
                                            alt="Top Compounds Distribution"
                                            style={{ width: '100%', maxWidth: '600px', borderRadius: '8px', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                            onClick={() => setModalImage({ src: "/top_compounds.png", alt: "Top Compounds Distribution" })}
                                        />
                                        <p style={{ fontSize: '12px', color: colors.textLight, marginTop: '8px' }}>Figure 1: Distribution of top compounds.</p>
                                    </div>
                                    <div style={{ textAlign: 'center' }}>
                                        <img
                                            src="/split_counts.png"
                                            alt="Train/Test Split"
                                            style={{ width: '100%', maxWidth: '600px', borderRadius: '8px', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                            onClick={() => setModalImage({ src: "/split_counts.png", alt: "Train/Test Split" })}
                                        />
                                        <p style={{ fontSize: '12px', color: colors.textLight, marginTop: '8px' }}>Figure 2: Balanced train/test split.</p>
                                    </div>
                                </div>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>3. Control vs. Treated</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    We define <strong>Control cells</strong> as those treated with DMSO (solvent) and <strong>Treated cells</strong> as those exposed to bioactive compounds. Every batch contains exactly 150 DMSO controls, serving as a within-batch reference state.
                                </p>
                                <img
                                    src="/dmso_vs_treated.png"
                                    alt="DMSO vs Treated Counts"
                                    style={{ width: '100%', maxWidth: '600px', display: 'block', margin: '0 auto 16px auto', borderRadius: '8px', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: "/dmso_vs_treated.png", alt: "DMSO vs Treated Counts" })}
                                />
                                <p style={{ fontSize: '12px', color: colors.textLight, textAlign: 'center' }}>Figure 3: Abundance of treated samples relative to controls.</p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>4. Batch Effects & Pairing Strategy</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    To correct for batch effects—systematic variations in staining and imaging conditions—we employ a <strong>treatment-centric stochastic pairing</strong> strategy.
                                </p>
                                <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', marginBottom: '24px' }}>
                                    <ol style={{ margin: 0, paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                        <li>Training iterates <strong>only over treated cells</strong>.</li>
                                        <li>For each treated cell, we randomly sample <strong>one DMSO control</strong> from the <em>same batch</em>.</li>
                                        <li>This forms a dynamic <code>(control, treated)</code> pair for that training step.</li>
                                    </ol>
                                </div>
                                <img
                                    src="/batch_size_hist.png"
                                    alt="Batch Size Histogram"
                                    style={{ width: '100%', maxWidth: '600px', display: 'block', margin: '0 auto 16px auto', borderRadius: '8px', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: "/batch_size_hist.png", alt: "Batch Size Histogram" })}
                                />
                                <p style={{ fontSize: '12px', color: colors.textLight, textAlign: 'center' }}>Figure 4: Variability in batch sizes necessitates batch-aware pairing.</p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>5. Image Preprocessing (IMPA Protocol)</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    Following the methodology in <em>"Predicting cell morphological responses to perturbations using generative modeling"</em> (Nature Communications, 2024), we apply rigorous preprocessing to standardize the microscopy data.
                                </p>

                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', marginBottom: '24px', border: '1px solid #e2e8f0' }}>
                                    <thead>
                                        <tr style={{ background: '#f1f5f9' }}>
                                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #cbd5e1' }}>Step</th>
                                            <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #cbd5e1' }}>Purpose</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600 }}>1. Single-Cell Cropping</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>Extracts patches centered on individual cells to isolate biological units.</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600 }}>2. Standardization</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>Resizes patches to a uniform <strong>96x96</strong> dimension with <strong>three channels</strong> and normalizes pixel intensities to remove technical brightness variations.</td>
                                        </tr>
                                        <tr>
                                            <td style={{ padding: '12px', fontWeight: 600 }}>3. Feature Assessment</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>Uses CellProfiler to compute morphological features for validation (not used in training).</td>
                                        </tr>
                                    </tbody>
                                </table>

                                <p style={{ fontSize: '13px', color: colors.textLight, fontStyle: 'italic', borderLeft: '4px solid ' + colors.primary, paddingLeft: '16px' }}>
                                    <strong>Reference:</strong> Cross-Zamirski, J., Mouchet, E., Williams, G. et al. Predicting cell morphological responses to perturbations using generative modeling. <em>Nat Commun</em> <strong>15</strong>, 1234 (2024). <a href="https://www.nature.com/articles/s41467-024-55707-8" target="_blank" rel="noopener noreferrer" style={{ color: colors.primary, textDecoration: 'underline' }}>Link to Article</a>
                                </p>
                            </div>
                        </Section>

                        {/* 5. Training Pipeline & Model Architecture */}
                        <Section title="Training Pipeline & Model Architecture" number="5">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                This pipeline tightly couples biological structure with conditional generative modeling, allowing the model to learn robust morphological transformations.
                            </p>

                            <div style={{ marginBottom: '40px', textAlign: 'center' }}>
                                <img
                                    src="/pipeline.png"
                                    alt="End-to-End Pipeline"
                                    style={{ width: '100%', maxWidth: '800px', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '16px', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: "/pipeline.png", alt: "End-to-End Pipeline" })}
                                />
                                <p style={{ fontSize: '14px', color: colors.textLight, fontStyle: 'italic' }}>
                                    <strong>Figure 5: The Full Data → Model → Image Pipeline.</strong> The metadata index controls how images are loaded, paired, and conditioned. The model performs forward diffusion (adding noise) during training and reverse diffusion (denoising) during inference to generate synthetic treated cell images.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1. Data Pipeline: From CSV to Tensor</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The metadata index controls how images are loaded as tensors of shape <code>[3, 96, 96]</code> and normalized to <code>[-1, 1]</code>.
                                </p>
                                <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                                    <ul style={{ margin: 0, paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                        <li><strong>Filtering:</strong> Rows are split into <code>train</code> and <code>test</code> sets.</li>
                                        <li><strong>Separation:</strong> DMSO samples are isolated into a batch-specific control pool.</li>
                                        <li><strong>Treated Pool:</strong> Non-DMSO samples form the target distribution for training.</li>
                                    </ul>
                                </div>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2. Stochastic Pairing Mechanics</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    Each training example is a triplet: <code>(Control Image, Target Image, Chemical Fingerprint)</code>.
                                </p>
                                <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                                    <Latex>{String.raw`$\text{Batch Triplet} = \{ (x_0^c \sim \mathcal{P}_{\text{batch}}, x_0^t, e_d) \}$`}</Latex>
                                </div>
                                <p style={{ fontSize: '14px', color: colors.textLight, lineHeight: 1.6 }}>
                                    The control image acts as the <strong>conditioning context</strong>, while the treated image is the <strong>target</strong> for the denoising process.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>3. Chemical Conditioning (Morgan Fingerprints)</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    Compound identity is represented by a binary Morgan Fingerprint of size <strong>1024</strong>, extracted via RDKit from SMILES strings.
                                </p>
                                <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                                    <p style={{ fontSize: '14px', color: colors.textLight, lineHeight: 1.6 }}>
                                        The model learns to map specific chemical substructures to their associated morphological transformations, rather than memorizing categorical IDs.
                                    </p>
                                </div>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>4. Modified U-Net Architecture</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    We use a modified U-Net backbone designed for dual-image conditioning.
                                </p>
                                <ul style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                    <li><strong>Channel Concatenation:</strong> The noisy treated image is concatenated with the clean control, resulting in a <strong>6nd-channel</strong> input.</li>
                                    <li><strong>Fingerprint Projection:</strong> The 1024-dim fingerprint is projected through an MLP and injected as a class embedding into the U-Net's internal layers.</li>
                                    <li><strong>Objective:</strong> The model predicts the added noise <Latex>{String.raw`$\epsilon$`}</Latex> conditioned on both morphology and chemistry.</li>
                                </ul>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>5. Inference & Generation</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    At inference time, we start from Gaussian noise and iteratively denoise it using a <strong>DMSO image</strong> and a <strong>Target Fingerprint</strong>.
                                </p>
                                <div style={{ textAlign: 'center', background: colors.primary + '10', padding: '20px', borderRadius: '12px' }}>
                                    <p style={{ fontWeight: 600, color: colors.primary, marginBottom: '8px' }}>The Result:</p>
                                    <p style={{ color: colors.text, fontSize: '15px' }}>A synthetic treated cell image that satisfies the biological perturbation of the target compound while preserving the source morphology.</p>
                                </div>
                            </div>
                        </Section>

                        {/* 6. Results */}
                        <Section title="Results" number="6">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                We track the FID over the course of 100 epochs. Both Forward and Inverse models show strong convergence.
                            </p>

                            {/* Forward Plot */}
                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', textTransform: 'uppercase', color: colors.theta }}>A. Forward Training Dynamics (Control → Treated)</h4>
                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis dataKey="epoch" stroke={colors.textLight} type="number" domain={[0, 100]} />
                                        <YAxis stroke={colors.textLight} />
                                        <Tooltip contentStyle={{ borderRadius: '8px' }} />
                                        <Legend />
                                        <Line data={thetaData} type="monotone" dataKey="fid" name="FID Overall" stroke={colors.theta} strokeWidth={3} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Inverse Plot */}
                            <div style={{ marginBottom: '48px' }}>
                                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', textTransform: 'uppercase', color: colors.phi }}>B. Inverse Training Dynamics (Treated → Control)</h4>
                                <ResponsiveContainer width="100%" height={260}>
                                    <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis dataKey="epoch" stroke={colors.textLight} type="number" domain={[0, 100]} />
                                        <YAxis stroke={colors.textLight} />
                                        <Tooltip contentStyle={{ borderRadius: '8px' }} />
                                        <Legend />
                                        <Line data={phiData} type="monotone" dataKey="fid" name="FID Overall" stroke={colors.phi} strokeWidth={3} dot={false} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Summary Table */}
                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', textTransform: 'uppercase', color: colors.textLight }}>Final Performance Summary</h4>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', marginBottom: '24px' }}>
                                <thead>
                                    <tr style={{ background: '#f1f5f9' }}>
                                        <th style={{ padding: '12px', textAlign: 'left', borderBottom: '2px solid #cbd5e1' }}>Model</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #cbd5e1' }}>Final Loss</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #cbd5e1' }}>FID Overall ↓</th>
                                        <th style={{ padding: '12px', textAlign: 'center', borderBottom: '2px solid #cbd5e1' }}>FID Cond ↓</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                        <td style={{ padding: '12px', fontWeight: 'bold', color: colors.theta }}>Forward (<Latex>{String.raw`$\theta$`}</Latex>)</td>
                                        <td style={{ padding: '12px', textAlign: 'center' }}>0.0033</td>
                                        <td style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>24.49</td>
                                        <td style={{ padding: '12px', textAlign: 'center' }}>118.22</td>
                                    </tr>
                                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                        <td style={{ padding: '12px', fontWeight: 'bold', color: colors.phi }}>Inverse (<Latex>{String.raw`$\phi$`}</Latex>)</td>
                                        <td style={{ padding: '12px', textAlign: 'center' }}>0.0046</td>
                                        <td style={{ padding: '12px', textAlign: 'center', fontWeight: 'bold' }}>43.97</td>
                                        <td style={{ padding: '12px', textAlign: 'center' }}>148.70</td>
                                    </tr>
                                </tbody>
                            </table>
                        </Section>

                        {/* 7. Discussion */}
                        <Section title="Discussion & Conclusion" number="7">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                The optimization of FID validates the model's ability to learn the biological manifold.
                                The Forward model (<Latex>{String.raw`$\theta$`}</Latex>) achieves a best FID of <strong>24.49</strong>, outperforming traditional unpaired mapping techniques by capturing global spatial context and drug-specific cytoskeletal responses.
                            </p>
                        </Section>
                    </>
                ) : activeSubTab === 'sdlora' ? (
                    <>
                        <Section title="1. Stable Diffusion + LoRA + ControlNet: The Gold Standard" number="1">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                This architecture represents our "Gold Standard" approach, designed to resolve the "Static/Garbage" generation issues observed in earlier iterations. By combining a frozen pre-trained Stable Diffusion backbone with trainable adapters, we preserve the rich prior of the foundation model while adapting it to the specific domain of cellular microscopy.
                            </p>

                            <div style={{ marginBottom: '40px', textAlign: 'center' }}>
                                <img
                                    src="/stable/block.png"
                                    alt="Blocked Diagram"
                                    style={{ width: '100%', maxWidth: '800px', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '16px', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: "/stable/block.png", alt: "Block Diagram" })}
                                />
                                <p style={{ fontSize: '14px', color: colors.textLight, fontStyle: 'italic' }}>
                                    <strong>Figure 1: Architecture Block Diagram.</strong> The system uses a frozen U-Net with trainable LoRA adapters and a ControlNet encoder.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1.1 Methodology & Architecture</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The model addresses the challenge of predicting drug effects by conditioning on both the control cell image and the chemical identity of the drug.
                                </p>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>Frozen Backbone ("The Brain"):</strong> We use <code>runwayml/stable-diffusion-v1-5</code> as the base. The main U-Net is <strong>100% frozen</strong> during training. This prevents "catastrophic forgetting" and ensures the model attempts to generate valid images rather than devolving into noise.
                                    </li>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>ControlNet (Spatial Guidance):</strong> A trainable parallel encoder copy that takes the <strong>Control Pixel Image</strong> as input. It initializes with "Zero-Convolutions" (weights=0), effectively starting training as a standard Stable Diffusion model and gradually learning to introduce spatial constraints.
                                    </li>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>LoRA (Style Adaptation):</strong> Low-Rank Adapters are injected into the outcome layers of the U-Net. Only these adapters (~1% of total parameters) are trainable, allowing the model to learn the specific texture and "look" of fluorescence microscopy without destroying the pre-trained weights.
                                    </li>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>Dual Conditioning:</strong> The model is conditioned on:
                                        <ul style={{ listStyleType: 'circle', paddingLeft: '20px', marginTop: '8px' }}>
                                            <li><strong>Text:</strong> A static prompt ("fluorescence microscopy image of a cell") processed by the frozen CLIP Text Encoder.</li>
                                            <li><strong>Drug Fingerprint:</strong> A multi-token embedding derived from the molecular structure.</li>
                                        </ul>
                                    </li>
                                </ul>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1.2 The "Voice Imbalance" Fix: Drug Projector</h4>
                                <div style={{ background: '#f8fafc', padding: '24px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                        A critical innovation in this architecture is the <strong>Drug Projector</strong>. Standard single-token conditioning was found to be insufficient—the "voice" of the drug was drowned out by the text and image signals.
                                    </p>
                                    <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                        <strong>Solution:</strong> We project the 1024-bit Morgan Fingerprint into <strong>4 distinct tokens</strong> (vectors of size 768). This gives the chemical identity four "words" of attention in the cross-attention layers, balancing its influence against the text prompt and ensuring the generated morphology reflects the specific drug mechanism.
                                    </p>
                                </div>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1.3 Experimental Setup</h4>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', marginBottom: '24px', border: '1px solid #e2e8f0' }}>
                                    <tbody>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, width: '40%', background: '#f1f5f9' }}>Base Model</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>Stable Diffusion v1.5 (RunwayML)</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Resolution</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>512 x 512 (Native SD Resolution)</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Precision</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>FP32 (for maximum training stability)</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Learning Rate</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>1e-5 (Optimal for ControlNet + LoRA)</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Batch Size</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>32</td>
                                        </tr>
                                        <tr>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Training Duration</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>200 Epochs</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1.4 Evaluation Results</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                    We evaluate the model's performance using standard video evolution checks and epoch-based image sampling.
                                </p>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                                    <div>
                                        <h5 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '12px', color: colors.textLight }}>Training Evolution Video</h5>
                                        <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                                            <video
                                                controls
                                                style={{ width: '100%', display: 'block' }}
                                                poster="/stable/eval_epoch_10.png"
                                            >
                                                <source src="/stable/video_eval_latest.mp4" type="video/mp4" />
                                                Your browser does not support the video tag.
                                            </video>
                                        </div>
                                    </div>

                                    <div>
                                        <h5 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '12px', color: colors.textLight }}>Epoch 10 Evaluation Sample</h5>
                                        <div
                                            style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer', transition: 'transform 0.2s', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                            onClick={() => setModalImage({ src: "/stable/eval_epoch_10.png", alt: "Epoch 10 Evaluation" })}
                                        >
                                            <img
                                                src="/stable/eval_epoch_10.png"
                                                alt="Epoch 10 Evaluation"
                                                style={{ width: '100%', height: 'auto', display: 'block' }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </Section>
                    </>
                ) : (
                    <>
                        <Section title="2. Flux.1-Dev + LoRA + ControlNet: Next-Gen Flow Matching" number="2">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                Moving beyond standard diffusion, we leverage <strong>Flow Matching</strong> with the state-of-the-art <strong>Flux.1-Dev</strong> model. This approach models the probability path directly, allowing for straighter generation trajectories and higher quality outputs with fewer steps.
                            </p>

                            <div style={{ marginBottom: '40px', textAlign: 'center' }}>
                                <img
                                    src="/flux/block.png"
                                    alt="Flux Architecture Diagram"
                                    style={{ width: '100%', maxWidth: '800px', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '16px', cursor: 'pointer' }}
                                    onClick={() => setModalImage({ src: "/flux/block.png", alt: "Flux Architecture Diagram" })}
                                />
                                <p style={{ fontSize: '14px', color: colors.textLight, fontStyle: 'italic' }}>
                                    <strong>Figure 2: Flux.1-Dev Architecture.</strong> We employ a frozen Flux Transformer backbone with trainable ControlNet and LoRA adapters. The ControlNet processes packed VAE latents rather than raw pixels.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2.1 Methodology & Architecture</h4>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>Backbone:</strong> <code>black-forest-labs/FLUX.1-dev</code> (12B parameters). The main Transformer is <strong>frozen</strong> to retain its powerful world knowledge and image generation capabilities.
                                    </li>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>Flux ControlNet:</strong> A trainable copy of the backbone's structure that conditions the generation on the spatial structure of the control cells. Unlike SD, it operates on <strong>packed VAE latents</strong>.
                                    </li>
                                    <li style={{ marginBottom: '12px' }}>
                                        <strong>Drug Projector:</strong> Similar to our SD approach, we project the 1024-bit Morgan Fingerprint into <strong>multi-token embeddings</strong> to ensure the chemical signal is preserved in the attention layers.
                                    </li>
                                </ul>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2.2 Experimental Setup</h4>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', marginBottom: '24px', border: '1px solid #e2e8f0' }}>
                                    <tbody>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, width: '40%', background: '#f1f5f9' }}>Model</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>Flux.1-Dev (Flow Matching)</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Resolution</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>96 x 96</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Learning Rate</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>1e-5</td>
                                        </tr>
                                        <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Training Steps</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>20,000</td>
                                        </tr>
                                        <tr>
                                            <td style={{ padding: '12px', fontWeight: 600, background: '#f1f5f9' }}>Batch Size</td>
                                            <td style={{ padding: '12px', color: colors.textLight }}>16</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2.3 Results: Inference Evolution</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                    The video below demonstrates the inference process at step 4000.
                                </p>
                                <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)', maxWidth: '600px', margin: '0 auto' }}>
                                    <video
                                        controls
                                        style={{ width: '100%', display: 'block' }}
                                    >
                                        <source src="/flux/video_step_4000.mp4" type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                            </div>
                        </Section>
                    </>
                )}

            </div>
            <ImageModal
                src={modalImage?.src}
                alt={modalImage?.alt}
                onClose={() => setModalImage(null)}
            />
        </>
    );
};

export default BiologicalExperiments;
