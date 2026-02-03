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
                        Cond DDPM & MEC
                    </SubTabButton>
                    <SubTabButton active={activeSubTab === 'uncond'} onClick={() => setActiveSubTab('uncond')}>
                        Uncond DDPM & MEC
                    </SubTabButton>
                    <SubTabButton active={activeSubTab === 'flux'} onClick={() => setActiveSubTab('flux')}>
                        Flux 12B + LoRA + ControlNet
                    </SubTabButton>
                    <SubTabButton active={activeSubTab === 'sdlora'} onClick={() => setActiveSubTab('sdlora')}>
                        Stable Diffusion + LoRA + ControlNet
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


                            {/* Visual Dataset Summary */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '32px' }}>
                                <div style={{ background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)', padding: '24px', borderRadius: '12px', border: '1px solid #bfdbfe', textAlign: 'center' }}>
                                    <h4 style={{ fontSize: '14px', color: '#1e40af', fontWeight: 600, marginBottom: '8px' }}>Total Single Cells</h4>
                                    <p style={{ fontSize: '32px', fontWeight: 800, color: '#1e3a8a', margin: 0 }}>97,504</p>
                                    <p style={{ fontSize: '12px', color: '#3b82f6', marginTop: '4px' }}>Across 35 Compounds</p>
                                </div>
                                <div style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                    <h4 style={{ fontSize: '13px', color: colors.textLight, fontWeight: 600, marginBottom: '16px', textTransform: 'uppercase' }}>Data Splits (Batch-Disjoint)</h4>
                                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '12px' }}>
                                        <div style={{ width: '80px', fontSize: '13px', fontWeight: 600, color: colors.text }}>Train</div>
                                        <div style={{ flex: 1, height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden', margin: '0 12px' }}>
                                            <div style={{ width: '90%', height: '100%', background: colors.primary, borderRadius: '4px' }}></div>
                                        </div>
                                        <div style={{ width: '60px', textAlign: 'right', fontSize: '13px', color: colors.textLight }}>87,716</div>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center' }}>
                                        <div style={{ width: '80px', fontSize: '13px', fontWeight: 600, color: colors.text }}>Test</div>
                                        <div style={{ flex: 1, height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden', margin: '0 12px' }}>
                                            <div style={{ width: '10%', height: '100%', background: colors.phi, borderRadius: '4px' }}></div>
                                        </div>
                                        <div style={{ width: '60px', textAlign: 'right', fontSize: '13px', color: colors.textLight }}>9,788</div>
                                    </div>
                                </div>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>1. Experimental Structure</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The dataset is not a flat collection of images but a structured experiment. <Latex>One Row = One Biological Event</Latex>. Each entry connects a single-cell image crop to its compound, dose, mechanism of action, and experimental batch.
                                </p>
                            </div>

                            <div style={{ marginBottom: '40px' }}>
                                <h4 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: colors.text }}>2. Scale & Integrity</h4>
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
                            {/* Qualitative Evaluation */}
                            <div style={{ marginBottom: '48px' }}>
                                <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '16px', textTransform: 'uppercase', color: colors.textLight }}>C. Qualitative Evaluation (Epoch 100)</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>

                                    {/* Forward Sample */}
                                    <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                        <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ background: colors.theta, color: 'white', fontSize: '12px', padding: '4px 8px', borderRadius: '4px', fontWeight: 600 }}>Forward</span>
                                            <span style={{ fontSize: '14px', color: colors.text, fontWeight: 600 }}>Control → Treated</span>
                                        </div>
                                        <div
                                            style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #cbd5e1', cursor: 'pointer', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}
                                            onClick={() => setModalImage({ src: "/bio/theta/theta_epoch_100.png", alt: "Forward Model Evaluation (Epoch 100)" })}
                                        >
                                            <img
                                                src="/bio/theta/theta_epoch_100.png"
                                                alt="Forward Evaluation"
                                                style={{ width: '100%', height: 'auto', display: 'block' }}
                                            />
                                        </div>
                                        <p style={{ marginTop: '12px', fontSize: '13px', color: colors.textLight, lineHeight: 1.5 }}>
                                            Generated samples (right) show accurate cytoskeletal reorganization matching the target drug profile while preserving cell count.
                                        </p>
                                        <div style={{ marginTop: '16px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #cbd5e1' }}>
                                            <video controls style={{ width: '100%', display: 'block' }}>
                                                <source src="/bio/theta/theta_video_100.mp4" type="video/mp4" />
                                                Your browser does not support the video tag.
                                            </video>
                                            <div style={{ padding: '8px', background: 'rgba(0,0,0,0.02)', borderTop: '1px solid #e2e8f0', fontSize: '11px', color: colors.textLight, textAlign: 'center' }}>
                                                Training Evolution (Epoch 1-100)
                                            </div>
                                        </div>
                                    </div>

                                    {/* Inverse Sample */}
                                    <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                                        <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ background: colors.phi, color: 'white', fontSize: '12px', padding: '4px 8px', borderRadius: '4px', fontWeight: 600 }}>Inverse</span>
                                            <span style={{ fontSize: '14px', color: colors.text, fontWeight: 600 }}>Treated → Control</span>
                                        </div>
                                        <div
                                            style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #cbd5e1', cursor: 'pointer', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}
                                            onClick={() => setModalImage({ src: "/bio/phi/phi_epoch_100.png", alt: "Inverse Model Evaluation (Epoch 100)" })}
                                        >
                                            <img
                                                src="/bio/phi/phi_epoch_100.png"
                                                alt="Inverse Evaluation"
                                                style={{ width: '100%', height: 'auto', display: 'block' }}
                                            />
                                        </div>
                                        <p style={{ marginTop: '12px', fontSize: '13px', color: colors.textLight, lineHeight: 1.5 }}>
                                            Restoration of the healthy phenotype (Control) from perturbed inputs, validating the reversibility of the learned mapping.
                                        </p>
                                        <div style={{ marginTop: '16px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #cbd5e1' }}>
                                            <video controls style={{ width: '100%', display: 'block' }}>
                                                <source src="/bio/phi/phi_video_100.mp4" type="video/mp4" />
                                                Your browser does not support the video tag.
                                            </video>
                                            <div style={{ padding: '8px', background: 'rgba(0,0,0,0.02)', borderTop: '1px solid #e2e8f0', fontSize: '11px', color: colors.textLight, textAlign: 'center' }}>
                                                Training Evolution (Epoch 1-100)
                                            </div>
                                        </div>
                                    </div>

                                </div>
                            </div>

                            <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '24px', textTransform: 'uppercase', color: colors.textLight }}>D. Comprehensive Benchmark Evaluation</h4>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '32px', marginBottom: '40px' }}>

                                {/* Table 1: Forward Dynamics */}
                                <div>
                                    <h5 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: colors.text }}>Table 1. Forward Dynamics (Control → Treated)</h5>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Samples</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>FID ↓</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>cFID ↓</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px' }}>1K</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center' }}>24.49</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center' }}>118.22</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px' }}>5K</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: colors.theta }}>11.94</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: colors.theta }}>74.2</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                {/* Table 2: Inverse Dynamics */}
                                <div>
                                    <h5 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: colors.text }}>Table 2. Inverse Dynamics (Treated → Control)</h5>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Samples</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>FID ↓</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>cFID ↓</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px' }}>1K</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center' }}>43.97</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center' }}>148.70</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px' }}>5K</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: colors.phi }}>35.50</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, color: colors.phi }}>90.80</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '32px' }}>

                                {/* Table 3: Scaling Comparison */}
                                <div>
                                    <h5 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: colors.text }}>Table 3. Method Scaling Comparison</h5>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Method</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>1K FID ↓</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>5K FID ↓</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', color: colors.textLight }}>PhenDiff</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>71.3</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>49.5</td>
                                            </tr>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', color: colors.textLight }}>IMPA</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>52.4</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>33.7</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px', fontWeight: 600, color: colors.primary }}>CellFlux </td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: colors.primary }}>34.7</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: colors.primary }}>18.7</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                {/* Table 4: Final Benchmark */}
                                <div>
                                    <h5 style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', color: colors.text }}>Table 4. Final Benchmark Comparison (5K)</h5>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Method</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>FID<sub style={{ fontSize: '10px' }}>o</sub> ↓</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>FID<sub style={{ fontSize: '10px' }}>c</sub> ↓</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', color: colors.textLight }}>PhenDiff (MICCAI'24)</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>49.5</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>109.2</td>
                                            </tr>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', color: colors.textLight }}>IMPA (Nature Comm'25)</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>33.7</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.textLight }}>76.5</td>
                                            </tr>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9', background: '#f0fdf4' }}>
                                                <td style={{ padding: '8px 12px', fontWeight: 600, color: colors.primary }}>CellFlux</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: colors.primary }}>18.7</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: colors.primary }}>56.8</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px', color: colors.theta }}>Pretrained Conditional DDPM</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>11.94</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>75.2</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                            </div>

                            <p style={{ marginTop: '24px', fontSize: '13px', color: colors.textLight, fontStyle: 'italic', textAlign: 'center' }}>
                                <strong>Benchmark Note:</strong> "CellFlux" represents our proposed Flux.1-Dev based method using Flow Matching, which outperforms previous state-of-the-art methods (PhenDiff, IMPA) in both overall image quality (FID) and conditional alignment (cFID).
                            </p>
                        </Section>

                        {/* 7. Evolution Strategies (ES) Fine-tuning */}
                        <Section title="Evolution Strategies (ES) Fine-tuning" number="7">

                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                We implement an <strong>end-to-end pipeline</strong> for training two conditional diffusion models (Theta <Latex>{String.raw`$\theta$`}</Latex> and Phi <Latex>{String.raw`$\phi$`}</Latex>) using a hybrid of Evolution Strategies (ES) and supervised diffusion loss. This approach ensures cycle consistency and high-fidelity translation between Control and Treated states.
                            </p>

                            {/* Block Diagram */}
                            <div style={{ marginBottom: '40px', textAlign: 'center', background: '#f8fafc', padding: '24px', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                <div
                                    style={{ borderRadius: '8px', overflow: 'hidden', cursor: 'pointer', marginBottom: '12px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    onClick={() => setModalImage({ src: "/finetune/es/block.jpg", alt: "ES-DDMEC Training Pipeline" })}
                                >
                                    <img
                                        src="/finetune/es/block.jpg"
                                        alt="ES-DDMEC Pipeline"
                                        style={{ width: '100%', maxHeight: '500px', objectFit: 'contain', display: 'block' }}
                                    />
                                </div>
                                <p style={{ fontSize: '13px', color: colors.textLight, fontStyle: 'italic' }}>
                                    <strong>Figure 7.1: End-to-End Training Pipeline.</strong> The system alternates between ES-based reward maximization (using the partner model as a critic) and supervised diffusion fine-tuning.
                                </p>
                            </div>

                            {/* 7.1 Problem Statement */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.1 Problem Formulation</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    The goal is to learn a bidirectional generative translation between <strong>Control</strong> (DMSO) and <strong>Treated</strong> (Compound) cell states:
                                </p>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Theta (<Latex>{String.raw`$\theta$`}</Latex>): Control <Latex>{String.raw`$\to$`}</Latex> Treated.</strong> Given a control image <Latex>{String.raw`$\mathbf{x}$`}</Latex> and chemical fingerprint <Latex>{String.raw`$c$`}</Latex>, generate the corresponding treated state.</li>
                                    <li><strong>Phi (<Latex>{String.raw`$\phi$`}</Latex>): Treated <Latex>{String.raw`$\to$`}</Latex> Control.</strong> Given a treated image <Latex>{String.raw`$\mathbf{y}$`}</Latex> and fingerprint <Latex>{String.raw`$c$`}</Latex>, restore the healthy control state.</li>
                                </ul>
                            </div>

                            {/* 7.2 Data Pipeline */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.2 Data Pipeline & Conditioning</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    We utilize a custom <code>BBBC021Dataset</code> and <code>PairedDataLoader</code> to ensure robust training:
                                </p>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Batch Pairing:</strong> Every treated sample is paired with a random control image <em>from the same experimental batch</em> to prevent the model from learning batch artifacts instead of biological effects.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Chemical Conditioning:</strong> 1024-bit Morgan Fingerprints are extracted using RDKit (radius=2) to deterministically represent molecular structure.</li>
                                    <li><strong>Robust Loading:</strong> A fault-tolerant path resolver handles the complex directory structure of the BBBC021 dataset.</li>
                                </ul>
                            </div>

                            {/* 7.3 Model Architecture */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.3 Architecture: Conditional Diffusion</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    We modify the standard UNet2DModel to support dual conditioning:
                                </p>
                                <ol style={{ listStyleType: 'decimal', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Image Concatenation:</strong> The conditioning image (e.g., Control) is concatenated channel-wise with the noisy input <Latex>{String.raw`$\mathbf{x}_t$`}</Latex>, resulting in a 6-channel input. Weights for the new channels are zero-initialized ("conv surgery") to preserve pretrained knowledge.</li>
                                    <li><strong>Fingerprint Embedding:</strong> The 1024-dim fingerprint is projected to the time-embedding dimension and injected via the <code>class_labels</code> mechanism, providing global semantic guidance.</li>
                                </ol>
                            </div>

                            {/* 7.4 Reward Definition */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.4 Reward: Diffusion-based Likelihood (DDMEC)</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    Instead of a discriminator, we use the partner diffusion model as a critic. The reward is the approximate negative log-likelihood of the generated sample under the inverse model:
                                </p>
                                <div style={{ background: '#f1f5f9', padding: '16px', borderRadius: '8px', marginBottom: '12px', textAlign: 'center', fontFamily: 'monospace', fontSize: '13px' }}>
                                    Reward <Latex>{String.raw`$\approx -\log p_{\phi}(\text{Control} \mid \text{Generated Treated})$`}</Latex>
                                </div>
                                <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                    This encourages <Latex>{String.raw`$\theta$`}</Latex> to generate treated cells that <Latex>{String.raw`$\phi$`}</Latex> confidently recognizes as mapping back to the control state, enforcing cycle consistency via likelihood maximization.
                                </p>
                            </div>

                            {/* 7.5 The 4-Phase Loop */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.5 The 4-Phase Co-Evolution Loop</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    Training alternates between gradient-free ES updates and supervised fine-tuning in a 4-step cycle:
                                </p>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px' }}>
                                    <div style={{ border: `1px solid ${colors.theta}`, borderRadius: '8px', padding: '16px', background: '#eff6ff' }}>
                                        <strong style={{ color: colors.theta, display: 'block', marginBottom: '8px' }}>Phase A: ES Update <Latex>{String.raw`$\theta$`}</Latex> (using <Latex>{String.raw`$\phi$`}</Latex>)</strong>
                                        <span style={{ fontSize: '13px', color: colors.textLight }}>
                                            Generate candidate treated images. Update <Latex>{String.raw`$\theta$`}</Latex> to maximize their likelihood under <Latex>{String.raw`$\phi$`}</Latex>.
                                        </span>
                                    </div>
                                    <div style={{ border: '1px solid #cbd5e1', borderRadius: '8px', padding: '16px', background: '#fff' }}>
                                        <strong style={{ color: colors.text, display: 'block', marginBottom: '8px' }}>Phase B: Supervised Update <Latex>{String.raw`$\phi$`}</Latex></strong>
                                        <span style={{ fontSize: '13px', color: colors.textLight }}>
                                            Use <Latex>{String.raw`$\theta$`}</Latex>'s generated images as input. Train <Latex>{String.raw`$\phi$`}</Latex> to denoise them back to the real Control images (Standard DDPM Loss).
                                        </span>
                                    </div>
                                    <div style={{ border: `1px solid ${colors.phi}`, borderRadius: '8px', padding: '16px', background: '#fef2f2' }}>
                                        <strong style={{ color: colors.phi, display: 'block', marginBottom: '8px' }}>Phase C: ES Update <Latex>{String.raw`$\phi$`}</Latex> (using <Latex>{String.raw`$\theta$`}</Latex>)</strong>
                                        <span style={{ fontSize: '13px', color: colors.textLight }}>
                                            Generate candidate control images. Update <Latex>{String.raw`$\phi$`}</Latex> to maximize their likelihood under <Latex>{String.raw`$\theta$`}</Latex>.
                                        </span>
                                    </div>
                                    <div style={{ border: '1px solid #cbd5e1', borderRadius: '8px', padding: '16px', background: '#fff' }}>
                                        <strong style={{ color: colors.text, display: 'block', marginBottom: '8px' }}>Phase D: Supervised Update <Latex>{String.raw`$\theta$`}</Latex></strong>
                                        <span style={{ fontSize: '13px', color: colors.textLight }}>
                                            Use <Latex>{String.raw`$\phi$`}</Latex>'s generated images as input. Train <Latex>{String.raw`$\theta$`}</Latex> to denoise them back to the real Treated images.
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* 7.6 Evaluation Results */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>7.6 Evaluation Results</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The following table presents the comprehensive evaluation metrics for the ES fine-tuned models after 50 epochs of training, evaluated on 5,000 samples:
                                </p>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0', marginBottom: '16px' }}>
                                    <thead>
                                        <tr style={{ background: '#f1f5f9' }}>
                                            <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Metric</th>
                                            <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Control (Phi)</th>
                                            <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Treated (Theta)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                            <td style={{ padding: '8px 12px', fontWeight: 600 }}>FID ↓</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.phi }}>148.15</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>60.36</td>
                                        </tr>
                                        <tr>
                                            <td style={{ padding: '8px 12px', fontWeight: 600 }}>CFID ↓</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.phi }}>151.5046</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>71.3897</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <p style={{ fontSize: '12px', color: colors.textLight, fontStyle: 'italic', lineHeight: 1.6 }}>
                                    <strong>Note:</strong> FID (Fréchet Inception Distance) measures overall image quality, and CFID (Conditional FID) evaluates conditional alignment between control and treated states. Lower values indicate better performance.
                                </p>
                            </div>

                        </Section>


                        {/* 8. PPO-DDMEC Fine-tuning */}
                        <Section title="PPO-DDMEC Fine-tuning: Diffusion as Policy" number="8">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                While Evolution Strategies (ES) provides a gradient-free approach to optimization, <strong>Proximal Policy Optimization (PPO)</strong> allows us to treat the diffusion process as a stochastic policy and optimize it directly using reinforcement learning. This method, <strong>DDMEC-PPO</strong>, leverages the probability density function of the diffusion model to enforce cycle consistency through a rigorous likelihood-based reward.
                            </p>

                            {/* Block Diagram */}
                            <div style={{ marginBottom: '40px', textAlign: 'center', background: '#f8fafc', padding: '24px', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                <div
                                    style={{ borderRadius: '8px', overflow: 'hidden', cursor: 'pointer', marginBottom: '12px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    onClick={() => setModalImage({ src: "/finetune/ppo/block.jpg", alt: "PPO-DDMEC Training Pipeline" })}
                                >
                                    <img
                                        src="/finetune/ppo/block.jpg"
                                        alt="PPO-DDMEC Pipeline"
                                        style={{ width: '100%', maxHeight: '500px', objectFit: 'contain', display: 'block' }}
                                    />
                                </div>
                                <p style={{ fontSize: '13px', color: colors.textLight, fontStyle: 'italic' }}>
                                    <strong>Figure 8.1: PPO-DDMEC Architecture.</strong> The diffusion sampling process is treated as a policy trajectory. We collect rollouts, compute log-probabilities, and update the model using the PPO clipped objective to maximize the cycle-consistency reward.
                                </p>
                            </div>

                            {/* 8.1 Diffusion as Policy */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>8.1 Key Idea: Diffusion Models as Policies</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    In this framework, we view the iterative denoising process as a sequential decision-making problem (an MDP):
                                </p>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    <li><strong>State <Latex>{String.raw`$s_t$`}</Latex>:</strong> The current noisy image <Latex>{String.raw`$\mathbf{x}_t$`}</Latex>.</li>
                                    <li><strong>Action <Latex>{String.raw`$a_t$`}</Latex>:</strong> The sampled less-noisy image <Latex>{String.raw`$\mathbf{x}_{t-1}$`}</Latex>.</li>
                                    <li><strong>Policy <Latex>{String.raw`$\pi_\theta(a_t|s_t)$`}</Latex>:</strong> The diffusion posterior <Latex>{String.raw`$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$`}</Latex>, parameterized as a Gaussian <Latex>{String.raw`$\mathcal{N}(\mu_\theta(\mathbf{x}_t), \sigma_t^2\mathbf{I})$`}</Latex>.</li>
                                </ul>
                                <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                    By computing the <strong>log-probability</strong> of each sampling step, we can apply standard RL algorithms like PPO to shift the distribution of generated images towards those that yield higher rewards.
                                </p>
                            </div>

                            {/* 8.2 Reward Signal */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>8.2 The Reward: Likelihood-based Cycle Consistency</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    Crucially, we do not use a "black box" discriminator. Instead, we use the <strong>partner diffusion model</strong> as the critic. If we generate a Treated image <Latex>{String.raw`$\hat{y}$`}</Latex> from Control <Latex>{String.raw`$x$`}</Latex>, the reward depends on how likely it is to recover <Latex>{String.raw`$x$`}</Latex> using the inverse model <Latex>{String.raw`$\phi$`}</Latex>.
                                </p>
                                <div style={{ background: '#f1f5f9', padding: '16px', borderRadius: '8px', marginBottom: '12px', textAlign: 'center', fontFamily: 'monospace', fontSize: '13px' }}>
                                    Reward <Latex>{String.raw`$r \approx -\log p_{\phi}(\text{Control} \mid \text{Generated Treated})$`}</Latex>
                                </div>
                                <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                    This reward is estimated via the <strong>diffusion training loss</strong> of the partner model (the "evidence lower bound" or ELBO proxy), ensuring that the generated images lie on the manifold learned by the inverse mapping.
                                </p>
                            </div>

                            {/* 8.3 PPO Update */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>8.3 The PPO Update</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                    We update the model to maximize the expected reward while preventing it from deviating too far from the original pretrained distribution (to avoid "mode collapse" or generating nonsense that happens to satisfy the critic).
                                </p>
                                <div style={{ background: '#fff', border: '1px solid #e2e8f0', padding: '16px', borderRadius: '8px' }}>
                                    <p style={{ marginBottom: '8px', fontWeight: 600, fontSize: '13px' }}>PPO Objective with "Pretraining Anchor":</p>
                                    <div style={{ textAlign: 'center', margin: '12px 0', fontSize: '1.1em' }}>
                                        <Latex>{String.raw`$L = \mathbb{E} [ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t) ] - \beta \cdot \|\epsilon_\theta - \epsilon_{\text{ref}}\|^2$`}</Latex>
                                    </div>
                                    <p style={{ fontSize: '12px', color: colors.textLight }}>
                                        where <Latex>{String.raw`$r_t$`}</Latex> is the probability ratio, <Latex>{String.raw`$A_t$`}</Latex> is the advantage, and the last term is an MSE penalty anchoring the model to a frozen reference UNet.
                                    </p>
                                </div>
                            </div>

                            {/* 8.4 Evaluation Results */}
                            <div style={{ marginBottom: '32px' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>8.4 Evaluation Results</h4>
                                <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                    The following table presents the comprehensive evaluation metrics for the PPO fine-tuned models:
                                </p>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0', marginBottom: '16px' }}>
                                    <thead>
                                        <tr style={{ background: '#f1f5f9' }}>
                                            <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Metric</th>
                                            <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Control (Phi)</th>
                                            <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Treated (Theta)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                            <td style={{ padding: '8px 12px', fontWeight: 600 }}>FID ↓</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.phi }}>200.12</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>80.82</td>
                                        </tr>
                                        <tr>
                                            <td style={{ padding: '8px 12px', fontWeight: 600 }}>CFID ↓</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.phi }}>251.5046</td>
                                            <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.theta }}>96.3897</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <p style={{ fontSize: '12px', color: colors.textLight, fontStyle: 'italic', lineHeight: 1.6 }}>
                                    <strong>Note:</strong> FID (Fréchet Inception Distance) measures overall image quality, and CFID (Conditional FID) evaluates conditional alignment between control and treated states. Lower values indicate better performance.
                                </p>
                            </div>
                        </Section>

                        {/* 9. Discussion */}
                        <Section title="Discussion & Conclusion" number="9">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                The optimization of FID validates the model's ability to learn the biological manifold. Our comprehensive evaluation reveals distinct performance characteristics across different fine-tuning approaches.
                            </p>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', marginTop: '24px', color: colors.text }}>Baseline Performance</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                The pretrained conditional DDPM establishes a strong baseline with FID of <strong>11.94</strong> and CFID of <strong>75.2</strong> for the forward direction (Control → Treated), demonstrating the effectiveness of supervised diffusion training in capturing morphological transformations.
                            </p>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', marginTop: '24px', color: colors.text }}>Evolution Strategies (ES) Fine-tuning</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                ES fine-tuning achieves competitive results with FID of <strong>60.36</strong> and CFID of <strong>71.3897</strong> for the treated direction (Theta). The control direction (Phi) shows FID of <strong>148.15</strong> and CFID of <strong>151.5046</strong>. While ES demonstrates effective cycle consistency through the likelihood-based reward mechanism, the gradient-free optimization approach shows higher variance in the inverse mapping direction.
                            </p>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', marginTop: '24px', color: colors.text }}>PPO Fine-tuning</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                PPO fine-tuning yields FID of <strong>80.82</strong> and CFID of <strong>96.3897</strong> for the treated direction (Theta), with control direction (Phi) achieving FID of <strong>200.12</strong> and CFID of <strong>251.5046</strong>. The policy gradient approach provides more stable optimization compared to ES, though both methods show that the inverse mapping (Treated → Control) presents greater challenges than the forward direction.
                            </p>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', marginTop: '24px', color: colors.text }}>Comparative Analysis</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                Comparing the two RL fine-tuning methods, ES achieves superior performance in the forward direction (FID: 60.36 vs 80.82), suggesting that gradient-free exploration may be more effective for learning the Control → Treated transformation. However, both methods struggle with the inverse mapping, indicating that restoring healthy phenotypes from perturbed states is inherently more challenging. The baseline supervised approach remains the strongest performer overall, highlighting the importance of direct supervision when paired data is available.
                            </p>
                        </Section>
                    </>
                ) : activeSubTab === 'uncond' ? (
                    <>
                        {/* 1) What data the models see */}
                        <Section title="1. Methodology: Data & Preprocessing" number="1">
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>1.1 Base Dataset: BBBC021Dataset</h4>
                            <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <li><strong>Source:</strong> Reads metadata from <code>./data/bbbc021_all/metadata/bbbc021_df_all.csv</code>.</li>
                                <li><strong>Filtering:</strong> Optionally filters by <code>SPLIT</code> column (e.g., "train"). Automatically retries without filtering if no samples are found.</li>
                                <li><strong>Content:</strong> Each row corresponds to one microscopy image stored as a <code>.npy</code> file.</li>
                            </ul>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>1.2 Robust File Resolution</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                The dataset employs a multi-strategy file finder to handle inconsistent paths:
                            </p>
                            <ol style={{ listStyleType: 'decimal', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <li><strong>Parse SAMPLE_KEY:</strong> Reconstructs expected paths (e.g., <code>Week7_34681...</code> &rarr; <code>Week7/34681/7_3338_348.0.npy</code>).</li>
                                <li><strong>Lookup Table:</strong> Checks <code>paths.csv</code> for exact filenames, relative path matches, or basename matches.</li>
                                <li><strong>Direct & Recursive:</strong> Fallback to direct path checks and recursive <code>rglob()</code> searches.</li>
                            </ol>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>1.3 Image Preprocessing</h4>
                            <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <li><strong>Format:</strong> Converts <code>[H,W,C]</code> to <code>[C,H,W]</code> if necessary and casts to <code>float32</code>.</li>
                                <li><strong>Normalization:</strong> Scales pixel values to <code>[-1, 1]</code>. Handles both <code>[0, 255]</code> and <code>[0, 1]</code> input ranges automatically.</li>
                                <li><strong>Clamping:</strong> Strictly clamps final values to the <code>[-1, 1]</code> range.</li>
                            </ul>
                        </Section>

                        {/* 2) How the data is split */}
                        <Section title="2. Training Sets Split" number="2">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                We use the <code>UnconditionalDataset</code> wrapper to create two distinct training populations based on compound treatment:
                            </p>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
                                <div style={{ background: '#f0fdf4', padding: '16px', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
                                    <h5 style={{ fontWeight: 600, color: '#166534', marginBottom: '8px' }}>Control Dataset</h5>
                                    <p style={{ fontSize: '13px', color: '#14532d' }}>
                                        Filters for <code>CPD_NAME == "DMSO"</code>. Contains only healthy control images.
                                    </p>
                                </div>
                                <div style={{ background: '#fef2f2', padding: '16px', borderRadius: '8px', border: '1px solid #fecaca' }}>
                                    <h5 style={{ fontWeight: 600, color: '#991b1b', marginBottom: '8px' }}>Perturbed Dataset</h5>
                                    <p style={{ fontSize: '13px', color: '#7f1d1d' }}>
                                        Filters for <code>CPD_NAME != "DMSO"</code>. Contains all treated/perturbed images.
                                    </p>
                                </div>
                            </div>
                            <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                Each dataloader returns simple image tensors of shape <code>[3, 96, 96]</code> suitable for unconditional training.
                            </p>
                        </Section>

                        {/* 3) What models are trained */}
                        <Section title="3. Model Training Setup" number="3">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                To capture the distinct morphological characteristics of the biological states, we train <strong>two independent unconditional DDPMs</strong> without weight sharing. The <strong>Control DDPM</strong> is trained exclusively on DMSO-treated samples to learn the baseline healthy distribution <Latex>{String.raw`$p(x \mid \text{DMSO})$`}</Latex>, while the <strong>Perturbed DDPM</strong> is trained on compound-treated images to model the modified phenotype <Latex>{String.raw`$p(x \mid \text{Treated})$`}</Latex>. This separation ensures that each model specializes in its respective domain before being coupled in the fine-tuning stage.
                            </p>
                        </Section>

                        {/* 4) Model Architecture */}
                        <Section title="4. Model Architecture: UnconditionalUNet" number="4">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                We employ a <strong>standard UNet2DModel</strong> architecture from the Diffusers library, adapted to our specific 96x96 resolution. To leverage transfer learning, we initialize the weights from <code>google/ddpm-cifar10-32</code> where dimensions align, allowing the model to inherit basic denoising filters.
                            </p>

                            {/* <div style={{ marginBottom: '32px', textAlign: 'center' }}>
                                <div
                                    style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer', display: 'inline-block' }}
                                    onClick={() => setModalImage({ src: "/stable/block.png", alt: "Architecture Block Diagram" })}
                                >
                                    <img
                                        src="/stable/block.png"
                                        alt="Architecture Block Diagram"
                                        style={{ maxWidth: '100%', maxHeight: '400px', display: 'block' }}
                                    />
                                </div>
                            </div> */}

                            <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0', marginBottom: '24px' }}>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', margin: 0, color: colors.textLight }}>
                                    <li style={{ marginBottom: '8px' }}><strong>Input/Output:</strong> Processed tensors of shape <code>[3, 96, 96]</code>.</li>
                                    <li style={{ marginBottom: '8px' }}><strong>Initialization:</strong> Partial loading from CIFAR-10 pretrained weights (strict=False) to accelerate convergence.</li>
                                    <li><strong>Capacity:</strong> Deep residual UNet with attention mechanisms at lower resolutions.</li>
                                </ul>
                            </div>
                        </Section>

                        {/* 5) Diffusion Methodology */}
                        <Section title="5. Diffusion Methodology (DDPM)" number="5">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                We utilize the <strong>DDPMScheduler</strong> with a linear beta schedule ranging from <Latex>{String.raw`$10^{-4}$`}</Latex> to <Latex>{String.raw`$0.02$`}</Latex> over <Latex>{String.raw`$T=1000$`}</Latex> timesteps. The model is optimized using the standard <Latex>{String.raw`$\epsilon$`}</Latex>-prediction objective. At each step <Latex>{String.raw`$t$`}</Latex>, the network predicts the noise component <Latex>{String.raw`$\epsilon_\theta(x_t, t)$`}</Latex> added to the original image <Latex>{String.raw`$x_0$`}</Latex>, minimizing the mean squared error <Latex>{String.raw`$||\epsilon - \epsilon_\theta||^2$`}</Latex>.
                            </p>
                        </Section>

                        {/* 6) Sampling Methodology */}
                        <Section title="6. Sampling Methodology" number="6">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                Reverse diffusion is used to generate samples:
                            </p>
                            <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                <li style={{ marginBottom: '8px' }}>Start from random Gaussian noise <Latex>{String.raw`$x_T \sim \mathcal{N}(0, I)$`}</Latex>.</li>
                                <li style={{ marginBottom: '8px' }}>Iterate backwards (typically 200 steps for visualization) using the scheduler to remove noise.</li>
                                <li>Clamp the final output to <code>[-1, 1]</code>.</li>
                            </ul>
                        </Section>

                        {/* 7) Training Loop */}
                        <Section title="7. Training Loop Structure" number="7">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                The pretraining process is driven by the <strong>AdamW optimizer</strong> with a learning rate of <Latex>{String.raw`$3 \times 10^{-5}$`}</Latex> and a weight decay of <Latex>{String.raw`$0.01$`}</Latex> to ensure regularization. We employ a <strong>Cosine Annealing LR scheduler</strong> that gradually decays the learning rate to <Latex>{String.raw`$10^{-6}$`}</Latex>, promoting stable convergence in the final epochs.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                Training runs for <strong>100 epochs</strong>. To rigorously monitor progress, we perform sample generation and saving checkpoints at every epoch, favoring visual verification over complex metrics like FID during this initial unconditional phase.
                            </p>
                        </Section>

                        {/* 8) Results */}
                        <Section title="8. Results: Pretraining Phase" number="8">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                Below are the training dynamics and generated samples for the unconditional pretraining phase.
                            </p>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>8.1 Training Convergence</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '40px' }}>
                                <div>
                                    <div
                                        style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                        onClick={() => setModalImage({ src: "/uncond/pretrain/uncond control ddpm.png", alt: "Uncond Control DDPM Training" })}
                                    >
                                        <img
                                            src="/uncond/pretrain/uncond control ddpm.png"
                                            alt="Uncond Control DDPM Training"
                                            style={{ width: '100%', display: 'block' }}
                                        />
                                    </div>
                                    <p style={{ fontSize: '13px', color: colors.textLight, marginTop: '8px', textAlign: 'center' }}>Control Model Loss</p>
                                </div>
                                <div>
                                    <div
                                        style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                        onClick={() => setModalImage({ src: "/uncond/pretrain/uncond perturbed ddpm.png", alt: "Uncond Perturbed DDPM Training" })}
                                    >
                                        <img
                                            src="/uncond/pretrain/uncond perturbed ddpm.png"
                                            alt="Uncond Perturbed DDPM Training"
                                            style={{ width: '100%', display: 'block' }}
                                        />
                                    </div>
                                    <p style={{ fontSize: '13px', color: colors.textLight, marginTop: '8px', textAlign: 'center' }}>Perturbed Model Loss</p>
                                </div>
                            </div>

                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px', color: colors.text }}>8.2 Generated Samples</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
                                <div>
                                    <div
                                        style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                        onClick={() => setModalImage({ src: "/uncond/pretrain/controls_samples.png", alt: "Control Samples" })}
                                    >
                                        <img
                                            src="/uncond/pretrain/controls_samples.png"
                                            alt="Control Samples"
                                            style={{ width: '100%', display: 'block' }}
                                        />
                                    </div>
                                    <p style={{ fontSize: '13px', color: colors.textLight, marginTop: '8px', textAlign: 'center' }}>Generated Control Samples</p>
                                </div>
                                <div>
                                    <div
                                        style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer' }}
                                        onClick={() => setModalImage({ src: "/uncond/pretrain/perturbed_samples.png", alt: "Perturbed Samples" })}
                                    >
                                        <img
                                            src="/uncond/pretrain/perturbed_samples.png"
                                            alt="Perturbed Samples"
                                            style={{ width: '100%', display: 'block' }}
                                        />
                                    </div>
                                    <p style={{ fontSize: '13px', color: colors.textLight, marginTop: '8px', textAlign: 'center' }}>Generated Perturbed Samples</p>
                                </div>
                            </div>
                        </Section>

                        {/* 9) PPO-DDMEC Methodology */}
                        <Section title="9. PPO-DDMEC Fine-tuning Methodology" number="9">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                Below is a mathematical explanation of the PPO-DDMEC fine-tuning process, detailing the data flow, model definitions, and the joint optimization objective.
                            </p>

                            {/* Block Diagram */}
                            <div style={{ marginBottom: '40px', textAlign: 'center', background: '#f8fafc', padding: '24px', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                                <div
                                    style={{ borderRadius: '8px', overflow: 'hidden', cursor: 'pointer', marginBottom: '12px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                    onClick={() => setModalImage({ src: "/uncond/finetune/block diaghram.jpg", alt: "PPO-DDMEC Training Logic" })}
                                >
                                    <img
                                        src="/uncond/finetune/block diaghram.jpg"
                                        alt="PPO-DDMEC Training Logic"
                                        style={{ width: '100%', maxHeight: '500px', objectFit: 'contain', display: 'block' }}
                                    />
                                </div>
                                <p style={{ fontSize: '13px', color: colors.textLight, fontStyle: 'italic', marginBottom: '16px' }}>
                                    <strong>Figure 9.1: PPO-DDMEC Training Logic.</strong> The diagram illustrates the interplay between the Forward (<Latex>{String.raw`$\theta$`}</Latex>) and Backward (<Latex>{String.raw`$\phi$`}</Latex>) models.
                                </p>
                                <p style={{ fontSize: '13px', color: colors.textLight, textAlign: 'left', lineHeight: 1.6 }}>
                                    The process involves generating a trajectory (Rollout) using the active policy (e.g., <Latex>{String.raw`$\theta$`}</Latex>), calculating a likelihood-based reward using the frozen partner model (<Latex>{String.raw`$\phi$`}</Latex>) as a critic, and updating the policy via PPO to maximize this reward while staying close to the pretrained marginals (KL constraint).
                                </p>
                            </div>

                            {/* 9.0 Problem Statement */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.0 Problem Formulation</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                We aim to learn two conditional diffusion "transport" models between Control (c) and Treated (t) states, conditioned on a drug descriptor (d):
                            </p>
                            <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', marginBottom: '24px' }}>
                                <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight }}>
                                    <li><strong>Forward / Treatment Model (<Latex>{String.raw`$\theta$`}</Latex>):</strong> <Latex>{String.raw`$p_\theta(t \mid c, d)$`}</Latex></li>
                                    <li><strong>Backward / Recovery Model (<Latex>{String.raw`$\phi$`}</Latex>):</strong> <Latex>{String.raw`$p_\phi(c \mid t, d)$`}</Latex></li>
                                </ul>
                            </div>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                The goals are: (1) realistic generation in target domains, (2) mutual consistency (cycle consistency), and (3) minimal drift from pretrained unconditional marginals.
                            </p>

                            {/* 9.1 Dataset */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.1 Dataset (Mathematical View)</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                The dataset returns triples <Latex>{String.raw`$(c, t, d)$`}</Latex> sampled from the empirical distribution <Latex>{String.raw`$\hat p_{\text{data}}(c,t,d)$`}</Latex>, where:
                            </p>
                            <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <li><Latex>{String.raw`$c \sim p_{\text{ctrl}}$`}</Latex>: Control (DMSO) image.</li>
                                <li><Latex>{String.raw`$t \sim p_{\text{trt}}$`}</Latex>: Treated image (from same batch).</li>
                                <li><Latex>{String.raw`$d \in \mathbb{R}^{1024}$`}</Latex>: Morgan fingerprint (<Latex>{String.raw`$d=\mathbf{0}$`}</Latex> for DMSO).</li>
                            </ul>

                            {/* 9.2 Conditional Diffusion */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.2 Conditional Diffusion Model Definition</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                <strong>Forward Process (DDPM):</strong>
                                <br />
                                <Latex>{String.raw`$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, \quad \epsilon\sim\mathcal{N}(0,I)$`}</Latex>
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                <strong>Network Parameterization:</strong>
                                <br />
                                The UNet predicts noise <Latex>{String.raw`$\epsilon_\theta(x_t, t, \text{cond\_img}, d)$`}</Latex> by concatenating <Latex>{String.raw`$[x_t, \text{cond\_img}]$`}</Latex> (6 channels) and injecting <Latex>{String.raw`$d$`}</Latex> as a class embedding.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <strong>Supervised Loss:</strong>
                                <br />
                                <Latex>{String.raw`$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\epsilon}\left[ |\epsilon - \epsilon_\theta(x_t,t,\text{cond},d)|^2 \right]$`}</Latex>
                            </p>

                            {/* 9.3 Pretraining */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.3 Pretraining Initialization ("Warm Start")</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                To avoid learning from scratch, we initialize weights from unconditional models trained on the target marginals:
                                <br />
                                <Latex>{String.raw`$p_{\theta_0}(t) \approx p_{\text{trt}}(t), \quad p_{\phi_0}(c) \approx p_{\text{ctrl}}(c)$`}</Latex>
                                <br />
                                Conditioning channels are zero-initialized to preserve this pretrained behavior initially.
                            </p>

                            {/* 9.4 Rollout */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.4 Rollout Distribution (Policy View)</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                Reverse diffusion is treated as a stochastic policy trajectory <Latex>{String.raw`$x_T \to \dots \to x_0$`}</Latex>. The transition probability is Gaussian:
                                <br />
                                <Latex>{String.raw`$p_\theta(x_{t'} \mid x_t, c, d) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$`}</Latex>
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                For PPO, we compute the log-probability <Latex>{String.raw`$\log \pi_\theta$`}</Latex> of each step during rollout to estimate the policy gradient.
                            </p>

                            {/* 9.5 Reward */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.5 Reward: DDMEC Likelihood Estimator</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                We score a generated sample <Latex>{String.raw`$x_{\text{gen}}$`}</Latex> using the <strong>partner model</strong> as a critic. For <Latex>{String.raw`$\theta$`}</Latex>, the reward is the log-likelihood of recovering the original control <Latex>{String.raw`$c$`}</Latex>:
                                <br />
                                <Latex>{String.raw`$r_\theta = \log p_\phi(c \mid x_{\text{gen}}, d)$`}</Latex>
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                This is approximated via a weighted MSE denoising loss:
                                <br />
                                <Latex>{String.raw`$r_\theta \approx -\frac{1}{n}\sum \text{snr}(t_k) \cdot |\epsilon - \hat\epsilon_\phi|^2$`}</Latex>
                                <br />
                                A higher reward implies <Latex>{String.raw`$x_{\text{gen}}$`}</Latex> is consistent with <Latex>{String.raw`$\phi$`}</Latex>'s learned mapping.
                            </p>

                            {/* 9.6 PPO Objective */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.6 PPO Objective & KL Constraint</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '12px' }}>
                                We optimize the PPO clipped surrogate objective with a KL penalty to a frozen reference model <Latex>{String.raw`$\theta_{\text{ref}}$`}</Latex>:
                            </p>
                            <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '8px', marginBottom: '24px' }}>
                                <p style={{ textAlign: 'center', marginBottom: '12px' }}>
                                    <Latex>{String.raw`$\mathcal{L}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) + \beta \mathcal{L}_{\text{KL}}(\theta)$`}</Latex>
                                </p>
                                <p style={{ fontSize: '13px', color: colors.textLight }}>
                                    Where <Latex>{String.raw`$\mathcal{L}_{\text{KL}}$`}</Latex> penalizes drift from the pre-trained distribution, preventing mode collapse.
                                </p>
                            </div>

                            {/* 9.7 Joint Constraints */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.7 Joint Constraint Phases</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                To enforce cycle consistency, we alternate PPO updates with <strong>joint diffusion regression</strong>. For example, after <Latex>{String.raw`$\theta$`}</Latex> generates <Latex>{String.raw`$\hat{t}$`}</Latex>, we update <Latex>{String.raw`$\phi$`}</Latex> to reconstruct <Latex>{String.raw`$c$`}</Latex> given <Latex>{String.raw`$\hat{t}$`}</Latex>:
                                <br />
                                <Latex>{String.raw`$\min_\phi \mathcal{L}_{\text{DDPM}}(\phi; x_0=c, \text{cond}=\hat{t}, d)$`}</Latex>
                                <br />
                                This "bootstraps" the two models against each other.
                            </p>

                            {/* 9.8 CFG */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.8 Note on Classifier-Free Guidance</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                During PPO training, we enforce <code>guidance_scale = 1.0</code> because PPO requires the exact path probability of the model's policy. CFG modifies the drift, which would mismatch the computed log-probabilities.
                            </p>

                            {/* 9.9 Evaluation */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.9 Evaluation Metrics</h4>
                            <ul style={{ listStyleType: 'disc', paddingLeft: '24px', lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                <li><strong>FID/KID:</strong> Compares overall distribution of generated vs. real images.</li>
                                <li><strong>cFID (Conditional FID):</strong> Computed per compound label (d) and averaged. Checks if the model preserves specific drug effects.</li>
                            </ul>

                            {/* 9.10 Summary */}
                            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: colors.text }}>9.10 Summary</h4>
                            <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                The training alternates between <strong>Policy Improvement</strong> (PPO maximizing cross-model likelihood) and <strong>Consistency</strong> (Supervised regression on generated samples). This ensures that <Latex>{String.raw`$\theta$`}</Latex> and <Latex>{String.raw`$\phi$`}</Latex> evolve together to form a coherent bidirectional mapping.
                            </p>
                        </Section>

                        {/* 10) Results: PPO-DDMEC */}
                        <Section title="10. Results & Discussion: PPO-DDMEC Fine-tuning" number="10">
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '24px' }}>
                                The following plot visualizes the fine-tuning results using PPO-DDMEC.
                            </p>
                            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                                <div
                                    style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #e2e8f0', cursor: 'pointer', display: 'inline-block' }}
                                    onClick={() => setModalImage({ src: "/uncond/finetune/plots ddmec uncond ddpms.png", alt: "PPO-DDMEC Fine-tuning Results" })}
                                >
                                    <img
                                        src="/uncond/finetune/plots ddmec uncond ddpms.png"
                                        alt="PPO-DDMEC Fine-tuning Results"
                                        style={{ maxWidth: '100%', maxHeight: '600px', display: 'block' }}
                                    />
                                </div>
                                <p style={{ fontSize: '13px', color: colors.textLight, marginTop: '12px', fontStyle: 'italic' }}>
                                    Figure 10.1: PPO-DDMEC Fine-tuning Performance.
                                </p>
                            </div>

                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                The <strong>Fréchet Inception Distance (FID)</strong> was used to assess the quality of generated samples. Both models began with very high FID scores (~370–380), indicating poor initial generation quality. During the first 3,000 iterations, FID decreased rapidly, demonstrating effective learning. The <strong>Phi model</strong> improved faster, reaching ~190, while <strong>Theta</strong> remained around ~260.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                From iterations 4,000–8,000, <strong>Phi</strong> consistently outperformed Theta, stabilizing at ~90–140 compared to Theta’s ~140–180. In later training, both models showed fluctuations with occasional spikes, suggesting some instability in PPO optimization, but Phi maintained lower FID overall.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight, marginBottom: '16px' }}>
                                PPO losses were small with low KL divergence (~0.01–0.04), indicating conservative updates. The joint constraint loss plateaued around 0.05–0.1, coinciding with the stagnation in FID, implying a trade-off between constraint satisfaction and perceptual quality.
                            </p>
                            <p style={{ lineHeight: 1.8, color: colors.textLight }}>
                                Overall, training reduced FID by more than 60%, confirming substantial improvement in sample realism. However, oscillations and plateauing suggest that stronger reward signals or improved PPO regularization may be required for further gains. <strong>Phi</strong> demonstrated superior stability and final performance compared to Theta.
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
                                    We evaluate the model's performance using standard video evolution checks, epoch-based image sampling, and quantitative metrics.
                                </p>

                                {/* Metrics Table */}
                                <div style={{ marginBottom: '32px' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0', marginBottom: '16px' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Metric</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', fontWeight: 600 }}>FID ↓</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.primary }}>66.49</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px', fontWeight: 600 }}>CFID ↓</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.primary }}>132.79</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                    <p style={{ fontSize: '12px', color: colors.textLight, fontStyle: 'italic', lineHeight: 1.6 }}>
                                        <strong>Note:</strong> FID (Fréchet Inception Distance) measures overall image quality, and CFID (Conditional FID) evaluates conditional alignment between control and treated states. Lower values indicate better performance.
                                    </p>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                                    <div>
                                        <h5 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '12px', color: colors.textLight }}>Prediction Video</h5>
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
                                    We evaluate the model's performance using quantitative metrics and inference visualization.
                                </p>

                                {/* Metrics Table */}
                                <div style={{ marginBottom: '32px' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', border: '1px solid #e2e8f0', marginBottom: '16px' }}>
                                        <thead>
                                            <tr style={{ background: '#f1f5f9' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>Metric</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                                                <td style={{ padding: '8px 12px', fontWeight: 600 }}>FID ↓</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.primary }}>77.49</td>
                                            </tr>
                                            <tr>
                                                <td style={{ padding: '8px 12px', fontWeight: 600 }}>CFID ↓</td>
                                                <td style={{ padding: '8px 12px', textAlign: 'center', color: colors.primary }}>152.79</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                    <p style={{ fontSize: '12px', color: colors.textLight, fontStyle: 'italic', lineHeight: 1.6 }}>
                                        <strong>Note:</strong> FID (Fréchet Inception Distance) measures overall image quality, and CFID (Conditional FID) evaluates conditional alignment between control and treated states. Lower values indicate better performance.
                                    </p>
                                </div>

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

                {/* References Section */}
                <Section title="References" number="10">
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
                </Section>

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
