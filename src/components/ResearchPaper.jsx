import React, { useState } from 'react';
import PPOComparison from './PPOComparison';
import BiologicalExperiments from './BiologicalExperiments';
import SyntheticExperiments from './SyntheticExperiments';

const colors = {
  primary: '#0f766e',
  text: '#1e293b',
  textLight: '#64748b',
  bgUser: '#f1f5f9'
};

const TabButton = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    style={{
      padding: '12px 24px',
      fontSize: '14px',
      fontWeight: 600,
      border: 'none',
      background: active ? 'white' : 'transparent',
      color: active ? colors.primary : colors.textLight,
      cursor: 'pointer',
      borderRadius: '8px 8px 0 0',
      borderBottom: active ? `2px solid ${colors.primary}` : '2px solid transparent',
      transition: 'all 0.2s ease'
    }}
  >
    {children}
  </button>
);

const ResearchPaper = () => {
  const [activeTab, setActiveTab] = useState('biological');

  return (
    <div style={{ fontFamily: "'IBM Plex Sans', sans-serif", backgroundColor: '#f1f5f9', minHeight: '100vh', color: colors.text }}>
      {/* Header - Common for all tabs */}
      <header style={{ background: 'linear-gradient(135deg, #0f766e 0%, #134e4a 100%)', padding: '60px 24px', color: 'white' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          <div style={{ display: 'inline-block', background: 'rgba(255,255,255,0.15)', padding: '6px 16px', borderRadius: '20px', fontSize: '13px', marginBottom: '20px' }}>
            ICML 2025 Workshop · Computational Biology
          </div>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, lineHeight: 1.2, marginBottom: '24px' }}>
            Evolution Strategies vs PPO for Cellular Morphology Prediction: BBBC021 Study
          </h1>
          <p style={{ fontSize: '1.1rem', opacity: 0.9, maxWidth: '700px', lineHeight: 1.6 }}>
            Diffusion-based minimum entropy coupling for drug-induced cellular morphology prediction
          </p>
          <div style={{ marginTop: '32px', display: 'flex', gap: '24px', flexWrap: 'wrap', fontSize: '14px', opacity: 0.85 }}>
            <span>📊 97,504 images</span>
            <span>💊 35 compounds</span>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '900px', margin: '0 auto', padding: '48px 24px' }}>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid #cbd5e1', marginBottom: '0', paddingLeft: '8px' }}>
          {/* <TabButton
            active={activeTab === 'ppo'}
            onClick={() => setActiveTab('ppo')}
          >
            PPO vs ES Comparison
          </TabButton> */}
          <TabButton
            active={activeTab === 'biological'}
            onClick={() => setActiveTab('biological')}
          >
            Biological Experiments
          </TabButton>
          <TabButton
            active={activeTab === 'synthetic'}
            onClick={() => setActiveTab('synthetic')}
          >
            Synthetic Experiments
          </TabButton>
        </div>

        {/* Tab Content Container */}
        <div style={{ minHeight: '400px', background: 'white', borderRadius: '0 0 16px 16px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)' }}>

          {activeTab === 'ppo' && <PPOComparison />}

          {activeTab === 'biological' && <BiologicalExperiments />}

          {activeTab === 'synthetic' && (
            <SyntheticExperiments />
          )}

        </div>

        {/* Footer */}
        <footer style={{ textAlign: 'center', padding: '32px', color: colors.textLight, fontSize: '14px' }}>
          <p><strong style={{ color: colors.text }}>Dataset:</strong> BBBC021 (97,504 images, 35 compounds)</p>
          <p><strong style={{ color: colors.text }}>Splits:</strong> 87,716 train / 9,788 test (batch-disjoint)</p>
        </footer>

      </main>
    </div>
  );
};

export default ResearchPaper;