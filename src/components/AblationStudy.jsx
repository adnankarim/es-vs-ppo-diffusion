import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import ImageModal from './ImageModal';

const CONFIGS = {
  1: { ppo: 8, es: 0 },
  2: { ppo: 35, es: 1 },
  5: { ppo: 40, es: 1 },
  10: { ppo: 36, es: 0 },
  20: { ppo: 33, es: 0 },
  30: { ppo: 21, es: 1 }
};

// Data from all_results.json (Verified Best Configs via script)
const METRICS = {
  1: {
    ppo: { score: 1.125, kl: 0.003, corr: 0.012, mae: 1.125, mi: 0.456 }, // Config 8
    es: { score: 1.122, kl: 0.001, corr: 0.013, mae: 1.122, mi: 0.463 }  // Config 0
  },
  2: {
    ppo: { score: 1.128, kl: 0.009, corr: 0.003, mae: 1.128, mi: 0.463 }, // Config 35
    es: { score: 1.128, kl: 0.001, corr: 0.002, mae: 1.128, mi: 0.483 }  // Config 1
  },
  5: {
    ppo: { score: 1.122, kl: 0.001, corr: 0.006, mae: 1.122, mi: 0.467 }, // Config 40
    es: { score: 1.116, kl: 0.001, corr: 0.007, mae: 1.116, mi: 0.486 }  // Config 1
  },
  10: {
    ppo: { score: 1.125, kl: 0.002, corr: 0.0004, mae: 1.125, mi: 0.478 }, // Config 36
    es: { score: 1.122, kl: 0.001, corr: -0.000, mae: 1.122, mi: 0.467 } // Config 0
  },
  20: {
    ppo: { score: 1.122, kl: 0.005, corr: 0.003, mae: 1.122, mi: 0.262 }, // Config 33
    es: { score: 1.120, kl: 0.001, corr: 0.001, mae: 1.120, mi: 0.256 }  // Config 0
  },
  30: {
    ppo: { score: 1.121, kl: 0.015, corr: 0.003, mae: 1.121, mi: 0.257 }, // Config 21
    es: { score: 1.122, kl: 0.002, corr: 0.003, mae: 1.122, mi: 0.254 }  // Config 1
  }
};

const AblationStudy = () => {
  const [activeDim, setActiveDim] = useState(10);
  const [modalImage, setModalImage] = useState(null);
  const dims = [1, 2, 5, 10, 20, 30];

  const ppoConfig = CONFIGS[activeDim].ppo;
  const esConfig = CONFIGS[activeDim].es;

  // Construct paths
  const ppoImg = `/ppo/plots/checkpoints_PPO_${activeDim}D_config_${ppoConfig}/epoch_010.png`;
  const esImg = `/es/plots/checkpoints_ES_${activeDim}D_config_${esConfig}/epoch_010.png`;

  const m = METRICS[activeDim];

  const chartData = [
    { name: 'Mutual Info', PPO: m.ppo.mi, ES: m.es.mi },
    { name: 'Score', PPO: m.ppo.score, ES: m.es.score },
    // MAE is scaled down to fit on chart or visualized separately
    // { name: 'MAE', PPO: m.ppo.mae, ES: m.es.mae } 
  ];

  return (
    <section style={{ background: 'white', borderRadius: '16px', padding: '32px', marginBottom: '32px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>


      {/* Dim Tabs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '24px' }}>
        {dims.map(dim => (
          <button
            key={dim}
            onClick={() => setActiveDim(dim)}
            style={{
              padding: '8px 16px',
              borderRadius: '20px',
              border: '1px solid #cbd5e1',
              background: activeDim === dim ? '#0f766e' : 'white',
              color: activeDim === dim ? 'white' : '#64748b',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            {dim}D
          </button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>

        {/* PPO Column */}
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '12px', overflow: 'hidden' }}>
          <div style={{ background: '#eff6ff', padding: '12px 16px', borderBottom: '1px solid #93c5fd', fontWeight: 600, color: '#1e40af', display: 'flex', justifyContent: 'space-between' }}>
            <span>PPO (Best Available: Config {ppoConfig})</span>
            <span>Epoch 10</span>
          </div>
          <div
            style={{ position: 'relative', aspectRatio: '1/1', background: '#f8fafc', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
            onClick={() => setModalImage({ src: ppoImg, alt: `PPO ${activeDim}D Result` })}
            title="Click to zoom"
          >
            <img
              src={ppoImg}
              alt={`PPO ${activeDim}D Result`}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentNode.innerHTML = '<div style="color:#94a3b8;padding:20px;text-align:center">Image not available<br/>(File missing)</div>';
              }}
            />
            <div style={{ position: 'absolute', bottom: '8px', right: '8px', background: 'rgba(255,255,255,0.8)', padding: '4px', borderRadius: '4px', fontSize: '12px' }}>🔍</div>
          </div>
          <div style={{ padding: '16px', fontSize: '13px', color: '#64748b' }}>
            <div><strong>Score:</strong> {m.ppo.score?.toFixed(3)}</div>
            <div><strong>MI:</strong> {m.ppo.mi?.toFixed(3)}</div>
            <div><strong>KL:</strong> {m.ppo.kl?.toFixed(4)}</div>
          </div>
        </div>

        {/* ES Column */}
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '12px', overflow: 'hidden' }}>
          <div style={{ background: '#fef2f2', padding: '12px 16px', borderBottom: '1px solid #fca5a5', fontWeight: 600, color: '#991b1b', display: 'flex', justifyContent: 'space-between' }}>
            <span>ES (Best Available: Config {esConfig})</span>
            <span>Epoch 10</span>
          </div>
          <div
            style={{ position: 'relative', aspectRatio: '1/1', background: '#f8fafc', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}
            onClick={() => setModalImage({ src: esImg, alt: `ES ${activeDim}D Result` })}
            title="Click to zoom"
          >
            <img
              src={esImg}
              alt={`ES ${activeDim}D Result`}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentNode.innerHTML = '<div style="color:#94a3b8;padding:20px;text-align:center">Image not available</div>';
              }}
            />
            <div style={{ position: 'absolute', bottom: '8px', right: '8px', background: 'rgba(255,255,255,0.8)', padding: '4px', borderRadius: '4px', fontSize: '12px' }}>🔍</div>
          </div>
          <div style={{ padding: '16px', fontSize: '13px', color: '#64748b' }}>
            <>
              <div><strong>Score:</strong> {m.es.score?.toFixed(3)}</div>
              <div><strong>MI:</strong> {m.es.mi?.toFixed(3)}</div>
              <div><strong>KL:</strong> {m.es.kl?.toFixed(4)}</div>
            </>
          </div>
        </div>
      </div>

      {/* Comparison Metrics Table */}
      <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#64748b', textTransform: 'uppercase' }}>Metrics Comparison ({activeDim})</h4>
      <div style={{ overflowX: 'auto', marginBottom: '24px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px', color: '#0f766e' }}>PPO</th>
              <th style={{ padding: '8px', color: '#db2777' }}>ES</th>
              <th style={{ padding: '8px', color: '#64748b' }}>Delta</th>
            </tr>
          </thead>
          <tbody>
            {[
              { label: 'MAE (Lower is Better)', key: 'mae', better: 'lower' },
              { label: 'KL Divergence', key: 'kl', better: 'lower' },
              { label: 'Mutual Info (MI)', key: 'mi', better: 'higher' },
            ].map((metric) => {
              const ppoVal = METRICS[activeDim].ppo[metric.key];
              const esVal = METRICS[activeDim].es[metric.key];
              const isPPOBetter = metric.better === 'lower' ? ppoVal < esVal : ppoVal > esVal;
              const isESBetter = metric.better === 'lower' ? esVal < ppoVal : esVal > ppoVal;

              const ppoColor = isPPOBetter ? '#16a34a' : '#1e293b'; // Green or Dark Slate
              const esColor = isESBetter ? '#16a34a' : '#1e293b';   // Green or Dark Slate
              const ppoWeight = isPPOBetter ? 700 : 400;
              const esWeight = isESBetter ? 700 : 400;

              return (
                <tr key={metric.key} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px', color: '#334155' }}>{metric.label}</td>
                  <td style={{ padding: '8px', color: ppoColor, fontWeight: ppoWeight }}>{ppoVal}</td>
                  <td style={{ padding: '8px', color: esColor, fontWeight: esWeight }}>{esVal}</td>
                  <td style={{ padding: '8px', color: '#64748b', fontSize: '13px' }}>
                    {metric.better === 'lower' ? (esVal - ppoVal).toFixed(3) : (ppoVal - esVal).toFixed(3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: '8px', padding: '16px', marginBottom: '24px', fontSize: '14px', color: '#0369a1' }}>
        <strong>Interpretation:</strong>
        {activeDim < 5 ?
          " In 1D, ES achieves slightly tighter convergence (MAE 1.122 vs 1.125) with lower KL divergence." :
          activeDim > 10 ?
            " In high dimensions (30D), PPO marginally outperforms ES (MAE 1.121 vs 1.122), showing better scalability despite higher KL cost." :
            " At intermediate dimensions (5D-10D), ES maintains a slight edge in MAE, but PPO demonstrates competitive mutual information alignment."
        }
      </div>

      <h4 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '12px', color: '#64748b', textTransform: 'uppercase' }}>MI & MAE Comparison</h4>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" domain={[1.0, 1.2]} label={{ value: 'MAE (Lower is Better)', position: 'insideBottom', offset: -5 }} stroke="#64748b" fontSize={12} />
          <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={12} width={80} />
          <Tooltip />
          <Legend />
          <Bar dataKey="PPO" fill="#0f766e" radius={[0, 4, 4, 0]} />
          <Bar dataKey="ES" fill="#dc2626" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>


      <ImageModal
        src={modalImage?.src}
        alt={modalImage?.alt}
        onClose={() => setModalImage(null)}
      />
    </section>
  );
};

export default AblationStudy;
