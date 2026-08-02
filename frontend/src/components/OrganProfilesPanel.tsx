import React, { useState } from 'react';
import { OrganType } from '../types';
import { GTEX_ORGAN_PROFILES } from '../utils/siderDataset';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { Database, Dna, Activity, Cpu, Layers, Info, BarChart2 } from 'lucide-react';

export const OrganProfilesPanel: React.FC = () => {
  const [selectedOrgan, setSelectedOrgan] = useState<OrganType>('liver');

  const profile = GTEX_ORGAN_PROFILES[selectedOrgan];

  // Map 128-dim gene expression vector to 20 sampled feature buckets for clean visualization
  const sampledExpressionData = profile.geneExpressionValues.slice(0, 20).map((val, idx) => ({
    geneDim: `Dim ${idx + 1}`,
    expression: Number(val.toFixed(2))
  }));

  const organList: { id: OrganType; icon: string; title: string }[] = [
    { id: 'liver', icon: '🫀', title: 'Liver (Hepatic)' },
    { id: 'heart', icon: '❤️', title: 'Heart (Cardiac)' },
    { id: 'kidney', icon: '🩺', title: 'Kidney (Renal)' },
    { id: 'brain', icon: '🧠', title: 'Brain (Central Nervous)' },
    { id: 'lung', icon: '🫁', title: 'Lung (Pulmonary)' },
  ];

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <Dna className="w-5 h-5 text-indigo-400" />
          <div>
            <h2 className="text-base font-bold text-white">GTEx V8 Human Organ Transcriptomic Profiles</h2>
            <p className="text-xs text-slate-400">RNA-Seq Gene Expression Profiles across 5 Primary Human Organs</p>
          </div>
        </div>

        {/* Organ Selector Chips */}
        <div className="flex overflow-x-auto space-x-2 pt-1 no-scrollbar">
          {organList.map((item) => (
            <button
              key={item.id}
              onClick={() => setSelectedOrgan(item.id)}
              className={`whitespace-nowrap px-4 py-2.5 rounded-xl text-xs font-bold transition-all border flex items-center space-x-2 ${
                selectedOrgan === item.id
                  ? 'bg-indigo-600 text-white border-indigo-400 shadow-lg shadow-indigo-600/30'
                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-800'
              }`}
            >
              <span>{item.icon}</span>
              <span>{item.title}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Organ Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Marker Genes & Details */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-indigo-600/20 text-indigo-300 rounded-2xl border border-indigo-500/30 text-xl">
              {selectedOrgan === 'liver' ? '🫀' : selectedOrgan === 'heart' ? '❤️' : selectedOrgan === 'kidney' ? '🩺' : selectedOrgan === 'brain' ? '🧠' : '🫁'}
            </div>
            <div>
              <h3 className="text-base font-bold text-white">{profile.name}</h3>
              <span className="text-xs text-indigo-400 font-mono font-semibold">GTEx V8 ID: gtex_{profile.id}_v8</span>
            </div>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed bg-slate-950 p-3 rounded-xl border border-slate-800/80">
            {profile.description}
          </p>

          {/* Key Marker Genes */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-slate-300 block">Top GTEx Marker & Target Genes:</span>
            <div className="flex flex-wrap gap-1.5">
              {profile.markerGenes.map((gene, idx) => (
                <span
                  key={idx}
                  className="text-xs font-mono font-bold bg-indigo-950/80 text-indigo-300 border border-indigo-800/80 px-2.5 py-1 rounded-lg"
                >
                  {gene}
                </span>
              ))}
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1 text-xs">
            <span className="font-bold text-emerald-400 block">16-Head Cross-Attention Mechanism:</span>
            <p className="text-slate-400 text-[11px] leading-normal">
              EpiADR-Net projects the 128-dimensional expression vector of this organ into key/value attention heads, cross-querying SMILES atomic feature embeddings to predict tissue-conditioned toxicity.
            </p>
          </div>

        </div>

        {/* Right Column: 128-dim Expression Vector Visualizer */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2">
              <BarChart2 className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white">RNA-Seq Expression Vector Distribution</h3>
            </div>
            <span className="text-xs text-slate-400 font-mono">128-Dim Vector</span>
          </div>

          <div className="h-64 bg-slate-950 p-3 rounded-xl border border-slate-800">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sampledExpressionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="geneDim" stroke="#64748b" fontSize={9} />
                <YAxis stroke="#64748b" fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Bar dataKey="expression" fill="#6366f1" radius={[4, 4, 0, 0]} name="Log2 (TPM + 1)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="text-[11px] text-slate-400 text-center font-mono">
            Showing sampled dimensions 1-20 of 128-dim normalized GTEx V8 transcriptomic organ expression vector.
          </p>
        </div>

      </div>

    </div>
  );
};
