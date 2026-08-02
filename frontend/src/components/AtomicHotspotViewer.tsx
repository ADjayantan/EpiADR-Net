import React from 'react';
import { AtomicHotspot, ToxicityPredictionResult } from '../types';
import { Layers, Sparkles, AlertCircle, ShieldAlert, Eye } from 'lucide-react';

interface AtomicHotspotViewerProps {
  prediction: ToxicityPredictionResult;
}

export const AtomicHotspotViewer: React.FC<AtomicHotspotViewerProps> = ({ prediction }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center pb-3 border-b border-slate-800 gap-2">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-950 border border-indigo-800 rounded-xl">
            <Layers className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white">
              Explainable AI (XAI): Atomic Toxicity Hotspot Mapping
            </h2>
            <p className="text-xs text-slate-400">
              Graph Attention Weights α_ij highlight toxicophore functional groups contributing to organ risks.
            </p>
          </div>
        </div>

        <span className="text-xs font-mono bg-slate-950 border border-slate-800 text-indigo-300 px-3 py-1 rounded-lg">
          Query: {prediction.compoundName}
        </span>
      </div>

      {/* SMILES Interactive Chemical Hotspot Bar */}
      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
        <div className="flex justify-between items-center text-xs">
          <span className="font-semibold text-slate-400">Atomic Attention Heatmap (Graph Transformer Layer)</span>
          <span className="text-[11px] text-slate-500 font-mono">Higher Attention = Higher Toxicity Weight</span>
        </div>

        {/* Atom Badges */}
        <div className="flex flex-wrap gap-1.5 p-3 bg-slate-900 rounded-lg border border-slate-800/80">
          {prediction.atomicHotspots.map((atom) => {
            let bgClass = 'bg-slate-800 text-slate-300 border-slate-700';
            if (atom.attentionWeight >= 0.6) bgClass = 'bg-rose-950 text-rose-200 border-rose-700 font-bold animate-pulse';
            else if (atom.attentionWeight >= 0.4) bgClass = 'bg-amber-950 text-amber-200 border-amber-700';
            else if (atom.attentionWeight >= 0.25) bgClass = 'bg-indigo-950 text-indigo-200 border-indigo-700';

            return (
              <div
                key={atom.atomIndex}
                className={`px-2.5 py-1 rounded-md border text-xs font-mono flex items-center space-x-1 ${bgClass}`}
                title={`Atom #${atom.atomIndex} (${atom.symbol}): Attention Weight = ${atom.attentionWeight}`}
              >
                <span>{atom.symbol}</span>
                <span className="text-[9px] opacity-75">#{atom.atomIndex}</span>
                <span className="text-[10px] font-bold">({atom.attentionWeight})</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Identified Toxicophore Functional Groups */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-400" />
          <span>Structural Toxicophores Detected</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {prediction.toxicophores.map((tox, idx) => (
            <div key={idx} className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
              <span className="text-xs font-bold text-amber-300 block">{tox}</span>
              <p className="text-[11px] text-slate-400">
                Key contributor to phase I reactive metabolite generation or off-target ion channel block.
              </p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
