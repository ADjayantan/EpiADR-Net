import React from 'react';
import { ToxicityPredictionResult } from '../types';
import { Database, ShieldCheck, AlertOctagon, HelpCircle } from 'lucide-react';

interface TanimotoDomainCardProps {
  prediction: ToxicityPredictionResult;
}

export const TanimotoDomainCard: React.FC<TanimotoDomainCardProps> = ({ prediction }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Database className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-white">Tanimoto Structural Applicability Domain</h3>
        </div>
        <span className="text-xs font-mono text-slate-400">Morgan Fingerprint (128-bit)</span>
      </div>

      <div className="flex flex-col md:flex-row items-start md:items-center justify-between bg-slate-950 p-4 rounded-xl border border-slate-800 gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className={`w-3 h-3 rounded-full ${
              prediction.domainColor === 'green' ? 'bg-emerald-500' : prediction.domainColor === 'yellow' ? 'bg-amber-500' : 'bg-rose-500'
            }`}></span>
            <span className="text-base font-extrabold text-white">
              {prediction.applicabilityDomain}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            {prediction.domainColor === 'green'
              ? 'High chemical similarity to SIDER 4.1 training manifold (Tanimoto ≥ 0.70). Predictions highly reliable.'
              : prediction.domainColor === 'yellow'
              ? 'Moderate structural overlap (0.40 ≤ Tanimoto < 0.70). Check Bayesian uncertainty σ.'
              : 'Novel chemical scaffold (Tanimoto < 0.40). Treat zero-shot organ prediction as exploratory.'}
          </p>
        </div>

        <div className="bg-slate-900 px-4 py-3 rounded-xl border border-slate-800 text-center font-mono space-y-0.5">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Tanimoto Score</span>
          <span className="text-xl font-extrabold text-indigo-300 block">{prediction.tanimotoSimilarity}</span>
        </div>
      </div>
    </div>
  );
};
