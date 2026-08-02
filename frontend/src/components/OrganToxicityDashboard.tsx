import React from 'react';
import { OrganType, ToxicityPredictionResult } from '../types';
import { GTEX_ORGAN_PROFILES } from '../utils/siderDataset';
import { Activity, Heart, Brain, Droplet, Wind, Flame } from 'lucide-react';

interface OrganToxicityDashboardProps {
  prediction: ToxicityPredictionResult;
  useTissueConditioning: boolean;
}

export const OrganToxicityDashboard: React.FC<OrganToxicityDashboardProps> = ({
  prediction,
  useTissueConditioning
}) => {
  const organMeta: Record<OrganType, { name: string; icon: React.ReactNode; color: string; desc: string }> = {
    liver: {
      name: 'Liver (Hepatic)',
      icon: <Droplet className="w-5 h-5 text-amber-400" />,
      color: 'from-amber-500 to-orange-600',
      desc: 'CYP3A4 / Phase I Metabolism & Drug-Induced Liver Injury (DILI)'
    },
    heart: {
      name: 'Heart (Cardiac)',
      icon: <Heart className="w-5 h-5 text-rose-400" />,
      color: 'from-rose-500 to-red-600',
      desc: 'hERG Potassium Channel Blocking & QT Prolongation / Arrhythmia'
    },
    kidney: {
      name: 'Kidney (Renal)',
      icon: <Activity className="w-5 h-5 text-sky-400" />,
      color: 'from-sky-500 to-indigo-600',
      desc: 'Organic Anion/Cation Transporter Inhibition & Tubular Necrosis'
    },
    brain: {
      name: 'Brain (Central Nervous)',
      icon: <Brain className="w-5 h-5 text-purple-400" />,
      color: 'from-purple-500 to-pink-600',
      desc: 'BBB Transcellular Permeability & Neuroreceptor Off-Target Binding'
    },
    lung: {
      name: 'Lung (Pulmonary)',
      icon: <Wind className="w-5 h-5 text-teal-400" />,
      color: 'from-teal-500 to-emerald-600',
      desc: 'Alveolar Epithelium Surfactant Disruption & Interstitial Pneumonitis'
    }
  };

  const organs: OrganType[] = ['liver', 'heart', 'kidney', 'brain', 'lung'];

  return (
    <div className="space-y-6">
      
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs bg-indigo-950 text-indigo-300 border border-indigo-700/60 font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider">
              {useTissueConditioning ? '16-Head Cross-Attention GTEx V8 Active' : 'Molecule-Only Baseline'}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              MC Dropout Passes: N=30
            </span>
          </div>
          <h2 className="text-xl font-bold text-white mt-1">
            Organ-Specific Zero-Shot Toxicity Safety Profile
          </h2>
          <p className="text-xs text-slate-400">
            Predicting multi-organ toxicity disaggregated across human gene expression microenvironments.
          </p>
        </div>

        {/* Tanimoto Domain Summary */}
        <div className={`px-4 py-2 rounded-xl border text-right ${
          prediction.domainColor === 'green'
            ? 'bg-emerald-950/40 border-emerald-700 text-emerald-300'
            : prediction.domainColor === 'yellow'
            ? 'bg-amber-950/40 border-amber-700 text-amber-300'
            : 'bg-rose-950/40 border-rose-700 text-rose-300'
        }`}>
          <span className="text-[10px] uppercase font-mono tracking-wider font-bold block">
            Applicability Domain
          </span>
          <span className="text-sm font-bold block">
            {prediction.applicabilityDomain}
          </span>
          <span className="text-xs font-mono opacity-80 block">
            Tanimoto Score: {prediction.tanimotoSimilarity}
          </span>
        </div>
      </div>

      {/* 5 Organ Toxicity Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {organs.map((organKey) => {
          const score = prediction.organScores[organKey];
          const meta = organMeta[organKey];
          const gtexProfile = GTEX_ORGAN_PROFILES[organKey];

          let badgeColor = 'bg-emerald-950 text-emerald-300 border-emerald-800';
          if (score.riskLevel === 'Severe') badgeColor = 'bg-rose-950 text-rose-300 border-rose-800 animate-pulse';
          else if (score.riskLevel === 'High') badgeColor = 'bg-amber-950 text-amber-300 border-amber-800';
          else if (score.riskLevel === 'Moderate') badgeColor = 'bg-sky-950 text-sky-300 border-sky-800';

          return (
            <div
              key={organKey}
              className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 shadow-sm space-y-4 transition-all"
            >
              {/* Card Header */}
              <div className="flex justify-between items-start">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-xl">
                    {meta.icon}
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{meta.name}</h3>
                    <p className="text-[11px] text-slate-400 leading-snug">{meta.desc}</p>
                  </div>
                </div>
              </div>

              {/* Risk Score Progress Bar */}
              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400 font-medium">Mean Risk Score (μ)</span>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${badgeColor}`}>
                      {score.riskLevel} Risk
                    </span>
                    <span className="text-base font-extrabold font-mono text-white">
                      {(score.meanRisk * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden border border-slate-800 p-0.5">
                  <div
                    className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${meta.color}`}
                    style={{ width: `${Math.max(5, score.meanRisk * 100)}%` }}
                  ></div>
                </div>

                <div className="flex justify-between items-center text-[11px] text-slate-500 font-mono">
                  <span>Bayesian Uncertainty (σ): ±{score.uncertaintySigma}</span>
                  <span>{score.meanRisk < 0.3 ? 'Safe' : score.meanRisk < 0.6 ? 'Monitor' : 'Toxic'}</span>
                </div>
              </div>

              {/* GTEx Gene Pathway Markers */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-slate-400">GTEx Marker Expression</span>
                  <span className="text-[10px] text-indigo-400 font-mono">128-dim Vector</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {gtexProfile.markerGenes.map((gene) => (
                    <span
                      key={gene}
                      className="px-1.5 py-0.5 bg-slate-900 border border-slate-800 text-[10px] font-mono text-slate-300 rounded"
                    >
                      {gene}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* MedDRA Clinical Toxicity Classes Overview */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <Flame className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">10 MedDRA Clinical Toxicity Classes</h3>
          </div>
          <span className="text-xs text-slate-400">
            Standardized Regulatory Safety Classifications
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          {Object.entries(prediction.meddraScores).map(([cls, val]) => {
            const label = cls.replace('_', ' ').toUpperCase();
            const numVal = Number(val);
            return (
              <div key={cls} className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-center space-y-1">
                <span className="text-[10px] font-mono font-bold text-slate-400 block truncate" title={label}>
                  {label}
                </span>
                <span className={`text-sm font-extrabold font-mono block ${
                  numVal >= 0.6 ? 'text-rose-400' : numVal >= 0.3 ? 'text-amber-400' : 'text-emerald-400'
                }`}>
                  {(numVal * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
};
