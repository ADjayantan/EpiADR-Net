import React, { useState } from 'react';
import { DrugRecord, ToxicityPredictionResult } from '../types';
import { SIDER_BENCHMARK_DRUGS } from '../utils/siderDataset';
import { globalEpiADREngine } from '../utils/epiAdrEngine';
import { MoleculeVisualizer } from './MoleculeVisualizer';
import { Activity, AlertTriangle, CheckCircle, Database, HelpCircle, Info, ShieldAlert, Sparkles, Zap } from 'lucide-react';

interface PredictionPanelProps {
  useTissueConditioning: boolean;
  onOpenAiReportWithData: (data: any) => void;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  useTissueConditioning,
  onOpenAiReportWithData
}) => {
  const [selectedDrugId, setSelectedDrugId] = useState<string>('doxorubicin');
  const [customSmiles, setCustomSmiles] = useState<string>('');
  const [compoundName, setCompoundName] = useState<string>('Doxorubicin');

  // Select active drug record
  const activeDrug = SIDER_BENCHMARK_DRUGS.find(d => d.id === selectedDrugId) || SIDER_BENCHMARK_DRUGS[0];
  const activeSmiles = customSmiles.trim() !== '' ? customSmiles : activeDrug.smiles;

  // Run prediction using EpiADR engine
  const prediction: ToxicityPredictionResult = globalEpiADREngine.predictCompoundToxicity(
    customSmiles.trim() !== '' ? 'Custom Molecule Candidate' : activeDrug.name,
    activeSmiles,
    useTissueConditioning
  );

  const handleSelectDrug = (drug: DrugRecord) => {
    setSelectedDrugId(drug.id);
    setCompoundName(drug.name);
    setCustomSmiles('');
  };

  const getRiskBadgeColor = (level: string) => {
    switch (level) {
      case 'Severe': return 'bg-rose-500/20 text-rose-300 border-rose-500/50';
      case 'High': return 'bg-amber-500/20 text-amber-300 border-amber-500/50';
      case 'Moderate': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50';
      default: return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50';
    }
  };

  const getDomainColor = (color: string) => {
    if (color === 'green') return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/40';
    if (color === 'yellow') return 'bg-amber-500/10 text-amber-400 border-amber-500/40';
    return 'bg-rose-500/10 text-rose-400 border-rose-500/40';
  };

  return (
    <div className="space-y-6">
      
      {/* Molecule Input & Preset Selector */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-white">SIDER 4.1 Benchmark Compound Library</h2>
          </div>
          <span className="text-xs bg-slate-950 text-slate-400 px-3 py-1 rounded-full border border-slate-800 font-mono">
            7,325 Training SMILES Records
          </span>
        </div>

        {/* Drug Selection Chips */}
        <div className="flex overflow-x-auto space-x-2 pb-1 no-scrollbar">
          {SIDER_BENCHMARK_DRUGS.map((drug) => {
            const isSelected = selectedDrugId === drug.id && customSmiles === '';
            return (
              <button
                key={drug.id}
                onClick={() => handleSelectDrug(drug)}
                className={`whitespace-nowrap px-3 py-2 rounded-xl text-xs font-bold transition-all border ${
                  isSelected
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow-md shadow-indigo-600/30'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:bg-slate-800'
                }`}
              >
                {drug.name}
              </button>
            );
          })}
        </div>

        {/* Custom SMILES Input */}
        <div className="pt-2 border-t border-slate-800 space-y-1.5">
          <label className="text-xs font-semibold text-slate-400 block">
            Or Paste Custom SMILES String:
          </label>
          <div className="flex space-x-2">
            <input
              type="text"
              value={customSmiles}
              onChange={(e) => setCustomSmiles(e.target.value)}
              placeholder="e.g. CC(=O)NC1=CC=C(O)C=C1 or C1=CC=CC=C1"
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs font-mono text-indigo-300 focus:outline-none focus:border-indigo-500"
            />
            {customSmiles !== '' && (
              <button
                onClick={() => setCustomSmiles('')}
                className="px-3 py-2 bg-slate-800 text-slate-300 rounded-xl text-xs font-bold hover:bg-slate-700"
              >
                Reset
              </button>
            )}
          </div>
        </div>

      </div>

      {/* Tanimoto Structural Applicability Domain Banner */}
      <div className={`p-4 rounded-2xl border flex flex-wrap items-center justify-between gap-3 shadow-md ${getDomainColor(prediction.domainColor)}`}>
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-slate-950/80 rounded-xl border border-slate-800">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xs font-bold uppercase tracking-wider">Tanimoto Applicability Domain</span>
              <span className="text-xs font-extrabold px-2 py-0.5 rounded-full bg-slate-950 border border-current">
                Score: {prediction.tanimotoSimilarity}
              </span>
            </div>
            <p className="text-xs font-semibold mt-0.5">{prediction.applicabilityDomain}</p>
          </div>
        </div>

        <div className="text-xs text-right font-mono text-slate-300">
          <span className="block">Morgan 128-bit Count Similarity</span>
          <span className="text-[11px] opacity-80">Reference Set: SIDER 4.1</span>
        </div>
      </div>

      {/* Main Grid: Molecular XAI + Organ Toxicity Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: 2D Molecular Hotspot Visualizer */}
        <div className="lg:col-span-5 space-y-4">
          <MoleculeVisualizer
            smiles={activeSmiles}
            compoundName={prediction.compoundName}
            hotspots={prediction.atomicHotspots}
            toxicophores={prediction.toxicophores}
          />

          <button
            onClick={() => onOpenAiReportWithData({
              compoundName: prediction.compoundName,
              smiles: activeSmiles,
              organScores: {
                liver: prediction.organScores.liver.meanRisk,
                heart: prediction.organScores.heart.meanRisk,
                kidney: prediction.organScores.kidney.meanRisk,
                brain: prediction.organScores.brain.meanRisk,
                lung: prediction.organScores.lung.meanRisk
              },
              tanimoto: prediction.tanimotoSimilarity,
              mcUncertainty: 0.03,
              useTissueConditioning
            })}
            className="w-full py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:opacity-90 text-white rounded-2xl font-bold text-xs shadow-xl flex items-center justify-center space-x-2 transition-all active:scale-95"
          >
            <Sparkles className="w-4 h-4 animate-spin-slow" />
            <span>Generate Gemini Preclinical Safety Report</span>
          </button>
        </div>

        {/* Right Column: 5 Primary Organ Zero-Shot Risk Cards */}
        <div className="lg:col-span-7 space-y-4">
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white">5 Primary Organ Zero-Shot Toxicity Scores</h3>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Bayesian MC Dropout (N=30, μ ± σ)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            
            {/* Liver Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 hover:border-slate-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-slate-200">🫀 Hepatotoxicity (Liver)</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getRiskBadgeColor(prediction.organScores.liver.riskLevel)}`}>
                  {prediction.organScores.liver.riskLevel} Risk
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Mean Risk Score (μ):</span>
                  <span className="font-bold text-indigo-300">{(prediction.organScores.liver.meanRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 h-full transition-all duration-500"
                    style={{ width: `${prediction.organScores.liver.meanRisk * 100}%` }}
                  ></div>
                </div>
                <span className="text-[10px] text-slate-500 font-mono block">Uncertainty σ = ±{prediction.organScores.liver.uncertaintySigma}</span>
              </div>
            </div>

            {/* Heart Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 hover:border-slate-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-slate-200">❤️ Cardiotoxicity (Heart)</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getRiskBadgeColor(prediction.organScores.heart.riskLevel)}`}>
                  {prediction.organScores.heart.riskLevel} Risk
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Mean Risk Score (μ):</span>
                  <span className="font-bold text-indigo-300">{(prediction.organScores.heart.meanRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 h-full transition-all duration-500"
                    style={{ width: `${prediction.organScores.heart.meanRisk * 100}%` }}
                  ></div>
                </div>
                <span className="text-[10px] text-slate-500 font-mono block">Uncertainty σ = ±{prediction.organScores.heart.uncertaintySigma}</span>
              </div>
            </div>

            {/* Kidney Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 hover:border-slate-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-slate-200">🩺 Nephrotoxicity (Kidney)</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getRiskBadgeColor(prediction.organScores.kidney.riskLevel)}`}>
                  {prediction.organScores.kidney.riskLevel} Risk
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Mean Risk Score (μ):</span>
                  <span className="font-bold text-indigo-300">{(prediction.organScores.kidney.meanRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 h-full transition-all duration-500"
                    style={{ width: `${prediction.organScores.kidney.meanRisk * 100}%` }}
                  ></div>
                </div>
                <span className="text-[10px] text-slate-500 font-mono block">Uncertainty σ = ±{prediction.organScores.kidney.uncertaintySigma}</span>
              </div>
            </div>

            {/* Brain Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 hover:border-slate-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-slate-200">🧠 Neurotoxicity (Brain)</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getRiskBadgeColor(prediction.organScores.brain.riskLevel)}`}>
                  {prediction.organScores.brain.riskLevel} Risk
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Mean Risk Score (μ):</span>
                  <span className="font-bold text-indigo-300">{(prediction.organScores.brain.meanRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 h-full transition-all duration-500"
                    style={{ width: `${prediction.organScores.brain.meanRisk * 100}%` }}
                  ></div>
                </div>
                <span className="text-[10px] text-slate-500 font-mono block">Uncertainty σ = ±{prediction.organScores.brain.uncertaintySigma}</span>
              </div>
            </div>

            {/* Lung Card (Full Span on small screens) */}
            <div className="sm:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3 hover:border-slate-700 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-extrabold text-slate-200">🫁 Pulmotoxicity (Lung)</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getRiskBadgeColor(prediction.organScores.lung.riskLevel)}`}>
                  {prediction.organScores.lung.riskLevel} Risk
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Mean Risk Score (μ):</span>
                  <span className="font-bold text-indigo-300">{(prediction.organScores.lung.meanRisk * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div
                    className="bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500 h-full transition-all duration-500"
                    style={{ width: `${prediction.organScores.lung.meanRisk * 100}%` }}
                  ></div>
                </div>
                <span className="text-[10px] text-slate-500 font-mono block">Uncertainty σ = ±{prediction.organScores.lung.uncertaintySigma}</span>
              </div>
            </div>

          </div>

          {/* MedDRA 10 Clinical Classes Breakdown */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-3">
            <h4 className="text-xs font-bold text-slate-300">MedDRA Clinical Side Effect Classes (10 Categories)</h4>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs font-mono">
              {Object.entries(prediction.meddraScores).map(([key, val]) => (
                <div key={key} className="bg-slate-950 p-2 rounded-xl border border-slate-800">
                  <span className="text-[10px] text-slate-400 block truncate capitalize">{key.replace('_', ' ')}</span>
                  <span className={`font-bold ${val > 0.6 ? 'text-rose-400' : val > 0.35 ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {(val * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};
