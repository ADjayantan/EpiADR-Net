import React, { useState } from 'react';
import { ToxicityPredictionResult } from '../types';
import { Sparkles, X, Bot, FileText, CheckCircle2, Loader2, ShieldAlert } from 'lucide-react';

interface GeminiSafetyModalProps {
  isOpen: boolean;
  onClose: () => void;
  prediction: ToxicityPredictionResult;
}

export const GeminiSafetyModal: React.FC<GeminiSafetyModalProps> = ({
  isOpen,
  onClose,
  prediction
}) => {
  const [reportText, setReportText] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleGenerateReport = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/ml-advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          compoundName: prediction.compoundName,
          smiles: prediction.smiles,
          organScores: {
            liver: prediction.organScores.liver.meanRisk,
            heart: prediction.organScores.heart.meanRisk,
            kidney: prediction.organScores.kidney.meanRisk,
            brain: prediction.organScores.brain.meanRisk,
            lung: prediction.organScores.lung.meanRisk
          },
          tanimoto: prediction.tanimotoSimilarity,
          mcUncertainty: prediction.organScores.liver.uncertaintySigma,
          useTissueConditioning: prediction.useTissueConditioning
        })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setReportText(data.advice);
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with AI Safety Advisor');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-3xl rounded-2xl shadow-2xl p-6 space-y-5 text-slate-100 relative my-8">
        
        {/* Modal Header */}
        <div className="flex justify-between items-start pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-purple-600 rounded-xl">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Gemini Preclinical AI Safety Advisor</h2>
              <p className="text-xs text-slate-400">
                Automated Medicinal Chemistry & Preclinical Safety Synthesis Report
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Compound Info Quick Strip */}
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-wrap justify-between items-center text-xs gap-2">
          <div>
            <span className="text-slate-400 block font-semibold">Query Compound</span>
            <span className="text-white font-bold">{prediction.compoundName}</span>
          </div>
          <div>
            <span className="text-slate-400 block font-semibold">SMILES</span>
            <span className="font-mono text-indigo-300 max-w-xs truncate block">{prediction.smiles}</span>
          </div>
          <div>
            <span className="text-slate-400 block font-semibold">Tissue Mode</span>
            <span className="text-emerald-400 font-bold">
              {prediction.useTissueConditioning ? 'GTEx V8 Active' : 'Baseline'}
            </span>
          </div>
        </div>

        {/* Modal Body */}
        <div className="min-h-[220px] max-h-[50vh] overflow-y-auto bg-slate-950 p-4 rounded-xl border border-slate-800/80 text-xs leading-relaxed space-y-3 font-sans">
          {!reportText && !isLoading && (
            <div className="flex flex-col items-center justify-center h-48 space-y-3 text-slate-400">
              <Bot className="w-8 h-8 text-indigo-400" />
              <p className="text-center max-w-md">
                Click below to generate an authoritative Preclinical Safety & Wet-Lab Assay Recommendation Report powered by Gemini 2.5.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="flex flex-col items-center justify-center h-48 space-y-3 text-indigo-300">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p className="text-xs font-semibold">Synthesizing GTEx transcriptomic cross-attention data & chemical toxicophores...</p>
            </div>
          )}

          {error && (
            <div className="bg-rose-950/60 border border-rose-800 p-4 rounded-xl text-rose-200 space-y-1">
              <span className="font-bold flex items-center space-x-2">
                <ShieldAlert className="w-4 h-4" />
                <span>Error Generating Report</span>
              </span>
              <p>{error}</p>
            </div>
          )}

          {reportText && (
            <div className="prose prose-invert prose-xs max-w-none space-y-3 whitespace-pre-line text-slate-200">
              {reportText}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="flex justify-between items-center pt-3 border-t border-slate-800">
          <span className="text-[11px] text-slate-500 font-mono">
            Powered by @google/genai & EpiADR-Net Engine
          </span>

          <div className="flex space-x-2">
            <button
              onClick={handleGenerateReport}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:opacity-90 text-white font-bold text-xs px-4 py-2 rounded-xl shadow transition-all active:scale-95"
            >
              <Sparkles className="w-4 h-4" />
              <span>{reportText ? 'Regenerate Report' : 'Generate Preclinical Report'}</span>
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold rounded-xl transition-all"
            >
              Close
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
