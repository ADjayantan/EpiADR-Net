import React, { useState } from 'react';
import { Sparkles, X, ShieldAlert, CheckCircle, RefreshCw } from 'lucide-react';

interface AiAdvisorModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialData?: any;
}

export const AiAdvisorModal: React.FC<AiAdvisorModalProps> = ({
  isOpen,
  onClose,
  initialData
}) => {
  const [reportText, setReportText] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  if (!isOpen) return null;

  const handleGenerateReport = async () => {
    setIsLoading(true);
    setReportText(null);

    try {
      const response = await fetch('/api/ml-advisor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(initialData || {
          compoundName: 'Doxorubicin',
          smiles: 'CC1C(C(CC(O1)OC2CC(CC3=C(C4=C(C(=C23)O)C(=O)C5=C(C4=O)C=CC=C5OC)O)(C(=O)CO)O)N)O',
          organScores: { liver: 0.65, heart: 0.96, kidney: 0.58, brain: 0.22, lung: 0.35 },
          tanimoto: 0.82,
          mcUncertainty: 0.03,
          useTissueConditioning: true
        })
      });

      const data = await response.json();
      setReportText(data.advice || 'Report generated successfully.');
    } catch (err: any) {
      setReportText(`Error generating report: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-hidden flex flex-col shadow-2xl">
        
        {/* Modal Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950">
          <div className="flex items-center space-x-2">
            <div className="p-2 bg-gradient-to-tr from-indigo-500 to-pink-500 rounded-xl text-white">
              <Sparkles className="w-5 h-5 animate-spin-slow" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Gemini Preclinical AI Safety Synthesis</h2>
              <p className="text-xs text-slate-400">Tissue-Conditioned Organ Toxicity Analysis & Wet-Lab Guidance</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs text-slate-200 leading-relaxed font-sans">
          
          {!reportText && !isLoading && (
            <div className="text-center py-8 space-y-3">
              <p className="text-slate-300 font-medium">
                Click below to synthesize an automated preclinical toxicity evaluation for <span className="text-indigo-400 font-bold">{initialData?.compoundName || 'Doxorubicin'}</span> using Gemini 2.5 Flash.
              </p>
              <button
                onClick={handleGenerateReport}
                className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-pink-600 hover:opacity-90 text-white font-bold rounded-xl shadow-lg transition-all"
              >
                Synthesize Preclinical Report
              </button>
            </div>
          )}

          {isLoading && (
            <div className="flex flex-col items-center justify-center py-12 space-y-3">
              <Sparkles className="w-8 h-8 text-indigo-400 animate-spin" />
              <span className="text-indigo-300 font-mono text-xs animate-pulse">
                Analyzing GTEx V8 transcriptomic profiles & SMILES toxicophore subgraphs...
              </span>
            </div>
          )}

          {reportText && (
            <div className="space-y-3 bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono whitespace-pre-wrap leading-relaxed text-indigo-200">
              {reportText}
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-950 flex justify-between items-center text-xs text-slate-400">
          <span className="font-mono text-[11px]">Powered by Gemini 2.5 Flash & EpiADR-Net v2.1</span>
          {reportText && (
            <button
              onClick={handleGenerateReport}
              className="flex items-center space-x-1 text-indigo-400 hover:text-indigo-300 font-bold"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Regenerate</span>
            </button>
          )}
        </div>

      </div>
    </div>
  );
};
