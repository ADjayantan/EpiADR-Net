import React, { useState } from 'react';
import { EpiADRHyperparameters, EpochTrainingMetric, ModelTrainingSummary } from '../types';
import { globalEpiADREngine } from '../utils/epiAdrEngine';
import { SIDER_BENCHMARK_DRUGS } from '../utils/siderDataset';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { Cpu, Play, Square, RotateCcw, Activity, Layers, CheckCircle, BarChart2, ShieldAlert } from 'lucide-react';

interface TrainingPanelProps {
  useTissueConditioning: boolean;
  setUseTissueConditioning: (val: boolean) => void;
}

export const TrainingPanel: React.FC<TrainingPanelProps> = ({
  useTissueConditioning,
  setUseTissueConditioning
}) => {
  const [hyperparams, setHyperparams] = useState<EpiADRHyperparameters>({
    useTissueConditioning,
    crossAttentionHeads: 16,
    learningRate: 0.001,
    epochs: 40,
    batchSize: 16,
    optimizer: 'adam',
    posWeight: 2.5,
    regularizationL2: 0.001,
    dropoutRate: 0.2,
    mcDropoutPasses: 30
  });

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epochHistory, setEpochHistory] = useState<EpochTrainingMetric[]>([]);
  const [trainingSummary, setTrainingSummary] = useState<ModelTrainingSummary | null>(null);

  const handleStartTraining = async () => {
    setIsTraining(true);
    setEpochHistory([]);
    setTrainingSummary(null);

    try {
      const summary = await globalEpiADREngine.trainEpiADRModel(
        SIDER_BENCHMARK_DRUGS,
        { ...hyperparams, useTissueConditioning },
        (metric) => {
          setEpochHistory(prev => [...prev, metric]);
        }
      );
      setTrainingSummary(summary);
    } catch (err: any) {
      console.error("Training error:", err);
    } finally {
      setIsTraining(false);
    }
  };

  const handleStopTraining = () => {
    globalEpiADREngine.cancelTraining();
    setIsTraining(false);
  };

  return (
    <div className="space-y-6">
      
      {/* Training Configuration Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Hyperparameter Controls */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-white">EpiADR-Net Model Architecture</h2>
          </div>

          {/* Mode Switcher */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
            <span className="text-xs font-semibold text-slate-300 block">Scientific Controlled Baseline</span>
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  setUseTissueConditioning(true);
                  setHyperparams({ ...hyperparams, useTissueConditioning: true });
                }}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all border ${
                  useTissueConditioning
                    ? 'bg-indigo-600 text-white border-indigo-400 shadow-md'
                    : 'bg-slate-900 text-slate-400 border-slate-800'
                }`}
              >
                Tissue-Conditioned
              </button>
              <button
                onClick={() => {
                  setUseTissueConditioning(false);
                  setHyperparams({ ...hyperparams, useTissueConditioning: false });
                }}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-bold transition-all border ${
                  !useTissueConditioning
                    ? 'bg-amber-600 text-white border-amber-400 shadow-md'
                    : 'bg-slate-900 text-slate-400 border-slate-800'
                }`}
              >
                Molecule-Only
              </button>
            </div>
            <p className="text-[11px] text-slate-400">
              {useTissueConditioning
                ? 'Fuses SMILES 256-bit fingerprint with GTEx V8 128-dim organ transcriptomic vectors via 16-Head Cross-Attention.'
                : 'Disables transcriptomic profiles to measure scientific accuracy uplift of human gene expression data.'}
            </p>
          </div>

          {/* Hyperparameters */}
          <div className="space-y-3 pt-1 text-xs">
            
            <div className="space-y-1">
              <div className="flex justify-between text-slate-300 font-semibold">
                <span>Epochs</span>
                <span className="font-mono text-indigo-300">{hyperparams.epochs}</span>
              </div>
              <input
                type="range"
                min="10"
                max="100"
                step="5"
                disabled={isTraining}
                value={hyperparams.epochs}
                onChange={(e) => setHyperparams({ ...hyperparams, epochs: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-slate-300 font-semibold">
                <span>Learning Rate (η)</span>
                <span className="font-mono text-indigo-300">{hyperparams.learningRate}</span>
              </div>
              <select
                value={hyperparams.learningRate}
                disabled={isTraining}
                onChange={(e) => setHyperparams({ ...hyperparams, learningRate: parseFloat(e.target.value) })}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200"
              >
                <option value="0.005">0.005 (Fast)</option>
                <option value="0.001">0.001 (Recommended)</option>
                <option value="0.0003">0.0003 (Fine)</option>
              </select>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-slate-300 font-semibold">
                <span>Class Imbalance Weight (pos_weight)</span>
                <span className="font-mono text-indigo-300">{hyperparams.posWeight}x</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="5.0"
                step="0.5"
                disabled={isTraining}
                value={hyperparams.posWeight}
                onChange={(e) => setHyperparams({ ...hyperparams, posWeight: parseFloat(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-slate-300 font-semibold">
                <span>L2 Weight Regularization</span>
                <span className="font-mono text-indigo-300">{hyperparams.regularizationL2}</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.01"
                step="0.001"
                disabled={isTraining}
                value={hyperparams.regularizationL2}
                onChange={(e) => setHyperparams({ ...hyperparams, regularizationL2: parseFloat(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>

          </div>

          {/* Controls */}
          <div className="pt-3 border-t border-slate-800 flex space-x-2">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold text-xs shadow-lg flex items-center justify-center space-x-2 transition-all active:scale-95"
              >
                <Play className="w-4 h-4 fill-white" />
                <span>Train EpiADR-Net</span>
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="flex-1 py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-xl font-bold text-xs shadow-lg flex items-center justify-center space-x-2 transition-all"
              >
                <Square className="w-4 h-4 fill-white" />
                <span>Halt Training</span>
              </button>
            )}
          </div>

        </div>

        {/* Right Column: Live Epoch Loss & AUROC Charts */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white">Live Training & Validation Dynamics</h3>
            </div>
            {isTraining && (
              <span className="flex items-center space-x-1 text-xs text-indigo-400 font-mono animate-pulse">
                <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
                <span>Optimizing Epoch {epochHistory.length}/{hyperparams.epochs}...</span>
              </span>
            )}
          </div>

          {/* Loss Curve */}
          <div className="h-52 bg-slate-950 p-2 rounded-xl border border-slate-800">
            <span className="text-[11px] text-slate-400 font-mono font-semibold px-2 block">Binary Crossentropy Loss</span>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={epochHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                <Line type="monotone" dataKey="trainLoss" stroke="#6366f1" strokeWidth={2} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="valLoss" stroke="#f43f5e" strokeWidth={2} dot={false} name="Val Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AUROC Curve */}
          <div className="h-52 bg-slate-950 p-2 rounded-xl border border-slate-800">
            <span className="text-[11px] text-slate-400 font-mono font-semibold px-2 block">Validation AUROC Trajectory</span>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={epochHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={10} domain={[0.5, 1.0]} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                <Line type="monotone" dataKey="valAUROC" stroke="#10b981" strokeWidth={2.5} dot={false} name="Validation AUROC" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Model Training Summary Metrics */}
          {trainingSummary && (
            <div className="bg-slate-950 p-4 rounded-xl border border-indigo-500/30 grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs font-mono animate-fade-in">
              <div>
                <span className="text-slate-400 block text-[10px]">Validation AUROC</span>
                <span className="text-emerald-400 font-bold text-base">{trainingSummary.finalValAUROC.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Macro F1 Score</span>
                <span className="text-indigo-400 font-bold text-base">{trainingSummary.finalF1Score.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Final Loss</span>
                <span className="text-rose-400 font-bold text-base">{trainingSummary.finalValLoss.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px]">Training Duration</span>
                <span className="text-slate-200 font-bold text-base">{trainingSummary.trainingTimeMs} ms</span>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
