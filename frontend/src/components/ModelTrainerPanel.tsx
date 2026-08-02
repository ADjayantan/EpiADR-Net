import React, { useState } from 'react';
import { EpiADRHyperparameters, ModelTrainingSummary } from '../types';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Play, Square, RotateCcw, Cpu, Sliders, CheckCircle2, ShieldAlert, Sparkles, Activity } from 'lucide-react';

interface ModelTrainerPanelProps {
  hyperparams: EpiADRHyperparameters;
  onChangeHyperparams: (params: EpiADRHyperparameters) => void;
  isTraining: boolean;
  onStartTraining: () => void;
  onStopTraining: () => void;
  trainingSummary: ModelTrainingSummary | null;
  useTissueConditioning: boolean;
}

export const ModelTrainerPanel: React.FC<ModelTrainerPanelProps> = ({
  hyperparams,
  onChangeHyperparams,
  isTraining,
  onStartTraining,
  onStopTraining,
  trainingSummary,
  useTissueConditioning
}) => {
  return (
    <div className="space-y-6">
      
      {/* Top Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs bg-indigo-950 text-indigo-300 border border-indigo-700/60 font-semibold px-2 py-0.5 rounded-full uppercase tracking-wider">
              EpiADR-Net Foundation Model Training Studio
            </span>
          </div>
          <h2 className="text-xl font-bold text-white mt-1">
            Model Training & Hyperparameter Tuning
          </h2>
          <p className="text-xs text-slate-400">
            Train Graph Transformer with 16-Head Bi-Directional Gene Pathway Cross-Attention on SIDER 4.1 & GTEx V8 dataset.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-3">
          {!isTraining ? (
            <button
              onClick={onStartTraining}
              className="flex items-center space-x-2 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:opacity-90 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-md transition-all active:scale-95"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>Train Model</span>
            </button>
          ) : (
            <button
              onClick={onStopTraining}
              className="flex items-center space-x-2 bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl shadow-md transition-all"
            >
              <Square className="w-4 h-4 fill-white" />
              <span>Cancel Training</span>
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Hyperparameters Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-bold text-white">Hyperparameters & Loss Tuning</h3>
            </div>
          </div>

          {/* Tissue Conditioning Toggle */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1.5">
            <label className="text-xs font-semibold text-slate-300 block">Tissue Conditioning</label>
            <div className="flex space-x-2">
              <button
                onClick={() => onChangeHyperparams({ ...hyperparams, useTissueConditioning: true })}
                disabled={isTraining}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold border transition-all ${
                  hyperparams.useTissueConditioning
                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                GTEx V8 (Active)
              </button>
              <button
                onClick={() => onChangeHyperparams({ ...hyperparams, useTissueConditioning: false })}
                disabled={isTraining}
                className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-semibold border transition-all ${
                  !hyperparams.useTissueConditioning
                    ? 'bg-rose-600 border-rose-500 text-white shadow-sm'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                Molecule-Only Baseline
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              Disabling tissue conditioning tests the molecule-only baseline scientific control.
            </p>
          </div>

          {/* Cross-Attention Heads */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-400">Cross-Attention Heads</span>
              <span className="font-mono text-indigo-300 font-bold">{hyperparams.crossAttentionHeads} Heads</span>
            </div>
            <select
              value={hyperparams.crossAttentionHeads}
              disabled={isTraining}
              onChange={(e) => onChangeHyperparams({ ...hyperparams, crossAttentionHeads: parseInt(e.target.value) })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200"
            >
              <option value="8">8 Heads (Fast)</option>
              <option value="16">16 Heads (Standard EpiADR-Net)</option>
              <option value="32">32 Heads (High Expressivity)</option>
            </select>
          </div>

          {/* Learning Rate & Epochs */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-400">Epochs</span>
                <span className="font-mono text-indigo-300 font-bold">{hyperparams.epochs}</span>
              </div>
              <input
                type="range"
                min="10"
                max="150"
                step="10"
                disabled={isTraining}
                value={hyperparams.epochs}
                onChange={(e) => onChangeHyperparams({ ...hyperparams, epochs: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-slate-400">Learning Rate</span>
                <span className="font-mono text-indigo-300 font-bold">{hyperparams.learningRate}</span>
              </div>
              <select
                value={hyperparams.learningRate}
                disabled={isTraining}
                onChange={(e) => onChangeHyperparams({ ...hyperparams, learningRate: parseFloat(e.target.value) })}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-200"
              >
                <option value="0.003">0.003</option>
                <option value="0.001">0.001 (Recommended)</option>
                <option value="0.0003">0.0003</option>
              </select>
            </div>
          </div>

          {/* Pos Weight Loss Class Imbalance */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-400">Class Imbalance Weight (pos_weight)</span>
              <span className="font-mono text-indigo-300 font-bold">{hyperparams.posWeight}x</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="5.0"
              step="0.5"
              disabled={isTraining}
              value={hyperparams.posWeight}
              onChange={(e) => onChangeHyperparams({ ...hyperparams, posWeight: parseFloat(e.target.value) })}
              className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[11px] text-slate-500">
              Balances rare positive toxicity labels in SIDER 4.1.
            </p>
          </div>

          {/* Monte Carlo Uncertainty Passes */}
          <div className="space-y-1 pt-1">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-400">MC Dropout Passes (N)</span>
              <span className="font-mono text-indigo-300 font-bold">N={hyperparams.mcDropoutPasses}</span>
            </div>
            <select
              value={hyperparams.mcDropoutPasses}
              disabled={isTraining}
              onChange={(e) => onChangeHyperparams({ ...hyperparams, mcDropoutPasses: parseInt(e.target.value) })}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200"
            >
              <option value="10">10 Passes (Fast)</option>
              <option value="30">30 Passes (Standard Bayesian)</option>
              <option value="50">50 Passes (Ultra-Precise)</option>
            </select>
          </div>
        </div>

        {/* Right Column: Training Progress & Metrics Chart */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Epoch Metrics Chart */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-bold text-white">Live Training Curves (Loss & Val AUROC)</h3>
              </div>
              {isTraining && (
                <span className="text-xs text-amber-400 font-mono animate-pulse font-semibold">
                  Training in Progress...
                </span>
              )}
            </div>

            {trainingSummary && trainingSummary.epochHistory.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingSummary.epochHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                    <YAxis yAxisId="left" stroke="#64748b" fontSize={11} domain={[0, 1]} />
                    <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={11} domain={[40, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Line yAxisId="left" type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#f43f5e" strokeWidth={2} dot={false} />
                    <Line yAxisId="left" type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
                    <Line yAxisId="right" type="monotone" dataKey="valAUROC" name="Val AUROC (%)" stroke="#10b981" strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-64 w-full bg-slate-950 rounded-xl border border-slate-800/80 flex flex-col items-center justify-center text-slate-500 space-y-2">
                <Cpu className="w-8 h-8 text-slate-600" />
                <p className="text-xs font-medium">Click "Train Model" to start training session.</p>
              </div>
            )}
          </div>

          {/* Final Metrics Cards */}
          {trainingSummary && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-center">
                <span className="text-[11px] font-semibold text-slate-400 block">Val AUROC</span>
                <span className="text-xl font-extrabold font-mono text-emerald-400">{trainingSummary.finalValAUROC}%</span>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-center">
                <span className="text-[11px] font-semibold text-slate-400 block">F1 Macro Score</span>
                <span className="text-xl font-extrabold font-mono text-indigo-400">{trainingSummary.finalF1Score}</span>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-center">
                <span className="text-[11px] font-semibold text-slate-400 block">Final Val Loss</span>
                <span className="text-xl font-extrabold font-mono text-amber-400">{trainingSummary.finalValLoss}</span>
              </div>
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl text-center">
                <span className="text-[11px] font-semibold text-slate-400 block">Training Time</span>
                <span className="text-xl font-extrabold font-mono text-slate-200">{trainingSummary.trainingTimeMs}ms</span>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
