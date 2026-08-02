import React from 'react';
import { DrugRecord } from '../types';
import { SIDER_BENCHMARK_DRUGS } from '../utils/siderDataset';
import { Database, Play, CheckCircle2, ArrowRight } from 'lucide-react';

interface SiderBenchmarkTableProps {
  onSelectDrug: (drug: DrugRecord) => void;
  selectedDrugId?: string;
}

export const SiderBenchmarkTable: React.FC<SiderBenchmarkTableProps> = ({
  onSelectDrug,
  selectedDrugId
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-sm space-y-4">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 border-b border-slate-800 gap-2">
        <div className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-indigo-400" />
          <div>
            <h2 className="text-base font-bold text-white">SIDER 4.1 Benchmark Reference Dataset</h2>
            <p className="text-xs text-slate-400">
              Select FDA-approved compounds to test zero-shot tissue-conditioned predictions against benchmark organ scores.
            </p>
          </div>
        </div>

        <span className="text-xs font-mono bg-indigo-950 text-indigo-300 border border-indigo-800 px-3 py-1 rounded-lg">
          7,325 Total ADR Records
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
              <th className="p-2.5">Compound Name</th>
              <th className="p-2.5">Indication</th>
              <th className="p-2.5">SMILES Structure</th>
              <th className="p-2.5 text-center">Liver Risk</th>
              <th className="p-2.5 text-center">Heart Risk</th>
              <th className="p-2.5 text-center">Kidney Risk</th>
              <th className="p-2.5 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {SIDER_BENCHMARK_DRUGS.map((drug) => {
              const isSelected = drug.id === selectedDrugId;
              return (
                <tr
                  key={drug.id}
                  className={`hover:bg-slate-800/50 transition-all ${isSelected ? 'bg-indigo-950/40' : ''}`}
                >
                  <td className="p-2.5 font-bold text-white">
                    <div className="flex items-center space-x-2">
                      <span>{drug.name}</span>
                      {drug.fdaApproved && (
                        <span className="text-[9px] bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono px-1 rounded">
                          FDA
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="p-2.5 text-slate-300">{drug.indication}</td>
                  <td className="p-2.5 font-mono text-slate-400 max-w-xs truncate" title={drug.smiles}>
                    {drug.smiles}
                  </td>
                  <td className="p-2.5 text-center font-mono font-bold text-amber-400">
                    {(drug.organScores.liver * 100).toFixed(0)}%
                  </td>
                  <td className="p-2.5 text-center font-mono font-bold text-rose-400">
                    {(drug.organScores.heart * 100).toFixed(0)}%
                  </td>
                  <td className="p-2.5 text-center font-mono font-bold text-sky-400">
                    {(drug.organScores.kidney * 100).toFixed(0)}%
                  </td>
                  <td className="p-2.5 text-center">
                    <button
                      onClick={() => onSelectDrug(drug)}
                      className={`flex items-center space-x-1 px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                        isSelected
                          ? 'bg-indigo-600 text-white'
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                      }`}
                    >
                      <span>Analyze</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
