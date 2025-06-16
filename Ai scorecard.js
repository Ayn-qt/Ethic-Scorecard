import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';

export default function EthicsScoringApp() {
  const [dilemma, setDilemma] = useState('');
  const [scores, setScores] = useState(null);
  const [explanations, setExplanations] = useState(null);

  const handleSubmit = async () => {
    const sampleScores = {
      Hinduism: 3,
      Islam: 6,
      Christianity: 4,
      Buddhism: 2,
      Sikhism: 5
    };
    const sampleExplanations = {
      Hinduism: 'Emphasizes non-violence (ahimsa) and duty (dharma).',
      Islam: 'Allows self-defense with caution for justice and mercy.',
      Christianity: 'Prioritizes peace and sanctity of life.',
      Buddhism: 'Focuses on compassion and minimizing harm.',
      Sikhism: 'Stresses justice and protection of the innocent.'
    };
    setScores(sampleScores);
    setExplanations(sampleExplanations);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h1 className="text-3xl font-bold text-center">AI Ethics Scoring System</h1>

      <div className="border p-4 rounded-lg space-y-4 shadow">
        <textarea
          className="w-full border p-2 rounded min-h-[120px]"
          placeholder="Enter your ethical dilemma here..."
          value={dilemma}
          onChange={(e) => setDilemma(e.target.value)}
        />
        <button onClick={handleSubmit} className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition">
          Analyze
        </button>
      </div>

      {scores && (
        <div className="border p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Ethics Score (0–10)</h2>
          <Bar
            data={{
              labels: Object.keys(scores),
              datasets: [
                {
                  label: 'Ethics Score',
                  data: Object.values(scores),
                  backgroundColor: 'rgba(75, 192, 192, 0.6)'
                }
              ]
            }}
            options={{
              scales: {
                y: {
                  beginAtZero: true,
                  max: 10
                }
              }
            }}
          />
        </div>
      )}

      {explanations && (
        <div className="border p-4 rounded-lg space-y-4 shadow">
          <h2 className="text-xl font-semibold">Explanations</h2>
          {Object.entries(explanations).map(([religion, text]) => (
            <div key={religion}>
              <h3 className="font-bold">{religion}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
