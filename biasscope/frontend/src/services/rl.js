// rl.js
// Handles the local storage memory bank for simulating RL loop.

const MEMORY_KEY = 'bias_scope_memory';

export const getMemory = () => {
  const mem = localStorage.getItem(MEMORY_KEY);
  if (mem) {
    const parsed = JSON.parse(mem);
    // Migrate old schema: ensure ambiguous list exists
    ['job', 'college', 'loan'].forEach(d => {
      if (parsed[d] && !parsed[d].ambiguous) {
        parsed[d].ambiguous = [];
      }
    });
    return parsed;
  }
  return {
    job: { positive: [], negative: [], ambiguous: [] },
    college: { positive: [], negative: [], ambiguous: [] },
    loan: { positive: [], negative: [], ambiguous: [] }
  };
};

export const saveMemory = (memory) => {
  localStorage.setItem(MEMORY_KEY, JSON.stringify(memory));
};

export const addFeedback = (domain, attribute, type) => {
  // domain: 'job', 'college', 'loan'
  // type: 'positive', 'negative', or 'ambiguous'
  const memory = getMemory();
  const d = domain.toLowerCase();
  if (!memory[d]) memory[d] = { positive: [], negative: [], ambiguous: [] };

  // Conflict resolution: remove from opposite lists
  if (type === 'positive') {
    memory[d].negative = memory[d].negative.filter(a => a !== attribute);
    memory[d].ambiguous = memory[d].ambiguous.filter(a => a !== attribute);
  } else if (type === 'negative') {
    memory[d].positive = memory[d].positive.filter(a => a !== attribute);
    memory[d].ambiguous = memory[d].ambiguous.filter(a => a !== attribute);
  } else if (type === 'ambiguous') {
    memory[d].positive = memory[d].positive.filter(a => a !== attribute);
    memory[d].negative = memory[d].negative.filter(a => a !== attribute);
  }

  // Add if not already present
  if (!memory[d][type].includes(attribute)) {
    memory[d][type].push(attribute);
  }

  saveMemory(memory);
  return memory;
};

export const getStats = () => {
  const memory = getMemory();
  let totalDecisions = parseInt(localStorage.getItem('bias_scope_total') || '0');
  let negBiasCount = 0;
  let posBiasCount = 0;
  let cautionCount = 0;
  let allAttributes = [];

  ['job', 'college', 'loan'].forEach(d => {
    negBiasCount += memory[d].negative.length;
    posBiasCount += memory[d].positive.length;
    cautionCount += (memory[d].ambiguous || []).length;

    memory[d].negative.forEach(attr => allAttributes.push({ attr, type: 'neg', domain: d }));
    memory[d].positive.forEach(attr => allAttributes.push({ attr, type: 'pos', domain: d }));
    (memory[d].ambiguous || []).forEach(attr => allAttributes.push({ attr, type: 'caution', domain: d }));
  });

  return {
    totalDecisions,
    negBiasCount,
    posBiasCount,
    cautionCount,
    allAttributes
  };
};

export const incrementDecisions = () => {
  let totalDecisions = parseInt(localStorage.getItem('bias_scope_total') || '0');
  localStorage.setItem('bias_scope_total', (totalDecisions + 1).toString());
};

export const resetMemory = () => {
  const memory = {
    job: { positive: [], negative: [], ambiguous: [] },
    college: { positive: [], negative: [], ambiguous: [] },
    loan: { positive: [], negative: [], ambiguous: [] }
  };
  saveMemory(memory);
  localStorage.setItem('bias_scope_total', '0');
  return memory;
};

export const exportMemory = () => {
  const memory = getMemory();
  const stats = getStats();
  const exportData = {
    memory,
    stats,
    exportedAt: new Date().toISOString(),
    version: '1.1.0'
  };
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'biasscope_memory.json';
  a.click();
  URL.revokeObjectURL(url);
};

export const importMemory = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        // Support both raw memory and wrapped export format
        const memory = data.memory || data;
        // Validate structure
        for (const d of ['job', 'college', 'loan']) {
          if (!memory[d]) {
            reject(new Error(`Missing domain: ${d}`));
            return;
          }
          if (!memory[d].positive || !memory[d].negative) {
            reject(new Error(`Invalid structure for domain: ${d}`));
            return;
          }
          if (!memory[d].ambiguous) memory[d].ambiguous = [];
        }
        saveMemory(memory);
        resolve(memory);
      } catch (err) { reject(err); }
    };
    reader.readAsText(file);
  });
};
