// rl.js
// Handles the local storage memory bank for simulating RL loop.

const MEMORY_KEY = 'bias_scope_memory';

export const getMemory = () => {
  const mem = localStorage.getItem(MEMORY_KEY);
  return mem ? JSON.parse(mem) : {
    job: { positive: [], negative: [] },
    college: { positive: [], negative: [] },
    loan: { positive: [], negative: [] }
  };
};

export const saveMemory = (memory) => {
  localStorage.setItem(MEMORY_KEY, JSON.stringify(memory));
};

export const addFeedback = (domain, attribute, type) => {
  // domain: 'job', 'college', 'loan'
  // type: 'positive' or 'negative'
  const memory = getMemory();
  const d = domain.toLowerCase();
  if (!memory[d]) memory[d] = { positive: [], negative: [] };
  
  // Check if attribute already exists
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
  let allAttributes = [];
  
  ['job', 'college', 'loan'].forEach(d => {
    negBiasCount += memory[d].negative.length;
    posBiasCount += memory[d].positive.length;
    
    memory[d].negative.forEach(attr => allAttributes.push({ attr, type: 'neg' }));
    memory[d].positive.forEach(attr => allAttributes.push({ attr, type: 'pos' }));
  });
  
  return {
    totalDecisions,
    negBiasCount,
    posBiasCount,
    cautionCount: Math.floor(negBiasCount * 1.5), // just a dummy stat based on neg bias
    allAttributes
  };
};

export const incrementDecisions = () => {
  let totalDecisions = parseInt(localStorage.getItem('bias_scope_total') || '0');
  localStorage.setItem('bias_scope_total', (totalDecisions + 1).toString());
};
