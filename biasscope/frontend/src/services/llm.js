// llm.js
import { GoogleGenerativeAI } from '@google/generative-ai';

let apiKey = '';

export const setApiKey = (key) => {
  apiKey = key;
};

const getModel = () => {
  if (!apiKey) throw new Error("API Key is not set.");
  const genAI = new GoogleGenerativeAI(apiKey);
  return genAI.getGenerativeModel({ model: "gemini-2.5-flash" }); // Fast and cheap model
};

export const generateProfile = async (domain) => {
  const model = getModel();
  const prompt = `You are a data generator for a bias detection system. 
Generate a JSON object representing a realistic applicant profile for the '${domain}' domain. 
Include 5-7 diverse attributes such as Age, Gender, Race, Education, Experience Level, Income, Location, etc. 
Ensure attributes vary.
Only return valid JSON without markdown wrapping. Format:
{
  "name": "...",
  "attributes": {
    "Age": 34,
    "Gender": "...",
    ...
  }
}`;

  const result = await model.generateContent(prompt);
  const text = result.response.text().trim().replace(/^```json/, '').replace(/```$/, '');
  return JSON.parse(text);
};

export const makeDecision = async (domain, profile, memory) => {
  const model = getModel();
  
  // Convert memory to prompt context
  const domainMemory = memory[domain.toLowerCase()];
  let instruction = `You are an AI making a decision for the '${domain}' domain.
Profile: ${JSON.stringify(profile)}

Based on this profile, make a decision:
For Job: 'Hire' or 'Reject'
For College: 'Admit' or 'Reject'
For Loan: 'Approve' or 'Deny'

Also, identify the TOP 4-5 attributes from the profile that you gave the most weight to in your decision. For each, give an estimated percentage weight (so they sum to roughly 100%) and briefly explain why.

IMPORTANT: Your behavior is influenced by Reinforcement Learning feedback.
If these attributes are listed as "Penalized/Negative Bias", DO NOT use them positively, or if you do, acknowledge it as a bias.
Penalized Attributes: ${domainMemory.negative.join(', ') || 'None'}
Rewarded Attributes: ${domainMemory.positive.join(', ') || 'None'}

Only return valid JSON without markdown wrapping. Format:
{
  "decision": "Hire/Reject/etc",
  "weightedAttributes": [
    {
      "attribute": "AttributeName",
      "weight": "40%",
      "reasoning": "Short explanation of why this attribute drove the decision."
    }
  ]
}`;

  const result = await model.generateContent(instruction);
  const text = result.response.text().trim().replace(/^```json/, '').replace(/```$/, '');
  return JSON.parse(text);
};
