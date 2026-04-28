// attributeClassifier.js
// Phase 2 — Attribute Classifier Service
// Classifies profile attributes into NORMAL, AMBIGUOUS (proxy-bias), or REDUNDANT.
// This is the academic differentiator of the BiasScope project.

// ─── Classification Constants ────────────────────────────────────────────────

export const CLASS = Object.freeze({
  NORMAL:    'NORMAL',     // 🟢 Safe, directly relevant to decision
  AMBIGUOUS: 'AMBIGUOUS',  // 🟡 Proxy for protected attributes
  REDUNDANT: 'REDUNDANT',  // 🔴 Duplicates another attribute
  UNKNOWN:   'UNKNOWN',    // ⚪ Not in any known list — needs review
});

// ─── Known safe/merit attributes (prevents false UNKNOWN flags) ──────────────

const KNOWN_MERIT = new Set([
  // Job
  'experience', 'skills', 'education_level', 'projects', 'certifications',
  'years_of_experience', 'technical_skills', 'leadership', 'performance_rating',
  'references', 'technical_score', 'projects_completed', 'applied_role',
  'previous_industry', 'highest_education',
  // Loan
  'credit_score', 'annual_income', 'employment_status', 'debt_to_income',
  'loan_amount', 'payment_history', 'employment_length', 'savings_balance',
  'assets', 'loan_purpose', 'existing_debt', 'loan_amount_requested',
  // College
  'gpa', 'sat_score', 'act_score', 'extracurriculars', 'awards',
  'research', 'volunteer_hours', 'ap_classes', 'essay_score',
  'recommendation_strength', 'leadership_roles', 'applied_major',
]);

// Protected attributes that should be flagged
const PROTECTED_ATTRS = new Set([
  'gender', 'sex', 'race', 'ethnicity', 'religion', 'age',
  'disability', 'nationality', 'sexual_orientation',
]);

// ─── Proxy-Bias Attribute Maps (per domain) ─────────────────────────────────
// Each key is a normalized attribute name known to be a proxy for a protected
// class. The value is the protected attribute it proxies for.

const AMBIGUOUS_MAPS = {
  job: {
    zip_code:           'race',
    zipcode:            'race',
    postal_code:        'race',
    neighborhood:       'race/income',
    address:            'race/income',
    university_name:    'socioeconomic_status',
    college_name:       'socioeconomic_status',
    school_name:        'socioeconomic_status',
    graduation_year:    'age',
    grad_year:          'age',
    years_since_graduation: 'age',
    name:               'race/ethnicity',
    full_name:          'race/ethnicity',
    first_name:         'race/ethnicity',
    last_name:          'race/ethnicity',
    surname:            'race/ethnicity',
    native_language:    'national_origin',
    language:           'national_origin',
    mother_tongue:      'national_origin',
    marital_status:     'gender',
    number_of_children: 'gender',
    children:           'gender',
    height:             'gender/disability',
    weight:             'gender/disability',
    club_memberships:   'socioeconomic_status',
    hobbies:            'socioeconomic_status',
    linkedin_photo:     'race/gender/age',
    photo:              'race/gender/age',
    profile_picture:    'race/gender/age',
  },

  loan: {
    zip_code:           'race',
    zipcode:            'race',
    postal_code:        'race',
    neighborhood:       'race/income',
    address:            'race/income',
    property_location:  'race',
    name:               'race/ethnicity',
    full_name:          'race/ethnicity',
    first_name:         'race/ethnicity',
    last_name:          'race/ethnicity',
    surname:            'race/ethnicity',
    marital_status:     'gender',
    number_of_dependents: 'gender',
    dependents:         'gender',
    number_of_children: 'gender',
    employer_name:      'socioeconomic_status',
    employer:           'socioeconomic_status',
    occupation_type:    'gender/race',
    native_language:    'national_origin',
    language:           'national_origin',
    phone_type:         'socioeconomic_status',
    bank_name:          'socioeconomic_status',
    education_level:    'socioeconomic_status',
    school_name:        'socioeconomic_status',
  },

  college: {
    zip_code:           'race',
    zipcode:            'race',
    postal_code:        'race',
    neighborhood:       'race/income',
    address:            'race/income',
    high_school_name:   'socioeconomic_status',
    highschool:         'socioeconomic_status',
    school_name:        'socioeconomic_status',
    parent_occupation:  'socioeconomic_status',
    parent_income:      'socioeconomic_status',
    parents_education:  'socioeconomic_status',
    family_income:      'socioeconomic_status',
    household_income:   'socioeconomic_status',
    name:               'race/ethnicity',
    full_name:          'race/ethnicity',
    first_name:         'race/ethnicity',
    last_name:          'race/ethnicity',
    surname:            'race/ethnicity',
    native_language:    'national_origin',
    language:           'national_origin',
    mother_tongue:      'national_origin',
    legacy_status:      'socioeconomic_status',
    alumni_connection:  'socioeconomic_status',
    donation_history:   'socioeconomic_status',
    extracurriculars:   'socioeconomic_status',
    club_memberships:   'socioeconomic_status',
    travel_experience:  'socioeconomic_status',
  },
};

// ─── Redundant Attribute Pairs ───────────────────────────────────────────────
// If both attributes in a pair exist in a profile, the second is flagged as REDUNDANT.

const REDUNDANT_PAIRS = [
  ['annual_income',      'monthly_income'],
  ['yearly_salary',      'monthly_salary'],
  ['income',             'monthly_income'],
  ['income',             'salary'],
  ['annual_income',      'salary'],
  ['age',                'date_of_birth'],
  ['age',                'birth_date'],
  ['age',                'birth_year'],
  ['years_of_experience','work_experience'],
  ['experience',         'years_of_experience'],
  ['experience',         'work_experience'],
  ['gpa',                'cgpa'],
  ['gpa',                'grade_point_average'],
  ['credit_score',       'credit_rating'],
  ['credit_score',       'fico_score'],
  ['total_assets',       'net_worth'],
  ['education_level',    'highest_degree'],
  ['education',          'education_level'],
  ['education',          'highest_degree'],
  ['debt_to_income',     'dti_ratio'],
  ['loan_amount',        'requested_amount'],
  ['property_value',     'home_value'],
];

// ─── Proxy Bias Explanations (for UI tooltips) ──────────────────────────────

export const PROXY_EXPLANATIONS = {
  zip_code:             'Correlates with race due to historical housing segregation patterns.',
  zipcode:              'Correlates with race due to historical housing segregation patterns.',
  postal_code:          'Correlates with race due to historical housing segregation patterns.',
  neighborhood:         'Correlates with race and income level due to residential segregation.',
  address:              'Correlates with race and income due to residential segregation patterns.',
  property_location:    'Correlates with race due to historical redlining practices.',
  university_name:      'Correlates with socioeconomic status and wealth/privilege.',
  college_name:         'Correlates with socioeconomic status and wealth/privilege.',
  school_name:          'Correlates with neighborhood wealth and racial demographics.',
  high_school_name:     'Correlates with neighborhood wealth and racial demographics.',
  highschool:           'Correlates with neighborhood wealth and racial demographics.',
  graduation_year:      'Can serve as a proxy for age discrimination.',
  grad_year:            'Can serve as a proxy for age discrimination.',
  years_since_graduation: 'Can serve as a proxy for age discrimination.',
  name:                 'Names can reveal ethnicity, national origin, or gender.',
  full_name:            'Names can reveal ethnicity, national origin, or gender.',
  first_name:           'First names can reveal ethnicity, national origin, or gender.',
  last_name:            'Surnames can reveal ethnicity or national origin.',
  surname:              'Surnames can reveal ethnicity or national origin.',
  native_language:      'Directly correlates with national origin — a protected class.',
  language:             'Can correlate with national origin or ethnicity.',
  mother_tongue:        'Directly correlates with national origin — a protected class.',
  marital_status:       'Historically used as proxy for gender discrimination (especially against women).',
  number_of_children:   'Disproportionately penalizes women in hiring and lending decisions.',
  children:             'Disproportionately penalizes women in hiring and lending decisions.',
  number_of_dependents: 'Disproportionately penalizes women in lending decisions.',
  dependents:           'Disproportionately penalizes women in lending decisions.',
  height:               'Correlates with gender and can indicate disability status.',
  weight:               'Correlates with gender and can indicate disability status.',
  parent_occupation:    'Proxy for family socioeconomic status.',
  parent_income:        'Direct measure of family wealth — disadvantages low-income applicants.',
  parents_education:    'Proxy for family socioeconomic status and generational privilege.',
  family_income:        'Direct measure of family wealth — disadvantages low-income applicants.',
  household_income:     'Direct measure of family wealth — disadvantages low-income applicants.',
  legacy_status:        'Perpetuates privilege — legacy admit rate is ~3× non-legacy (NACAC).',
  alumni_connection:    'Perpetuates privilege through institutional connections.',
  donation_history:     'Correlates with family wealth and institutional access.',
  club_memberships:     'Can correlate with socioeconomic status (country clubs, elite orgs).',
  hobbies:              'Certain hobbies correlate with socioeconomic background.',
  extracurriculars:     'Access to activities correlates with family income and opportunity.',
  travel_experience:    'Strongly correlates with family wealth and socioeconomic status.',
  linkedin_photo:       'Photos reveal race, gender, age, and disability status.',
  photo:                'Photos reveal race, gender, age, and disability status.',
  profile_picture:      'Photos reveal race, gender, age, and disability status.',
  employer_name:        'Can correlate with socioeconomic class and institutional prestige.',
  employer:             'Can correlate with socioeconomic class and institutional prestige.',
  occupation_type:      'Certain occupations are heavily gender- and race-stratified.',
  phone_type:           'Phone model/type can indicate socioeconomic status.',
  bank_name:            'Banking institution can correlate with income level and race.',
  education_level:      'Can serve as proxy for socioeconomic background.',
};

// ─── Fuzzy Name Normalization ────────────────────────────────────────────────
// Gemini returns attribute names in various formats (e.g., "Zip Code", "zip_code",
// "ZIP", "Zip-Code"). This normalizer converts all to snake_case for lookup.

const normalize = (name) => {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')  // Replace non-alphanumeric runs with _
    .replace(/^_|_$/g, '');        // Strip leading/trailing underscores
};

// ─── Core Classification Function ───────────────────────────────────────────

/**
 * Classifies each attribute in a profile object.
 *
 * @param {string} domain - One of 'job', 'loan', 'college'
 * @param {Object} attributesObject - { "Zip Code": "90210", "GPA": 3.8, ... }
 * @returns {Object} - { "Zip Code": "AMBIGUOUS", "GPA": "NORMAL", ... }
 */
export const classifyAttributes = (domain, attributesObject) => {
  const d = domain.toLowerCase();
  const ambiguousMap = AMBIGUOUS_MAPS[d] || {};
  const attrNames = Object.keys(attributesObject);
  const normalizedNames = attrNames.map(normalize);
  const result = {};

  // First pass: detect redundant pairs
  const redundantSet = new Set();
  for (const [a, b] of REDUNDANT_PAIRS) {
    const idxA = normalizedNames.indexOf(a);
    const idxB = normalizedNames.indexOf(b);
    if (idxA !== -1 && idxB !== -1) {
      // The second attribute in the pair is flagged as redundant
      redundantSet.add(attrNames[idxB]);
    }
  }

  // Second pass: classify each attribute
  for (let i = 0; i < attrNames.length; i++) {
    const original = attrNames[i];
    const norm = normalizedNames[i];

    if (redundantSet.has(original)) {
      result[original] = CLASS.REDUNDANT;
    } else if (PROTECTED_ATTRS.has(norm)) {
      result[original] = CLASS.AMBIGUOUS; // Protected attrs shown as ambiguous in UI
    } else if (norm in ambiguousMap) {
      result[original] = CLASS.AMBIGUOUS;
    } else if (KNOWN_MERIT.has(norm)) {
      result[original] = CLASS.NORMAL;
    } else {
      // Not in any known list — flag for review
      result[original] = CLASS.UNKNOWN;
    }
  }

  return result;
};

// ─── Helper: Get explanation for a flagged attribute ─────────────────────────

/**
 * Returns the proxy-bias explanation for an attribute, or null if not found.
 *
 * @param {string} attributeName - The raw attribute name (e.g., "Zip Code")
 * @returns {string|null} - Human-readable explanation or null
 */
export const getExplanation = (attributeName) => {
  const norm = normalize(attributeName);
  return PROXY_EXPLANATIONS[norm] || null;
};

// ─── Helper: Get the protected class an attribute proxies for ────────────────

/**
 * Returns which protected class an ambiguous attribute correlates with.
 *
 * @param {string} domain - One of 'job', 'loan', 'college'
 * @param {string} attributeName - The raw attribute name
 * @returns {string|null} - Protected class name or null
 */
export const getProxiedClass = (domain, attributeName) => {
  const d = domain.toLowerCase();
  const norm = normalize(attributeName);
  const map = AMBIGUOUS_MAPS[d] || {};
  return map[norm] || null;
};

// ─── Helper: Batch classify and enrich with explanations ─────────────────────

/**
 * Classifies attributes and returns enriched results with explanations.
 *
 * @param {string} domain - One of 'job', 'loan', 'college'
 * @param {Object} attributesObject - Profile attributes
 * @returns {Object} - { attrName: { class, explanation?, proxiedClass? } }
 */
export const classifyWithDetails = (domain, attributesObject) => {
  const classifications = classifyAttributes(domain, attributesObject);
  const result = {};

  for (const [attr, cls] of Object.entries(classifications)) {
    const entry = { class: cls };

    if (cls === CLASS.AMBIGUOUS) {
      entry.explanation = getExplanation(attr);
      entry.proxiedClass = getProxiedClass(domain, attr);
    } else if (cls === CLASS.REDUNDANT) {
      entry.explanation = 'This attribute is redundant — it duplicates information already captured by another attribute in the profile.';
    } else if (cls === CLASS.UNKNOWN) {
      entry.explanation = '⚠️ This attribute is not in any known list. It may be a proxy for protected attributes. Use Reward/Penalize to classify it.';
    }

    result[attr] = entry;
  }

  return result;
};
