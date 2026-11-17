const BLOOM_NAMES: Record<number, string> = {
  1: "Remember",
  2: "Understand",
  3: "Apply",
  4: "Analyze",
  5: "Evaluate",
  6: "Create",
};


export function normalizeBloomData(raw: any, rawText: string) {
  const rawQuestions = Array.isArray(raw?.questions) ? raw.questions : [];

  let totalMarks = 0;

  const questions = rawQuestions.map((q: any, idx: number) => {
    // Support both Python camelCase and existing dashboard snake_case fields.
    const questionNumber =
      q.question_number ||
      q.fullIdentifier ||
      q.questionNumber ||
      `Q${idx + 1}`;

    const questionText =
      q.text ||
      q.questionText ||
      q.rawText ||
      `Question ${idx + 1}`;

    const assignedLevel = Number(
      q.bloom_level ?? q.aiBloomLevel ?? q.assignedBloomLevel
    );

    const assignedName =
      q.bloom_name ||
      q.aiBloomName ||
      q.assignedBloomName ||
      "";

    let levelNum = assignedLevel;

    if (isNaN(levelNum) || levelNum < 1 || levelNum > 6) {
      const name = String(assignedName).toLowerCase();

      if (name.includes("remember")) levelNum = 1;
      else if (name.includes("understand")) levelNum = 2;
      else if (name.includes("apply")) levelNum = 3;
      else if (name.includes("analyze") || name.includes("analyse")) levelNum = 4;
      else if (name.includes("evaluate")) levelNum = 5;
      else if (name.includes("create")) levelNum = 6;
      else levelNum = 2;
    }

    const marksNum =
      parseInt(String(q.marks ?? 5).replace(/[^0-9]/g, ""), 10) || 5;

    totalMarks += marksNum;

    const isAmbiguous = Boolean(q.is_ambiguous);
    const altLevel = q.alternative_level
      ? Number(q.alternative_level)
      : null;

    const low = Math.max(1, levelNum - 1);
    const high = Math.min(6, levelNum + 1);

    // Python confidence is 0-1; dashboard confidence is 0-100.
    let confidence = Number(q.confidence ?? 0.85);

    if (confidence <= 1) {
      confidence *= 100;
    }

    confidence = Math.min(
      100,
      Math.max(isAmbiguous ? 50 : 70, confidence || 85)
    );

    const bloomName =
      BLOOM_NAMES[levelNum] || assignedName || "Understand";

    const educatorSpectrum =
      q.educator_spectrum ||
      `L${low} (${BLOOM_NAMES[low]}) â†” L${high} (${BLOOM_NAMES[high]})`;

    return {
      id: q.id || q.fullIdentifier || `q_${idx + 1}`,
      question_number: String(questionNumber),
      text: String(questionText),
      marks: marksNum,
      bloom_level: levelNum,
      bloom_name: bloomName,
      action_verb: q.action_verb || "Analyze",
      knowledge_dimension: [
        "Factual",
        "Conceptual",
        "Procedural",
        "Metacognitive",
      ].includes(q.knowledge_dimension)
        ? q.knowledge_dimension
        : "Conceptual",
      difficulty: ["Low", "Medium", "High"].includes(q.difficulty)
        ? q.difficulty
        : "Medium",
      confidence,
      rationale:
        q.rationale ||
        q.reason ||
        "Cognitive classification based on the question's requirements.",
      level_up_suggestion:
        q.level_up_suggestion ||
        "Add a real-world problem-solving or evaluative dimension.",
      is_ambiguous: isAmbiguous,
      alternative_level:
        altLevel && altLevel >= 1 && altLevel <= 6 ? altLevel : null,
      ambiguity_note: q.ambiguity_note || null,
      educator_spectrum: educatorSpectrum,
      spectrum_lower: q.spectrum_lower ?? (levelNum > 1 ? low : null),
      spectrum_higher: q.spectrum_higher ?? (levelNum < 6 ? high : null),
      lower_perspective_rationale:
        q.lower_perspective_rationale ||
        "The question may require routine recall if students have already practised the procedure.",
      higher_perspective_rationale:
        q.higher_perspective_rationale ||
        "The question may require deeper reasoning when presented in an unfamiliar context.",
      contextual_factors:
        q.contextual_factors || [
          "Instructional Scaffolding",
          "Problem Novelty",
          "Multi-Step Reasoning",
        ],
      humanized_notes:
        q.humanized_notes ||
        "Cognitive difficulty can depend on students' previous exposure to the topic.",
    };
  });

  const totalQuestions = questions.length;
  const denominator = Math.max(totalQuestions, 1);

  const lotsCount = questions.filter(
    (q: any) => q.bloom_level <= 2
  ).length;

  const hotsCount = totalQuestions - lotsCount;

  const lotsPercentage = Math.round((lotsCount / denominator) * 100);
  const hotsPercentage = 100 - lotsPercentage;

  const dist: Record<
    string,
    { count: number; percentage: number; name: string }
  > = {};

  for (let l = 1; l <= 6; l++) {
    const count = questions.filter(
      (q: any) => q.bloom_level === l
    ).length;

    dist[String(l)] = {
      count,
      percentage: Math.round((count / denominator) * 100),
      name: BLOOM_NAMES[l],
    };
  }

  let dominantLevelNum = 1;
  let maxCount = -1;

  for (let l = 1; l <= 6; l++) {
    if (dist[String(l)].count > maxCount) {
      maxCount = dist[String(l)].count;
      dominantLevelNum = l;
    }
  }

  let balanceRating = "Balanced (NEP 2020 Aligned)";

  if (lotsPercentage >= 65) {
    balanceRating = "Recall-Heavy (Skewed towards LOTS)";
  } else if (hotsPercentage >= 75) {
    balanceRating = "High Cognitive Rigor (HOTS Dominant)";
  } else if (dist["3"].percentage >= 40) {
    balanceRating = "Application-Centric (Practical Problem Solving)";
  }

  const rawSummary = raw?.summary || {};

  const summary = {
    total_questions: totalQuestions,
    total_marks: totalMarks,
    lots_percentage: lotsPercentage,
    hots_percentage: hotsPercentage,
    lots_count: lotsCount,
    hots_count: hotsCount,
    distribution: dist,
    dominant_level: `L${dominantLevelNum} â€¢ ${BLOOM_NAMES[dominantLevelNum]}`,
    balance_rating: balanceRating,
    pedagogical_critique:
      rawSummary.pedagogical_critique ||
      `The paper contains ${lotsPercentage}% LOTS and ${hotsPercentage}% HOTS questions.`,
    recommendations: rawSummary.recommendations || [
      "Balance recall-based questions with analytical and application-based questions.",
      "Include case studies and evaluation-based questions.",
      "Ensure that marks correspond to the cognitive complexity of each question.",
    ],
    subject_detected:
      rawSummary.subject_detected || "General Academic Assessment",
    ambiguous_count:
      rawSummary.ambiguous ??
      questions.filter((q: any) => q.is_ambiguous).length,
    humanized_perception_note:
      rawSummary.humanized_perception_note ||
      "Cognitive rigor depends on the learning context and students' prior knowledge.",
  };

  return {
    questions,
    summary,
  };
}
