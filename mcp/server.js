import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import {
  getNotesIndex,
  searchNotes,
  getNoteContent,
  listCategoriesAndNotes,
} from './knowledge.js';
import {
  getByteDanceQuestions,
  getPracticeQuizzes,
  getQuestionBankStats,
  queryQuestions,
  getQuestion,
  queryQuizzes,
} from './questionBank.js';

/**
 * Create and configure the mlphdinterview MCP server instance.
 */
export function createServer() {
  const server = new McpServer({
    name: 'mlphdinterview-mcp',
    version: '1.0.0',
    description: 'Knowledge base and interview question bank for Machine Learning Systems, System Design, Quant, LeetCode, and ML Engineering (CurryTang/mlphdinterview).',
  });

  // ==========================================
  // Tool 1: search_knowledge_base
  // ==========================================
  server.tool(
    'search_knowledge_base',
    'Search across 290+ technical interview preparation notes (ML Systems, System Design, Quant, LeetCode, ML Coding, Business Algorithms, Jobs) for keywords or concepts. Returns ranked matches with snippets and section headings.',
    {
      query: z.string().describe('Search term or concept (e.g. "MAI-Thinking-1", "KV Cache", "Kafka zero copy", "PagedAttention", "Consistent Hashing", "8-Byte Aligned Memory Allocator", "Martingale")'),
      category: z.enum(['all', 'Mlsys', 'SystemDesign', 'quant', 'Leetcode', 'MLCoding', 'BusinessAlgorithm', 'Jobs']).optional().default('all').describe('Filter by domain category'),
      language: z.enum(['all', 'zh', 'en']).optional().default('all').describe('Preferred note language (zh: Chinese, en: English, all: Both)'),
      limit: z.number().min(1).max(20).optional().default(5).describe('Maximum number of ranked results to return'),
    },
    async ({ query, category, language, limit }) => {
      const results = searchNotes(query, { category, language, limit });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                query,
                matchedCount: results.length,
                results: results.map((r) => ({
                  title: r.title,
                  path: r.path,
                  category: r.category,
                  language: r.language,
                  snippet: r.snippet,
                  keySections: r.headings,
                })),
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================
  // Tool 2: get_note
  // ==========================================
  server.tool(
    'get_note',
    'Retrieve full or section-specific content of a note from the knowledge base by its file path or note title. Supports both Chinese and English versions.',
    {
      note_path: z.string().describe('Relative path to note (e.g. "notes/Mlsys/MLSYS06 LLM Inference Engine Architecture nano-vllm.md") or note title (e.g. "nano-vllm" or "Kafka")'),
      section: z.string().optional().describe('Optional heading keyword to extract only that section (e.g. "PagedAttention", "Mechanical Sympathy", "Zero-Copy")'),
      language: z.enum(['zh', 'en']).optional().describe('Preferred language version'),
      max_length: z.number().optional().default(8000).describe('Maximum character length to return to conserve context window'),
    },
    async ({ note_path, section, language, max_length }) => {
      const note = getNoteContent(note_path, { section, language, maxLength: max_length });
      if (note.error) {
        return {
          content: [{ type: 'text', text: JSON.stringify({ error: note.error }, null, 2) }],
          isError: true,
        };
      }

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                title: note.title,
                path: note.path,
                category: note.category,
                language: note.language,
                headings: note.headings.map((h) => `${'#'.repeat(h.level)} ${h.text}`),
                content: note.content,
                isTruncated: note.isTruncated,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================
  // Tool 3: list_knowledge_topics
  // ==========================================
  server.tool(
    'list_knowledge_topics',
    'List all available knowledge categories and note topics in the repository.',
    {
      category: z.enum(['all', 'Mlsys', 'SystemDesign', 'quant', 'Leetcode', 'MLCoding', 'BusinessAlgorithm', 'Jobs']).optional().default('all').describe('Filter by category'),
    },
    async ({ category }) => {
      const data = listCategoriesAndNotes(category);
      return {
        content: [{ type: 'text', text: JSON.stringify(data, null, 2) }],
      };
    }
  );

  // ==========================================
  // Tool 4: get_interview_question
  // ==========================================
  server.tool(
    'get_interview_question',
    'Fetch high-frequency interview questions from the curated question bank (including the 65 ByteDance interview questions, System Design case studies, MLSys engineering challenges, Quant puzzles, and LeetCode algorithmic patterns). Supports random sampling for mock interviews.',
    {
      category: z.enum(['all', 'bytedance_65', 'Leetcode', 'MLCoding', 'SystemDesign', 'Mlsys', 'quant']).optional().default('all').describe('Filter questions by category or bytedance_65'),
      question_id: z.string().optional().describe('Specific question ID (e.g. "01", "bytedance-18", "bytedance-65", "case-chatgpt-sse")'),
      keyword: z.string().optional().describe('Filter questions by keyword or topic'),
      random: z.boolean().optional().default(false).describe('Randomly pick a question for mock interview'),
      include_solution: z.boolean().optional().default(false).describe('Whether to include reference solution immediately (default: false for mock interview testing)'),
    },
    async ({ category, question_id, keyword, random, include_solution }) => {
      if (question_id || random) {
        const q = getQuestion(question_id || 'random', { includeSolution: include_solution, category, random });
        return {
          content: [{ type: 'text', text: JSON.stringify(q, null, 2) }],
        };
      }

      const questions = queryQuestions({ category, keyword, limit: 10 });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                count: questions.length,
                questions: questions.map((q) => ({
                  id: q.id,
                  titleZh: q.titleZh,
                  titleEn: q.titleEn,
                  category: q.category,
                  complexity: q.complexity,
                  keyPoints: q.keyPoints,
                  specification: q.spec,
                  codePath: q.codePath,
                })),
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================
  // Tool 5: get_practice_quiz
  // ==========================================
  server.tool(
    'get_practice_quiz',
    'Fetch interactive multiple-choice flashcards and quizzes from the question bank. Useful for testing candidate knowledge in mock interviews and flashcard review.',
    {
      topic: z.string().optional().default('').describe('Topic keyword to filter quizzes (e.g. "attention", "sse", "kafka", "pagedattention", "passenger")'),
      category: z.enum(['all', 'Mlsys', 'SystemDesign', 'quant', 'MLCoding', 'Leetcode']).optional().default('all').describe('Filter quizzes by category'),
      count: z.number().min(1).max(10).optional().default(3).describe('Number of quiz questions to retrieve'),
      reveal_answer: z.boolean().optional().default(false).describe('Whether to reveal correct answer and explanation (false allows ChatGPT to quiz the candidate first)'),
    },
    async ({ topic, category, count, reveal_answer }) => {
      const quizzes = queryQuizzes({ topic, category, count, revealAnswer: reveal_answer });
      return {
        content: [{ type: 'text', text: JSON.stringify(quizzes, null, 2) }],
      };
    }
  );

  // ==========================================
  // Tool 6: evaluate_answer
  // ==========================================
  server.tool(
    'evaluate_answer',
    'Get reference solution, key scoring points, time/space complexity bounds, and standard evaluation criteria for an interview question or quiz to grade candidate answers.',
    {
      question_id: z.string().describe('The question ID (e.g. "bytedance-01", "case-chatgpt-sse", "quiz-sse-nginx")'),
      candidate_answer: z.string().optional().describe('Optional answer provided by the candidate to compare against reference solution'),
    },
    async ({ question_id, candidate_answer }) => {
      if (question_id.startsWith('quiz-')) {
        const quizzes = getPracticeQuizzes();
        const quiz = quizzes.find((q) => q.id === question_id);
        if (!quiz) {
          return { content: [{ type: 'text', text: JSON.stringify({ error: `Quiz not found: ${question_id}` }, null, 2) }] };
        }
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(
                {
                  id: quiz.id,
                  question: quiz.question,
                  correctAnswer: quiz.answer,
                  explanation: quiz.explanation,
                  sourceNote: quiz.sourceNote,
                  candidateAnswer: candidate_answer || null,
                  isMatch: candidate_answer ? candidate_answer.trim().toUpperCase() === quiz.answer : null,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      const q = getQuestion(question_id, { includeSolution: true });
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(
              {
                id: q.id,
                titleZh: q.titleZh,
                titleEn: q.titleEn,
                complexityExpectation: q.complexity,
                mustCoverKeyPoints: q.keyPoints,
                problemSpecification: q.specification,
                referenceSolution: q.referenceSolution,
                candidateAnswerProvided: candidate_answer ? true : false,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================
  // Resources
  // ==========================================
  server.resource('catalog', 'mlphd://catalog', async (uri) => {
    const catalog = listCategoriesAndNotes('all');
    return {
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(catalog, null, 2),
          mimeType: 'application/json',
        },
      ],
    };
  });

  server.resource('bytedance-questions', 'mlphd://questions/bytedance', async (uri) => {
    const qs = getByteDanceQuestions();
    return {
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(qs, null, 2),
          mimeType: 'application/json',
        },
      ],
    };
  });

  server.resource('stats', 'mlphd://stats', async (uri) => {
    const stats = getQuestionBankStats();
    const index = getNotesIndex();
    return {
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(
            {
              ...stats,
              totalIndexedNotes: index.totalCount,
              categories: Object.keys(index.byCategory),
            },
            null,
            2
          ),
          mimeType: 'application/json',
        },
      ],
    };
  });

  // ==========================================
  // Prompts
  // ==========================================
  server.prompt(
    'mock-interview',
    'Start a realistic technical mock interview in MLSys, System Design, Quant, or LeetCode.',
    {
      domain: z.enum(['MLSys', 'SystemDesign', 'Quant', 'Algorithms', 'MLCoding']).default('MLSys').describe('Interview domain'),
      difficulty: z.enum(['Medium', 'Hard']).default('Hard').describe('Target interview difficulty'),
      company_style: z.string().default('ByteDance / OpenAI').describe('Company interview style'),
    },
    ({ domain, difficulty, company_style }) => {
      return {
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: `You are a Principal Engineer and interviewer at ${company_style}. You are conducting a rigorous ${difficulty}-level mock interview in the ${domain} domain.

Interview Instructions:
1. First, use "get_interview_question" (with random: true and category matching ${domain}) or "get_practice_quiz" to pick an authentic interview problem from the CurryTang/mlphdinterview question bank.
2. Present ONLY the problem description, constraints, and input/output specifications to the candidate. Do NOT give away the answer or hints right away.
3. Ask the candidate to explain their thought process, algorithmic complexity, architectural trade-offs, or write out code.
4. If the candidate gets stuck or gives an incomplete answer, provide subtle leading hints based on the "keyPoints" in the question bank.
5. Once the candidate finishes, call "evaluate_answer" to compare their answer against the knowledge base reference solution, grading them on: Correctness, Complexity, Edge Cases, and Production System Scalability.`,
            },
          },
        ],
      };
    }
  );

  server.prompt(
    'topic-drill',
    'Conduct a targeted technical review and quiz session on any concept from the mlphdinterview knowledge base.',
    {
      topic: z.string().describe('Concept or topic (e.g. "KV Cache", "Kafka zero copy", "FlashAttention", "SSE", "Martingale")'),
    },
    ({ topic }) => {
      return {
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: `You are an expert technical coach preparing a candidate for top ML Systems, Quant, and System Design interviews.

Conduct a deep-dive technical review on: "${topic}".
1. Call "search_knowledge_base" with query "${topic}" to retrieve the canonical notes from CurryTang/mlphdinterview.
2. Summarize the core theoretical principles, mathematical derivations, and physical systems trade-offs without promotional fluff.
3. Call "get_practice_quiz" or "get_interview_question" to test the candidate with a challenging question on this exact topic.
4. Guide the candidate step-by-step through solving it.`,
            },
          },
        ],
      };
    }
  );

  return server;
}
