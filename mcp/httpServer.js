import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { createServer } from './server.js';
import { searchNotes, getNoteContent, listCategoriesAndNotes } from './knowledge.js';
import { getQuestionBankStats, queryQuestions, getQuestion, queryQuizzes } from './questionBank.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const OPENAPI_FILE = path.join(__dirname, 'openapi.json');

/**
 * Start the standalone HTTP / SSE server.
 */
export function startHttpServer(port = 3333) {
  // Map of active SSE transports: sessionId -> SSEServerTransport
  const transports = new Map();

  const server = http.createServer(async (req, res) => {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, Last-Event-ID');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    const pathname = parsedUrl.pathname;
    const searchParams = parsedUrl.searchParams;

    // 1. MCP SSE Transport: GET /sse
    if (pathname === '/sse' && req.method === 'GET') {
      console.log('[MCP SSE] Client connected to /sse');
      const mcpServer = createServer();
      const transport = new SSEServerTransport('/message', res);
      
      const sessionId = transport.sessionId;
      transports.set(sessionId, transport);

      transport.onclose = () => {
        console.log(`[MCP SSE] Transport closed: ${sessionId}`);
        transports.delete(sessionId);
      };

      await mcpServer.connect(transport);
      return;
    }

    // 2. MCP POST Message: POST /message
    if (pathname === '/message' && req.method === 'POST') {
      const sessionId = searchParams.get('sessionId');
      const transport = transports.get(sessionId);
      if (!transport) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `Session not found: ${sessionId}` }));
        return;
      }
      await transport.handlePostMessage(req, res);
      return;
    }

    // 3. REST Endpoint: OpenAPI Schema
    if (pathname === '/openapi.json' && req.method === 'GET') {
      if (fs.existsSync(OPENAPI_FILE)) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(fs.readFileSync(OPENAPI_FILE, 'utf-8'));
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'openapi.json not found' }));
      }
      return;
    }

    // 4. REST Endpoint: GET /api/search
    if (pathname === '/api/search' && req.method === 'GET') {
      const q = searchParams.get('q') || '';
      const category = searchParams.get('category') || 'all';
      const limit = parseInt(searchParams.get('limit') || '5', 10);
      const results = searchNotes(q, { category, limit });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ query: q, count: results.length, results }, null, 2));
      return;
    }

    // 5. REST Endpoint: GET /api/note
    if (pathname === '/api/note' && req.method === 'GET') {
      const notePath = searchParams.get('path') || '';
      const section = searchParams.get('section') || undefined;
      const lang = searchParams.get('lang') || undefined;
      const result = getNoteContent(notePath, { section, language: lang });
      res.writeHead(result.error ? 404 : 200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result, null, 2));
      return;
    }

    // 6. REST Endpoint: GET /api/question/random
    if (pathname === '/api/question/random' && req.method === 'GET') {
      const category = searchParams.get('category') || 'all';
      const includeSolution = searchParams.get('include_solution') === 'true';
      const question = getQuestion('random', { category, includeSolution, random: true });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(question, null, 2));
      return;
    }

    // 7. REST Endpoint: GET /api/question/:id or /api/question
    if (pathname.startsWith('/api/question') && req.method === 'GET') {
      const subPath = pathname.replace('/api/question', '').replace(/^\//, '');
      if (subPath) {
        const includeSolution = searchParams.get('include_solution') === 'true';
        const question = getQuestion(subPath, { includeSolution });
        res.writeHead(question.error ? 404 : 200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(question, null, 2));
        return;
      }
      const category = searchParams.get('category') || 'all';
      const keyword = searchParams.get('keyword') || '';
      const questions = queryQuestions({ category, keyword });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ count: questions.length, questions }, null, 2));
      return;
    }

    // 8. REST Endpoint: GET /api/quiz
    if (pathname === '/api/quiz' && req.method === 'GET') {
      const topic = searchParams.get('topic') || '';
      const category = searchParams.get('category') || 'all';
      const count = parseInt(searchParams.get('count') || '3', 10);
      const reveal = searchParams.get('reveal') === 'true';
      const quizzes = queryQuizzes({ topic, category, count, revealAnswer: reveal });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(quizzes, null, 2));
      return;
    }

    // 9. REST Endpoint: GET /api/stats
    if (pathname === '/api/stats' && req.method === 'GET') {
      const stats = getQuestionBankStats();
      const catalog = listCategoriesAndNotes('all');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ...stats, catalogSummary: Object.keys(catalog.catalog) }, null, 2));
      return;
    }

    // 10. Dashboard landing page: GET /
    if (pathname === '/' && req.method === 'GET') {
      const stats = getQuestionBankStats();
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>mlphdinterview MCP & Knowledge Server</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #24292f; }
    h1, h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
    .badge { display: inline-block; padding: 4px 10px; background: #0969da; color: #fff; border-radius: 20px; font-size: 13px; font-weight: 600; }
    .card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin: 16px 0; }
    pre { background: #24292f; color: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
    code { font-family: ui-monospace, SFMono-Regular, monospace; }
    a { color: #0969da; text-decoration: none; }
    a:hover { text-decoration: underline; }
    ul { padding-left: 20px; }
  </style>
</head>
<body>
  <h1>mlphdinterview MCP & Knowledge Server <span class="badge">Running</span></h1>
  <p><strong>CurryTang/mlphdinterview</strong> 模型上下文协议 (MCP) 与知识库题库服务已就绪。</p>

  <div class="card">
    <h3>📊 题库与知识库实时统计</h3>
    <ul>
      <li><strong>收录 Markdown 专业笔记</strong>：290+ 篇 (MLSys, System Design, Quant, LeetCode, MLCoding, BusinessAlgorithm)</li>
      <li><strong>字节跳动独家手撕与高频真题</strong>：${stats.bytedanceQuestionsCount} 题</li>
      <li><strong>分布式系统设计与 MLSys 实战案例</strong>：${stats.systemDesignAndMLSysCasesCount} 题</li>
      <li><strong>核心多选与理论速测 Flashcards</strong>：${stats.quizzesCount} 题</li>
    </ul>
  </div>

  <h2>接入方式指引</h2>

  <h3>1. ChatGPT 远程 MCP (SSE)</h3>
  <p>将此 Node 服务部署或通过隧道提供 ChatGPT 可访问的 HTTPS 地址。GitHub Pages 静态网站本身不运行 MCP 服务。</p>
  <p>在 ChatGPT 网页版 Plugins → Add custom MCP server 中，名称填 <code>mlphdinterview</code>，Server URL 填你的 HTTPS 地址加 <code>/sse</code>，鉴权选择 <code>No authentication</code>。创建并安装插件后，在对话中输入 @ 选择它。入口受账号与工作区权限限制。</p>
  <pre><code>https://your-mcp-host.example/sse</code></pre>
  <p>上方为示例域名，需替换。参见 <a href="https://developers.openai.com/api/docs/guides/custom-mcp-server" target="_blank" rel="noreferrer">OpenAI 官方接入说明</a>。</p>

  <h3>2. ChatGPT 网页版 (Custom GPT Action)</h3>
  <p>若将当前服务暴露至公网 (通过 ngrok 或 Cloudflare Tunnel)，在 Custom GPT 中直接导入 OpenAPI 架构：</p>
  <p>Schema 地址：<a href="/openapi.json" target="_blank"><code>/openapi.json</code></a>。将 schema 的 <code>servers[0].url</code> 改为你的 HTTPS 服务根地址。Actions 使用 REST / OpenAPI，与上面的 MCP 接入独立。</p>

  <h3>3. Claude Desktop / Cursor / Windsurf</h3>
  <pre><code>{
  "mcpServers": {
    "mlphdinterview": {
      "command": "node",
      "args": ["${path.resolve(__dirname, 'index.js')}"]
    }
  }
}</code></pre>

  <h2>REST API 快速测试</h2>
  <ul>
    <li><a href="/api/stats" target="_blank">/api/stats</a> - 题库总览统计</li>
    <li><a href="/api/question/random" target="_blank">/api/question/random</a> - 随机抽题 (模拟面试)</li>
    <li><a href="/api/question/bytedance-01" target="_blank">/api/question/bytedance-01</a> - 查看字节跳动手撕第 1 题</li>
    <li><a href="/api/search?q=KV%20Cache" target="_blank">/api/search?q=KV Cache</a> - 搜索知识点</li>
    <li><a href="/api/quiz?count=2&reveal=true" target="_blank">/api/quiz?count=2&reveal=true</a> - 测验选择题</li>
    <li><a href="/openapi.json" target="_blank">/openapi.json</a> - OpenAPI 规范描述文件</li>
  </ul>
</body>
</html>`);
      return;
    }

    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: `Not Found: ${pathname}` }));
  });

  server.listen(port, () => {
    console.log(`[mlphdinterview] MCP HTTP/SSE server listening on http://localhost:${port}`);
    console.log(`[mlphdinterview] MCP SSE endpoint: http://localhost:${port}/sse`);
    console.log(`[mlphdinterview] OpenAPI schema:  http://localhost:${port}/openapi.json`);
  });

  return server;
}
