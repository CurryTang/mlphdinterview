const guideUrl = 'https://github.com/CurryTang/mlphdinterview/blob/main/mcp/README.md';
const officialUrl = 'https://developers.openai.com/api/docs/guides/custom-mcp-server';

const copy = {
  zh: {
    title: '把笔记接入 ChatGPT',
    status: 'MCP 服务已实现 · 需自行部署',
    description: '让 ChatGPT 检索笔记、抽取面试题、组织测验并查阅参考答案。仓库提供 6 个 MCP 工具，通过 SSE 接入。',
    hosting: '本站由 GitHub Pages 托管，仅提供静态阅读页面，没有托管在线 MCP 服务。下方地址是示例，请替换成你部署的服务地址。',
    setup: '展开 ChatGPT MCP 配置步骤',
    start: '1. 启动服务',
    startBody: '在本地克隆仓库并启动 Node 服务：',
    expose: '在另一个终端运行 ngrok http 3333，或将服务部署到支持 Node 的服务器，获得 ChatGPT 可访问的 HTTPS 地址。服务需要持续运行。',
    connect: '2. 在 ChatGPT 中添加',
    steps: [
      '打开 ChatGPT 网页版的 Plugins（插件）页面，点击加号 → Add custom MCP server。是否可用取决于账号和工作区权限。',
      '名称填写 mlphdinterview，Server URL 填写你的 HTTPS 地址加 /sse；此服务未配置鉴权，Authentication 选择 No authentication。',
      '创建并安装插件，在对话中输入 @ 选择它，再发送下面的示例提示词。',
    ],
    endpoint: 'Server URL 示例（需替换域名）',
    tryIt: '3. 开始复习',
    prompt: '请使用 mlphdinterview，先调用 search_knowledge_base 检索 KV Cache，再调用 get_interview_question 随机抽取一道 MLSys 面试题。先只出题，等我回答后再查看参考答案。',
    alternatives: 'Custom GPT Actions / 其他 MCP 客户端',
    actions: 'Custom GPT 的 Actions 使用 REST / OpenAPI：在 GPT Editor → Configure → Create new action 导入服务的 /openapi.json，并将 schema 的 servers[0].url 改为你的 HTTPS 服务根地址。它与上面的 MCP 配置是两种独立接入方式。',
    clients: 'Claude Desktop、Cursor 等支持 stdio 的客户端，可运行 node /绝对路径/mlphdinterview/mcp/index.js。完整配置与工具参数见接入文档。',
    docs: '完整接入文档 ↗',
    official: 'OpenAI 官方配置说明 ↗',
  },
  en: {
    title: 'Connect these notes to ChatGPT',
    status: 'MCP server implemented · Self-hosting required',
    description: 'Let ChatGPT search notes, draw interview questions, run quizzes, and retrieve reference answers with 6 MCP tools over SSE.',
    hosting: 'This GitHub Pages site serves static reading pages. It does not host a live MCP server. The URL below is an example; replace it with your deployed server address.',
    setup: 'Show ChatGPT MCP setup steps',
    start: '1. Start the server',
    startBody: 'Clone the repository locally and start the Node server:',
    expose: 'In another terminal, run ngrok http 3333, or deploy to a server that supports Node, to obtain an HTTPS address accessible to ChatGPT. Keep the service running.',
    connect: '2. Add it to ChatGPT',
    steps: [
      'Open Plugins in ChatGPT on the web. Select the plus button → Add custom MCP server. Availability depends on your account and workspace permissions.',
      'Use mlphdinterview as the name. Set Server URL to your HTTPS address followed by /sse. This server has no authentication configured, so select No authentication.',
      'Create and install the plugin. Type @ in a conversation to select it, then send the example prompt below.',
    ],
    endpoint: 'Example Server URL (replace the domain)',
    tryIt: '3. Start reviewing',
    prompt: 'Use mlphdinterview: first call search_knowledge_base to look up KV Cache, then call get_interview_question to draw a random MLSys question. Only show the question first; wait for my answer before retrieving the reference solution.',
    alternatives: 'Custom GPT Actions / Other MCP clients',
    actions: 'Custom GPT Actions use REST / OpenAPI: in GPT Editor → Configure → Create new action, import your server’s /openapi.json and replace servers[0].url in the schema with your HTTPS server root URL. This is a separate integration from MCP above.',
    clients: 'For clients that support stdio, such as Claude Desktop and Cursor, run node /absolute/path/mlphdinterview/mcp/index.js. See the integration guide for configuration and tool parameters.',
    docs: 'Full integration guide ↗',
    official: 'Official OpenAI setup guide ↗',
  },
};

export default function McpSetup({ language }) {
  const text = copy[language] ?? copy.zh;
  return (
    <section className="home-mcp" id="mcp" aria-labelledby="mcp-title">
      <div className="section-heading">
        <p className="eyebrow">CHATGPT / MCP</p>
        <h2 id="mcp-title">{text.title}</h2>
      </div>
      <p className="mcp-status">{text.status}</p>
      <p>{text.description}</p>
      <p className="mcp-hosting">{text.hosting}</p>
      <details className="mcp-setup">
        <summary>{text.setup}</summary>
        <div className="mcp-setup-grid">
          <div>
            <h3>{text.start}</h3>
            <p>{text.startBody}</p>
            <pre><code>{'git clone https://github.com/CurryTang/mlphdinterview.git\ncd mlphdinterview\nnpm ci\nnpm run mcp:http'}</code></pre>
            <p>{text.expose}</p>
          </div>
          <div>
            <h3>{text.connect}</h3>
            <ol>{text.steps.map((step) => <li key={step}>{step}</li>)}</ol>
            <p>{text.endpoint}</p>
            <pre><code>https://your-mcp-host.example/sse</code></pre>
          </div>
        </div>
        <h3>{text.tryIt}</h3>
        <blockquote>{text.prompt}</blockquote>
      </details>
      <details className="mcp-setup">
        <summary>{text.alternatives}</summary>
        <p>{text.actions}</p>
        <p>{text.clients}</p>
      </details>
      <div className="mcp-links">
        <a href={guideUrl} target="_blank" rel="noreferrer">{text.docs}</a>
        <a href={officialUrl} target="_blank" rel="noreferrer">{text.official}</a>
      </div>
    </section>
  );
}
