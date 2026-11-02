#!/usr/bin/env node

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createServer } from './server.js';
import { startHttpServer } from './httpServer.js';

const args = process.argv.slice(2);
const isHttpMode = args.includes('--http') || args.includes('--sse');

if (isHttpMode) {
  // HTTP / SSE Mode (for ChatGPT Web / Custom GPTs / Remote MCP)
  const portArgIndex = args.findIndex((a) => a === '-p' || a === '--port');
  let port = 3333;
  if (portArgIndex >= 0 && args[portArgIndex + 1]) {
    port = parseInt(args[portArgIndex + 1], 10);
  } else {
    const nextArg = args[args.findIndex((a) => a === '--http' || a === '--sse') + 1];
    if (nextArg && /^\d+$/.test(nextArg)) {
      port = parseInt(nextArg, 10);
    }
  }

  startHttpServer(port);
} else {
  // Stdio mode for clients such as Claude Desktop and Cursor.
  const server = createServer();
  const transport = new StdioServerTransport();

  // Redirect console.log to stderr so stdout remains clean JSON-RPC for MCP protocol
  const originalLog = console.log;
  console.log = (...msgs) => console.error('[MCP Info]', ...msgs);

  server.connect(transport).catch((err) => {
    console.error('[MCP Fatal Error]', err);
    process.exit(1);
  });

  process.on('SIGINT', async () => {
    await server.close();
    process.exit(0);
  });
}
