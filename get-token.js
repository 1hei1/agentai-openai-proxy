/**
 * Agent.AI OAuth 一键获取 Token
 * 用法: node get-token.js [agent_id]
 * 
 * 流程：
 * 1. 自动注册 OAuth client
 * 2. 生成授权链接 → 浏览器打开登录
 * 3. 本地启动回调服务器自动接收 code
 * 4. 自动换取 token 并输出
 */

const http = require('http');
const https = require('https');
const crypto = require('crypto');
const { URL } = require('url');

const AGENT_ID = process.argv[2] || 'k0uu50s2ddfcjzo9';
const BASE = `https://api.agent.ai/api/v2/agents/${AGENT_ID}`;
const CALLBACK_PORT = 8080;
const REDIRECT_URI = `http://localhost:${CALLBACK_PORT}/callback`;

function httpsPost(url, data, contentType = 'application/json') {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const body = contentType === 'application/json' ? JSON.stringify(data) : data;
    const req = https.request({
      hostname: urlObj.hostname,
      path: urlObj.pathname,
      method: 'POST',
      headers: { 'Content-Type': contentType, 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
      let result = '';
      res.on('data', chunk => result += chunk);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, data: JSON.parse(result) }); }
        catch { resolve({ status: res.statusCode, data: result }); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function main() {
  console.log('🔐 Agent.AI OAuth Token 获取工具\n');
  console.log(`   Agent ID: ${AGENT_ID}`);
  console.log(`   Base URL: ${BASE}\n`);

  // Step 1: 注册 client
  console.log('1️⃣  注册 OAuth Client...');
  const reg = await httpsPost(`${BASE}/oauth/register`, {
    client_name: `cli-${Date.now()}`,
    redirect_uris: [REDIRECT_URI],
    grant_types: ['authorization_code', 'refresh_token'],
    response_types: ['code'],
    token_endpoint_auth_method: 'none',
  });

  if (!reg.data.client_id) {
    console.error('❌ 注册失败:', reg.data);
    process.exit(1);
  }
  const clientId = reg.data.client_id;
  console.log(`   ✅ client_id: ${clientId}\n`);

  // Step 2: 生成 PKCE
  const codeVerifier = crypto.randomBytes(48).toString('base64url');
  const codeChallenge = crypto.createHash('sha256').update(codeVerifier).digest('base64url');

  // Step 3: 构造授权 URL
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: clientId,
    redirect_uri: REDIRECT_URI,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256',
    scope: 'openid profile email mcp:access',
  });
  const authUrl = `${BASE}/authorize?${params}`;

  // Step 4: 启动本地回调服务器
  console.log('2️⃣  启动本地回调服务器...');
  
  const codePromise = new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, `http://localhost:${CALLBACK_PORT}`);
      if (url.pathname === '/callback') {
        const code = url.searchParams.get('code');
        const error = url.searchParams.get('error');
        
        if (code) {
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
          res.end('<h1>✅ 授权成功！</h1><p>可以关闭此页面了。</p><script>window.close()</script>');
          server.close();
          resolve(code);
        } else {
          res.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' });
          res.end(`<h1>❌ 授权失败</h1><p>${error || '未知错误'}</p>`);
          server.close();
          resolve(null);
        }
      }
    });

    server.listen(CALLBACK_PORT, () => {
      console.log(`   ✅ 回调服务器运行在 http://localhost:${CALLBACK_PORT}\n`);
    });

    // 60秒超时
    setTimeout(() => { server.close(); resolve(null); }, 120000);
  });

  console.log('3️⃣  请在浏览器中打开以下链接并登录授权：\n');
  console.log(`   ${authUrl}\n`);
  console.log('   ⏳ 等待授权回调... (120秒超时)\n');

  const code = await codePromise;

  if (!code) {
    console.error('❌ 未收到授权码');
    process.exit(1);
  }

  console.log(`   ✅ 收到授权码: ${code.substring(0, 20)}...\n`);

  // Step 5: 换取 token
  console.log('4️⃣  换取 Access Token...');
  const tokenBody = new URLSearchParams({
    grant_type: 'authorization_code',
    code: code,
    code_verifier: codeVerifier,
    client_id: clientId,
    redirect_uri: REDIRECT_URI,
  }).toString();

  const tokenResp = await httpsPost(`${BASE}/oauth/token`, tokenBody, 'application/x-www-form-urlencoded');

  if (!tokenResp.data.access_token) {
    console.error('❌ 换取 token 失败:', tokenResp.data);
    process.exit(1);
  }

  console.log('   ✅ Token 获取成功!\n');

  // 输出结果
  console.log('═══════════════════════════════════════════');
  console.log('📋 复制以下信息添加到 accounts.json:');
  console.log('═══════════════════════════════════════════\n');

  const account = {
    name: `account-${Date.now()}`,
    access_token: tokenResp.data.access_token,
    refresh_token: tokenResp.data.refresh_token || '',
    client_id: clientId,
    enabled: true,
  };

  console.log(JSON.stringify(account, null, 2));

  console.log('\n═══════════════════════════════════════════');
  console.log(`⏰ Token 有效期: ${tokenResp.data.expires_in / 3600} 小时`);
  console.log('   过期后会自动用 refresh_token 刷新');
  console.log('═══════════════════════════════════════════\n');

  // 也可以直接通过 API 添加
  console.log('💡 或者直接调 API 添加到代理服务:');
  console.log(`   curl -X POST http://localhost:9090/admin/accounts -H "Content-Type: application/json" -d '${JSON.stringify(account)}'`);
  console.log('');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
