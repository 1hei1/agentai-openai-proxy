const http=require('http'),https=require('https'),{randomUUID}=require('crypto'),fs=require('fs'),path=require('path'),crypto=require('crypto');
const CFG={BASE:process.env.AGENT_MCP_BASE||'https://api.agent.ai/api/v2/agents/k0uu50s2ddfcjzo9',TOOL:process.env.MCP_TOOL_NAME||'call_an_llm_copy',MODEL:process.env.MODEL_NAME||'agent-ai',KEY:process.env.API_KEY||'sk-agentai-proxy',PORT:parseInt(process.env.PORT||'9090'),FILE:process.env.ACCOUNTS_FILE||path.join(__dirname,'accounts.json'),LOG_DIR:process.env.LOG_DIR||path.join(__dirname,'logs')};
let accts=[],cidx=0;const pOAuth=new Map();

// ── 日志记录 ──
if(!fs.existsSync(CFG.LOG_DIR))fs.mkdirSync(CFG.LOG_DIR,{recursive:true});

function logRequest(data){
  const now=new Date();
  const file=path.join(CFG.LOG_DIR,`${now.toISOString().slice(0,10)}.jsonl`);
  const entry={
    id:data.id||randomUUID(),
    time:now.toISOString(),
    account:data.account||'unknown',
    model:data.model||CFG.MODEL,
    messages:data.messages||[],
    prompt:data.prompt||'',
    response:data.response||'',
    tool_calls:data.tool_calls||null,
    tools:data.tools?data.tools.map(t=>t.function?.name||t.name):[],
    duration_ms:data.duration_ms||0,
    error:data.error||null,
  };
  fs.appendFileSync(file,JSON.stringify(entry)+'\n');
}

function load(){
  if(fs.existsSync(CFG.FILE)){try{accts=JSON.parse(fs.readFileSync(CFG.FILE,'utf-8')).map((a,i)=>({nm:a.name||`account-${i}`,at:a.access_token||'',rt:a.refresh_token||'',ci:a.client_id||'',base:a.agent_mcp_base||CFG.BASE,exp:a.expires_at||0,req:0,err:0,last:0,on:a.enabled!==false}));console.log(`📋 ${accts.length} accounts`)}catch(e){console.error(e.message)}}
  if(!accts.length&&process.env.AGENT_ACCESS_TOKEN)accts.push({nm:'default',at:process.env.AGENT_ACCESS_TOKEN,rt:process.env.AGENT_REFRESH_TOKEN||'',ci:process.env.AGENT_CLIENT_ID||'',base:CFG.BASE,exp:0,req:0,err:0,last:0,on:true});
}
function save(){fs.writeFileSync(CFG.FILE,JSON.stringify(accts.map(a=>({name:a.nm,access_token:a.at,refresh_token:a.rt,client_id:a.ci,agent_mcp_base:a.base,expires_at:a.exp,enabled:a.on})),null,2))}

function jf(url,o={}){return new Promise((ok,no)=>{const u=new URL(url),r=https.request({hostname:u.hostname,port:443,path:u.pathname+u.search,method:o.method||'GET',headers:{'Content-Type':'application/json',...o.headers}},s=>{let d='';s.on('data',c=>d+=c);s.on('end',()=>{try{ok({s:s.statusCode,d:JSON.parse(d)})}catch{ok({s:s.statusCode,d})}})});r.on('error',no);r.setTimeout(120000,()=>{r.destroy();no(new Error('timeout'))});if(o.body)r.write(o.body);r.end()})}
function fp(url,data){return new Promise((ok,no)=>{const u=new URL(url),b=typeof data==='string'?data:new URLSearchParams(data).toString(),r=https.request({hostname:u.hostname,path:u.pathname,method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded','Content-Length':Buffer.byteLength(b)}},s=>{let d='';s.on('data',c=>d+=c);s.on('end',()=>{try{ok({s:s.statusCode,d:JSON.parse(d)})}catch{ok({s:s.statusCode,d})}})});r.on('error',no);r.write(b);r.end()})}

async function refresh(a){if(!a.rt||!a.ci)return false;const r=await fp(`${a.base}/oauth/token`,{grant_type:'refresh_token',refresh_token:a.rt,client_id:a.ci});if(r.s===200&&r.d.access_token){a.at=r.d.access_token;if(r.d.refresh_token)a.rt=r.d.refresh_token;a.exp=Date.now()+(r.d.expires_in||86400)*1000-60000;save();console.log(`🔄 ${a.nm}`);return true}return false}
async function tok(a){if(a.exp>0&&Date.now()>a.exp&&a.rt)await refresh(a);return a.at}

const toolNameCache=new Map();
async function discoverToolName(acc){
  const ck=acc.base;if(toolNameCache.has(ck))return toolNameCache.get(ck);
  const t=await tok(acc);if(!t)return CFG.TOOL;
  try{const r=await jf(`${acc.base}/mcp`,{method:'POST',headers:{Authorization:`Bearer ${t}`},body:JSON.stringify({jsonrpc:'2.0',method:'tools/list',id:1})});
    if(r.s===200&&r.d.result?.tools?.length){const nm=r.d.result.tools[0].name;toolNameCache.set(ck,nm);console.log(`🔍 [${acc.nm}] tool: ${nm}`);return nm}}catch(e){console.error(`discover fail ${acc.nm}:`,e.message)}
  return CFG.TOOL;
}

function pick(){const en=accts.filter(a=>a.on&&a.at);if(!en.length)throw new Error('No accounts');const s=[...en].sort((a,b)=>a.err-b.err);cidx=cidx%s.length;return s[cidx++]}

// 记录最后使用的账号名（用于日志）
let lastAccountUsed='unknown';

async function mcp(method,params,retry=0){
  const mx=Math.min(accts.filter(a=>a.on).length,3),a=pick(),t=await tok(a);
  lastAccountUsed=a.nm;
  if(!t){if(retry<mx){a.on=false;return mcp(method,params,retry+1)}throw new Error('No token')}
  if(method==='tools/call'&&params){const rn=await discoverToolName(a);params={...params,name:rn}}
  const p={jsonrpc:'2.0',method,id:1};if(params)p.params=params;
  try{
    const r=await jf(`${a.base}/mcp`,{method:'POST',headers:{Authorization:`Bearer ${t}`},body:JSON.stringify(p)});
    if(r.s===401||r.s===403){if(await refresh(a)&&retry<mx)return mcp(method,params,retry+1);a.err++;throw new Error('Auth fail')}
    if(r.s!==200||r.d.error){a.err++;if(retry<mx)return mcp(method,params,retry+1);throw new Error(`MCP ${r.s}`)}
    a.req++;a.last=Date.now();a.err=Math.max(0,a.err-1);console.log(`✅ [${a.nm}] #${a.req}`);return r.d.result||{};
  }catch(e){if(retry<mx){a.err++;return mcp(method,params,retry+1)}throw e}
}

function m2p(msgs,tools){
  const p=[];
  if(tools?.length){const td=tools.map(t=>({name:t.function.name,description:t.function.description||'',parameters:t.function.parameters}));p.push('你可以使用以下工具。如需调用，严格用JSON回复：\n{"tool_calls":[{"name":"工具名","arguments":{参数}}]}\n\n可用工具：\n```json\n'+JSON.stringify(td,null,2)+'\n```\n\n不需要工具则直接回复文字。')}
  for(const m of msgs){if(m.role==='system')p.push(`[System] ${m.content}`);else if(m.role==='user')p.push(`[User] ${m.content}`);else if(m.role==='assistant'){if(m.content)p.push(`[Assistant] ${m.content}`);if(m.tool_calls)for(const tc of m.tool_calls)p.push(`[Called: ${(tc.function||tc).name}(${(tc.function||tc).arguments})]`)}else if(m.role==='tool')p.push(`[Tool ${m.tool_call_id||m.name}] ${m.content}`)}
  return p.join('\n\n');
}
function ptc(t){try{const d=JSON.parse(t.trim());if(d.tool_calls)return d.tool_calls}catch{}for(const re of[/```json\s*(\{[\s\S]*?\})\s*```/,/\{"tool_calls"\s*:\s*\[[\s\S]*?\]\}/]){const m=t.match(re);if(m){try{const d=JSON.parse(m[1]||m[0]);if(d.tool_calls)return d.tool_calls}catch{}}}return null}

function bResp(text,model,tools){
  const id=`chatcmpl-${randomUUID().slice(0,12)}`,cr=Math.floor(Date.now()/1000),tc=tools?.length?ptc(text):null;
  if(tc)return{id,object:'chat.completion',created:cr,model,choices:[{index:0,message:{role:'assistant',content:null,tool_calls:tc.map((c,i)=>({index:i,id:`call_${randomUUID().slice(0,8)}`,type:'function',function:{name:c.name,arguments:JSON.stringify(c.arguments||{})}}))},finish_reason:'tool_calls'}],usage:{prompt_tokens:0,completion_tokens:0,total_tokens:0}};
  return{id,object:'chat.completion',created:cr,model,choices:[{index:0,message:{role:'assistant',content:text},finish_reason:'stop'}],usage:{prompt_tokens:0,completion_tokens:0,total_tokens:0}};
}
function bStream(text,model,tools){const r=bResp(text,model,tools),c=r.choices[0];return`data: ${JSON.stringify({id:r.id,object:'chat.completion.chunk',created:r.created,model,choices:[{index:0,delta:c.message,finish_reason:null}]})}\n\ndata: ${JSON.stringify({id:r.id,object:'chat.completion.chunk',created:r.created,model,choices:[{index:0,delta:{},finish_reason:c.finish_reason}]})}\n\ndata: [DONE]\n\n`}

function pb(req){return new Promise(r=>{let b='';req.on('data',c=>b+=c);req.on('end',()=>{try{r(JSON.parse(b))}catch{r({})}})})}
function sj(res,s,d){res.writeHead(s,{'Content-Type':'application/json','Access-Control-Allow-Origin':'*'});res.end(JSON.stringify(d))}
function sh(res,h){res.writeHead(200,{'Content-Type':'text/html;charset=utf-8'});res.end(h)}
function vk(req){if(!CFG.KEY)return true;return(req.headers.authorization||'')===`Bearer ${CFG.KEY}`}

function HTML(){return fs.existsSync(path.join(__dirname,'admin.html'))?fs.readFileSync(path.join(__dirname,'admin.html'),'utf-8'):'<h1>admin.html not found</h1>'}

const srv=http.createServer(async(req,res)=>{
  if(req.method==='OPTIONS'){res.writeHead(200,{'Access-Control-Allow-Origin':'*','Access-Control-Allow-Methods':'GET,POST,PUT,DELETE,OPTIONS','Access-Control-Allow-Headers':'Content-Type,Authorization'});return res.end()}
  const url=req.url.split('?')[0];
  try{
    if((url==='/'||url==='/admin')&&req.method==='GET')return sh(res,HTML());
    if(url==='/health')return sj(res,200,{status:'ok',accounts:accts.map(a=>({name:a.nm,enabled:a.on,hasToken:!!a.at,requests:a.req,errors:a.err,lastUsed:a.last||null}))});

    if(url==='/admin/accounts'&&req.method==='GET')return sj(res,200,{total:accts.length,accounts:accts.map((a,i)=>({index:i,name:a.nm,enabled:a.on,hasToken:!!a.at,hasRefreshToken:!!a.rt,requests:a.req,errors:a.err,lastUsed:a.last||null}))});
    if(url==='/admin/accounts'&&req.method==='POST'){const b=await pb(req);accts.push({nm:b.name||`account-${accts.length}`,at:b.access_token||'',rt:b.refresh_token||'',ci:b.client_id||'',base:b.agent_mcp_base||CFG.BASE,exp:0,req:0,err:0,last:0,on:b.enabled!==false});save();return sj(res,200,{status:'ok',total:accts.length})}
    if(url.match(/^\/admin\/accounts\/\d+$/)&&req.method==='PUT'){const i=parseInt(url.split('/').pop()),b=await pb(req);if(i<0||i>=accts.length)return sj(res,404,{error:'Not found'});if('enabled'in b)accts[i].on=b.enabled;if(b.access_token)accts[i].at=b.access_token;if(b.refresh_token)accts[i].rt=b.refresh_token;if(b.client_id)accts[i].ci=b.client_id;save();return sj(res,200,{status:'ok'})}
    if(url.match(/^\/admin\/accounts\/\d+$/)&&req.method==='DELETE'){const i=parseInt(url.split('/').pop());if(i<0||i>=accts.length)return sj(res,404,{error:'Not found'});const rm=accts.splice(i,1)[0];save();return sj(res,200,{status:'ok',removed:rm.nm})}

    if(url==='/admin/oauth/start'&&req.method==='POST'){
      const b=await pb(req),aid=b.agent_id||'k0uu50s2ddfcjzo9',base=`https://api.agent.ai/api/v2/agents/${aid}`;
      const reg=await jf(`${base}/oauth/register`,{method:'POST',body:JSON.stringify({client_name:`web-${Date.now()}`,redirect_uris:['http://localhost:8080/callback'],grant_types:['authorization_code','refresh_token'],response_types:['code'],token_endpoint_auth_method:'none'})});
      if(!reg.d.client_id)return sj(res,500,{error:'Register failed'});
      const cv=crypto.randomBytes(48).toString('base64url'),cc=crypto.createHash('sha256').update(cv).digest('base64url'),sid=crypto.randomUUID();
      const ps=new URLSearchParams({response_type:'code',client_id:reg.d.client_id,redirect_uri:'http://localhost:8080/callback',code_challenge:cc,code_challenge_method:'S256',scope:'openid profile email mcp:access'});
      pOAuth.set(sid,{aid,base,ci:reg.d.client_id,cv,t:Date.now()});
      for(const[k,v]of pOAuth)if(Date.now()-v.t>600000)pOAuth.delete(k);
      return sj(res,200,{session_id:sid,auth_url:`${base}/authorize?${ps}`,client_id:reg.d.client_id});
    }
    if(url==='/admin/oauth/exchange'&&req.method==='POST'){
      const b=await pb(req),s=pOAuth.get(b.session_id);if(!s)return sj(res,400,{error:'Session expired'});
      const tr=await fp(`${s.base}/oauth/token`,{grant_type:'authorization_code',code:b.code,code_verifier:s.cv,client_id:s.ci,redirect_uri:'http://localhost:8080/callback'});
      if(!tr.d.access_token)return sj(res,400,{error:`Failed: ${JSON.stringify(tr.d)}`});
      accts.push({nm:`oauth-${accts.length}`,at:tr.d.access_token,rt:tr.d.refresh_token||'',ci:s.ci,base:s.base,exp:Date.now()+(tr.d.expires_in||86400)*1000-60000,req:0,err:0,last:0,on:true});
      save();pOAuth.delete(b.session_id);return sj(res,200,{status:'ok',total:accts.length});
    }
    if(url==='/admin/reload'&&req.method==='POST'){load();return sj(res,200,{status:'ok',total:accts.length})}

    // ── 日志查询 API ──
    if(url==='/admin/logs'&&req.method==='GET'){
      const qs=new URL(req.url,`http://localhost`).searchParams;
      const date=qs.get('date')||new Date().toISOString().slice(0,10);
      const limit=parseInt(qs.get('limit')||'50');
      const file=path.join(CFG.LOG_DIR,`${date}.jsonl`);
      if(!fs.existsSync(file))return sj(res,200,{date,total:0,logs:[]});
      const lines=fs.readFileSync(file,'utf-8').trim().split('\n').filter(Boolean);
      const logs=lines.slice(-limit).map(l=>{try{return JSON.parse(l)}catch{return null}}).filter(Boolean).reverse();
      return sj(res,200,{date,total:lines.length,showing:logs.length,logs});
    }

    if(url==='/v1/models'){if(!vk(req))return sj(res,401,{error:'Invalid key'});return sj(res,200,{object:'list',data:[{id:CFG.MODEL,object:'model',created:Math.floor(Date.now()/1000),owned_by:'agent-ai'}]})}
    if(url==='/v1/chat/completions'&&req.method==='POST'){
      if(!vk(req))return sj(res,401,{error:'Invalid key'});
      const body=await pb(req),prompt=m2p(body.messages||[],body.tools);
      const startTime=Date.now();
      let text='',error=null,respTc=null;
      try{
        const result=await mcp('tools/call',{name:CFG.TOOL,arguments:{user_input:prompt}});
        text=(result.content||[]).filter(c=>c.type==='text').map(c=>typeof c.text==='string'?c.text:typeof c.text==='object'?JSON.stringify(c.text):String(c.text)).join('\n');
      }catch(e){error=e.message;throw e}
      finally{
        // 记录日志
        const tc=body.tools?.length?ptc(text):null;
        logRequest({
          account:lastAccountUsed,
          model:body.model||CFG.MODEL,
          messages:body.messages||[],
          prompt,
          response:text,
          tool_calls:tc,
          tools:body.tools||[],
          duration_ms:Date.now()-startTime,
          error,
        });
      }
      if(body.stream){res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache',Connection:'keep-alive','Access-Control-Allow-Origin':'*'});res.end(bStream(text,body.model||CFG.MODEL,body.tools))}
      else sj(res,200,bResp(text,body.model||CFG.MODEL,body.tools));
      return;
    }
    sj(res,404,{error:'Not found'});
  }catch(e){console.error('Error:',e);sj(res,500,{error:{message:e.message,type:'bad_response_status_code',param:'',code:'bad_response_status_code'}})}
});

load();
srv.listen(CFG.PORT,'0.0.0.0',()=>{console.log(`🚀 http://0.0.0.0:${CFG.PORT} | accounts:${accts.length} | key:${CFG.KEY} | logs:${CFG.LOG_DIR}`)});
