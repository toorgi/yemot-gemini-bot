import express from 'express';
import { YemotRouter, ExitError } from 'yemot-router2';
import { GoogleGenerativeAI } from '@google/generative-ai';
import YemotApi from 'yemot-api';
import fs from 'fs';
import path from 'path';

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const apiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',').map(k => k.trim()).filter(Boolean);

if (!apiKeys.length) {
  console.warn('Gemini is not configured yet. Set GEMINI_API_KEYS.');
}

const MODEL_NAMES = (process.env.GEMINI_MODELS || 'gemini-3.5-flash-lite,gemini-3.5-flash')
  .split(',').map(x => x.trim()).filter(Boolean);

const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 40000); 

const GEMINI_FAILURE_COOLDOWN_MS = Number(process.env.GEMINI_FAILURE_COOLDOWN_MS || 30000);
const geminiCooldownUntil = new Map();

function geminiTargetKey(modelIndex, keyIndex) {
  return `${modelIndex}:${keyIndex}`;
}
function isGeminiTargetCoolingDown(modelIndex, keyIndex) {
  return (geminiCooldownUntil.get(geminiTargetKey(modelIndex, keyIndex)) || 0) > Date.now();
}
function markGeminiTargetFailure(modelIndex, keyIndex, error) {
  if ([404, 503, 429, 500, 408].includes(error?.status) || error?.message?.includes('429')) {
    geminiCooldownUntil.set(
      geminiTargetKey(modelIndex, keyIndex),
      Date.now() + GEMINI_FAILURE_COOLDOWN_MS
    );
  }
}

const CONTENT_FILTER_INSTRUCTION = `כלל ברזל: אסור לספק או לעודד תוכן מיני, אלים, סמים, או הימורים. במקרה כזה החזר בדיוק: "היי עצור הקו מסונן ולא ניתן לדבר איתו על תוכן שאינו מתאים לערכי הצניעות"`;

const conversationLog = [];
const activeCalls = new Map();
const remindersList = [];
const projectsList = []; 
const systemLogs = [];

const appSettings = {
  firstCallMessage: process.env.FIRST_CALL_MESSAGE || 'שלום איך אפשר לעזור לך היום אמור בבקשה על מה תרצה לדבר אחרי הצפצוף ולסיום ההקלטה הקש סולמית',
  systemInstruction: process.env.AI_SYSTEM_INSTRUCTION || ''
};

const MAX_CONVERSATION_LOG = 1000;

const SUPABASE_URL = (process.env.SUPABASE_URL || '').replace(/\/$/, '');
const SUPABASE_KEY = (process.env.SUPABASE_KEY || '').trim();
const SUPABASE_ENABLED = !!(SUPABASE_URL && SUPABASE_KEY);

function addSystemLog(message, type = 'info') {
  const time = new Date().toLocaleTimeString('he-IL', { timeZone: 'Asia/Jerusalem' });
  systemLogs.unshift({ id: Date.now().toString(), time, message, type });
  if (systemLogs.length > 100) systemLogs.pop();
}

async function supabaseRequest(path, options = {}) {
  if (!SUPABASE_ENABLED) return null;
  const response = await fetch(SUPABASE_URL + path, {
    ...options,
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: 'Bearer ' + SUPABASE_KEY,
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });
  if (!response.ok) throw new Error('Supabase HTTP ' + response.status + ': ' + await response.text());
  return response;
}

function normalizePhone(value) {
  const phone = String(value || '').trim();
  return phone || 'לא מזוהה';
}
function getCallerNumber(call) {
  return normalizePhone(call?.values?.ApiPhone ?? call?.req?.query?.ApiPhone ??
    call?.req?.body?.ApiPhone ?? call?.query?.ApiPhone);
}

async function loadConversationLog() {
  if (!SUPABASE_ENABLED) return;
  try {
    const r = await supabaseRequest(
      '/rest/v1/conversations?select=id,created_at,phone,call_id,user_text,gemini_text&call_id=not.like.%5F%5Freminder%5F%5F%3A*&order=created_at.desc&limit=' + MAX_CONVERSATION_LOG
    );
    const rows = await r.json();
    const filteredRows = rows.filter(row => !(row.call_id && row.call_id.startsWith('__project__:')));
    conversationLog.splice(0, conversationLog.length, ...filteredRows.reverse().map(row => ({
      id: String(row.id), time: row.created_at, phone: normalizePhone(row.phone),
      callId: String(row.call_id || ''), user: row.user_text || '', gemini: row.gemini_text || ''
    })));
    addSystemLog('היסטוריית שיחות נטענה בהצלחה', 'success');
  } catch (e) {
    addSystemLog('שגיאה בטעינת היסטוריה: ' + e.message, 'error');
  }
}

async function loadRemindersFromSupabase() {
  if (!SUPABASE_ENABLED) return;
  try {
    const r = await supabaseRequest('/rest/v1/conversations?select=id,phone,call_id,user_text&call_id=like.__reminder__%3A*&order=created_at.asc&limit=1000');
    const rows = await r.json();
    remindersList.splice(0, remindersList.length);
    for (const row of rows) {
      try {
        const data = JSON.parse(row.user_text || '{}');
        if (!data.id) continue;
        remindersList.push({ ...data, _supabaseId: row.id });
      } catch (e) {}
    }
  } catch (e) {}
}

async function loadProjectsFromSupabase() {
  if (!SUPABASE_ENABLED) return;
  try {
    const r = await supabaseRequest('/rest/v1/conversations?select=id,phone,created_at,call_id,user_text&call_id=like.__project__%3A*&order=created_at.desc&limit=100');
    const rows = await r.json();
    projectsList.splice(0, projectsList.length);
    for (const row of rows) {
      try {
        const data = JSON.parse(row.user_text || '{}');
        if (!data.id) continue;
        projectsList.push({
          id: data.id, phone: row.phone, title: data.title || 'ללא שם',
          content: data.content || '', time: row.created_at, _supabaseId: row.id
        });
      } catch (e) { }
    }
    addSystemLog(`נטענו ${projectsList.length} פרויקטים`, 'success');
  } catch (e) {
    addSystemLog('שגיאה בטעינת פרויקטים: ' + e.message, 'error');
  }
}

async function persistReminder(reminder) {
  if (!SUPABASE_ENABLED) return;
  const payload = JSON.stringify({
    id: reminder.id, phone: reminder.phone, time: reminder.time, text: reminder.text,
    type: reminder.type, status: reminder.status, triggered: reminder.triggered, consumed: reminder.consumed,
    lastTriggeredDate: reminder.lastTriggeredDate || null
  });
  try {
    if (reminder._supabaseId) {
      await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(reminder._supabaseId), {
        method: 'PATCH', headers: { Prefer: 'return=minimal' },
        body: JSON.stringify({ phone: reminder.phone, call_id: '__reminder__:' + reminder.id, user_text: payload, gemini_text: '' })
      });
    } else {
      const r = await supabaseRequest('/rest/v1/conversations', {
        method: 'POST', headers: { Prefer: 'return=representation' },
        body: JSON.stringify({ phone: reminder.phone, call_id: '__reminder__:' + reminder.id, user_text: payload, gemini_text: '' })
      });
      const created = await r.json();
      if (Array.isArray(created) && created[0]?.id != null) reminder._supabaseId = created[0].id;
    }
  } catch (e) {}
}

async function deletePersistedReminder(reminder) {
  if (!SUPABASE_ENABLED || !reminder?._supabaseId) return;
  try {
    await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(reminder._supabaseId), { method: 'DELETE' });
  } catch (e) {}
}

async function persistProject(project) {
  if (!SUPABASE_ENABLED) return;
  const payload = JSON.stringify({ id: project.id, title: project.title, content: project.content });
  try {
    const r = await supabaseRequest('/rest/v1/conversations', {
      method: 'POST', headers: { Prefer: 'return=representation' },
      body: JSON.stringify({ phone: project.phone, call_id: '__project__:' + project.id, user_text: payload, gemini_text: 'PROJECT_SAVED' })
    });
    const created = await r.json();
    if (Array.isArray(created) && created[0]?.id != null) project._supabaseId = created[0].id;
  } catch (e) {
    addSystemLog('שגיאה בשמירת הפרויקט: ' + e.message, 'error');
  }
}

async function deletePersistedProject(project) {
  if (!SUPABASE_ENABLED || !project?._supabaseId) return;
  try {
    await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(project._supabaseId), { method: 'DELETE' });
  } catch (e) {}
}

async function persistConversationEntry(entry) {
  if (!SUPABASE_ENABLED) return;
  try {
    await supabaseRequest('/rest/v1/conversations', {
      method: 'POST', headers: { Prefer: 'return=minimal' },
      body: JSON.stringify({ phone: entry.phone, call_id: entry.callId || null, user_text: entry.user, gemini_text: entry.gemini })
    });
  } catch (e) {}
}

async function addConversationEntry({phone, callId, userText, geminiText}) {
  const entry = {
    id: Date.now() + '-' + conversationLog.length, time: new Date().toISOString(),
    phone: normalizePhone(phone), callId: String(callId || ''), user: userText || '', gemini: geminiText || ''
  };
  conversationLog.push(entry);
  if (conversationLog.length > MAX_CONVERSATION_LOG) conversationLog.splice(0, conversationLog.length - MAX_CONVERSATION_LOG);
  void persistConversationEntry(entry);
}

function sanitizeForYemot(text) {
  if (!text) return '';
  return String(text).replace(/[."“”‘’']/g, ' ').replace(/[-–—]/g, ' ').replace(/\s+/g, ' ').trim();
}

function extractJsonSafely(raw) {
  try {
    let clean = raw.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    return JSON.parse(clean);
  } catch {
    return null;
  }
}

function withTimeout(promise, ms, label) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      const e = new Error(`Timeout after ${ms}ms: ${label}`);
      e.status = 408; e.isTimeout = true; reject(e);
    }, ms);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutId));
}

function logDetailedError(context, err) {
  if (err instanceof ExitError || err?.name === 'ExitError') return;
  console.error(`[${context}]`, err?.message || err);
  addSystemLog(`[${context}] ${err?.message || err}`, 'error');
}

const genAIClients = apiKeys.map(key => new GoogleGenerativeAI(key));
const jsonConfig = { thinkingConfig: { thinkingLevel: 'minimal' }, responseMimeType: 'application/json' };
const modelsJson = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: jsonConfig})));

const textConfig = { thinkingConfig: { thinkingLevel: 'minimal' } };
const modelsText = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: textConfig})));

const modelsWeb = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, tools:[{googleSearch:{}}], generationConfig: textConfig})));

function getExclusiveInstruction() {
  return [CONTENT_FILTER_INSTRUCTION, appSettings.systemInstruction].filter(Boolean).join('\n');
}

async function generateWithRetry(contents, mode = 'json') {
  if (!genAIClients.length || !modelsJson.length || !modelsJson[0]?.length)
    throw Object.assign(new Error('Gemini is not configured'), {status:400});

  let groups = modelsJson;
  if (mode === 'text') groups = modelsText;
  if (mode === 'web') groups = modelsWeb;

  const deadline = Date.now() + REQUEST_TIMEOUT_MS;
  let lastError;

  for (let mi = 0; mi < groups.length; mi++) {
    for (let ki = 0; ki < genAIClients.length; ki++) {
      if (isGeminiTargetCoolingDown(mi, ki)) continue;
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0) {
        const e = new Error(`Timeout after ${REQUEST_TIMEOUT_MS}ms`); e.status = 408; e.isTimeout = true; throw e;
      }
      try {
        const modelInstance = groups[mi][ki];
        if (!modelInstance) continue;
        return await withTimeout(modelInstance.generateContent(contents), remainingMs, `${MODEL_NAMES[mi]} key #${ki + 1}`);
      } catch(e) {
        lastError = e;
        markGeminiTargetFailure(mi, ki, e);
        if ([404, 503, 429, 500, 408].includes(e.status) || e.message?.includes('429')) continue;
        throw e;
      }
    }
  }
  throw lastError || Object.assign(new Error('Gemini request failed'), {status:502});
}

const yemotApi = new YemotApi(process.env.YEMOT_API_USERNAME, process.env.YEMOT_API_PASSWORD);
const router = YemotRouter({
  printLog: true, defaults: { removeInvalidChars: true },
  uncaughtErrorHandler: e => logDetailedError('call handler', e)
});

function audioParts(audioBase64) {
  return [{inlineData:{mimeType:process.env.YEMOT_AUDIO_MIME_TYPE || 'audio/wav', data:audioBase64}}];
}

async function processAudioTurn(audioBase64, callerPhone) {
  const history = conversationLog.filter(x=>x.phone===callerPhone).slice(-3).map(x=>`Q:${x.user}\nA:${x.gemini}`).join('\n');
  
  const prompt = `${getExclusiveInstruction()}
הקשר קודם:
${history}

הקלטה חדשה. החזר אך ורק JSON תקני:
{"transcript":"תמלול מדויק","answer":"תשובה קצרה וקולעת בעברית להקראה. אם צריך לבנות או לחפש אמור: 'מכין את זה, מיד'","needsWebSearch":false,"wantsProject":false}
wantsProject=true לבקשת קוד/אתר/מאמר. needsWebSearch=true למידע עדכני שאינך יודע בוודאות.`;

  const result = await generateWithRetry([...audioParts(audioBase64), {text:prompt}], 'json');
  const parsed = extractJsonSafely(result.response.text());
  
  if (!parsed) throw Object.assign(new Error('Gemini returned invalid JSON'), {status:502});
  return {
    transcript: sanitizeForYemot(parsed.transcript || ''),
    answer: String(parsed.answer || '').trim(),
    needsWebSearch: parsed.needsWebSearch === true,
    wantsProject: parsed.wantsProject === true
  };
}

async function callHandler(call) {
  const callerPhone = getCallerNumber(call);
  const callId = call?.callId || call?.values?.ApiCallId || '';
  const activeKey = String(callId || (Date.now()+'-'+callerPhone));
  
  let pendingReminderText = null;
  const remIndex = remindersList.findIndex(r => r.phone === callerPhone && r.triggered && !r.consumed);
  if (remIndex !== -1) {
    pendingReminderText = remindersList[remIndex].text;
    remindersList[remIndex].consumed = true;
  }

  const activeCallObj = {
    id: activeKey, phone: callerPhone, callId: String(callId||''),
    startedAt: new Date().toISOString(), status: 'ממתין להקלטה',
    killRequested: false
  };
  activeCalls.set(activeKey, activeCallObj);
  addSystemLog(`שיחה חדשה התחילה מהמספר: ${callerPhone}`, 'info');

  let firstTurn = true;
  let openingPrompt = pendingReminderText ? 
    sanitizeForYemot(`שלום, זוהי תזכורת עבורך: ${pendingReminderText}. על מה תרצה לדבר כעת לאחר הצפצוף?`) :
    (conversationLog.some(x=>x.phone===callerPhone) ? 'שלום שוב שמח לשמוע ממך על מה תרצה לדבר עכשיו' : appSettings.firstCallMessage);

  try {
    while(true) {
      if (activeCallObj.killRequested) {
        try { await call.id_list_message([{type: 'text', data: 'השיחה נותקה על ידי מנהל המערכת להתראות'}]); } catch(_){}
        break;
      }

      const prompt = firstTurn ? openingPrompt : 'אמור שאלה נוספת ולסיום הקש סולמית';
      firstTurn=false;

      let recordPath;
      try {
        recordPath=await call.read([{type:'text',data:prompt}],'record', {min_length:1,max_length:60,no_confirm_menu:true});
      } catch(readErr) {
        if (readErr instanceof ExitError || readErr?.name === 'ExitError') break;
        throw readErr;
      }

      if (activeCallObj.killRequested) break;
      if(!recordPath || recordPath==='None') {
        try { await call.id_list_message([{type: 'text', data: 'לא נקלט דבר להתראות'}]); } catch(_){}
        break;
      }

      activeCallObj.status = 'הקלטה התקבלה — מפענח פקודה';
      let audioBuffer;
      try {
        const response=await withTimeout(yemotApi.download_file('ivr2:'+recordPath), REQUEST_TIMEOUT_MS,'yemotApi.download_file');
        audioBuffer=response.data;
      } catch(e) {
        try { await call.id_list_message([{type:'text',data:'אירעה שגיאה בהורדת ההקלטה נסה שוב'}], {prependToNextAction:true}); } catch(_){}
        continue;
      }

      const audioBase64=Buffer.isBuffer(audioBuffer)?audioBuffer.toString('base64'):Buffer.from(audioBuffer).toString('base64');
      let replyText, transcript='';
      
      try {
        const turnResult = await processAudioTurn(audioBase64, callerPhone);
        transcript = turnResult.transcript || 'לא ניתן היה לתמלל';
        replyText = turnResult.answer;

        if (turnResult.wantsProject) {
            activeCallObj.status = 'בונה פרויקט/מאמר...';
            // ההנחיה עודכנה כדי להבדיל בין טקסט (שיוקרא) לבין קוד (שלא יוקרא)
            const projectPrompt = `${getExclusiveInstruction()}
המשתמש מבקש: "${transcript}"

הנחיות קריטיות:
1. אם המשתמש ביקש לכתוב מאמר, סיפור או טקסט: עליך לכתוב את כל המאמר המלא בתוך תגית [ANSWER] כדי שיוקרא לו באוזן! ושים את אותו מאמר גם בתגית [CONTENT] כדי שיישמר במערכת.
2. אם המשתמש ביקש לבנות קוד או אתר: ייצר קוד מודרני ועשיר עם Tailwind CSS בתוך [CONTENT], ובתוך [ANSWER] כתוב רק משפט אחד קצר (למשל "האתר מוכן וממתין לך במערכת").

חובה להחזיר תבנית זו בדיוק:
[ANSWER] הטקסט להקראה באוזן [/ANSWER]
[TITLE] כותרת עד 4 מילים [/TITLE]
[CONTENT] הקוד המלא או התוכן לשמירה בדאשבורד [/CONTENT]`;

            const projRes = await generateWithRetry([{text: projectPrompt}], 'text');
            const rawOutput = projRes.response.text();
            
            const ansMatch = rawOutput.match(/\[ANSWER\]([\s\S]*?)\[\/ANSWER\]/i);
            const titleMatch = rawOutput.match(/\[TITLE\]([\s\S]*?)\[\/TITLE\]/i);
            const contentMatch = rawOutput.match(/\[CONTENT\]([\s\S]*?)\[\/CONTENT\]/i);

            if (ansMatch) replyText = sanitizeForYemot(ansMatch[1]);
            else replyText = "הפרויקט מוכן וממתין בדאשבורד.";

            if (titleMatch && contentMatch) {
                let cleanContent = contentMatch[1].trim().replace(/^```[a-z]*\n?/i, '').replace(/\n?