import express from 'express';
import { YemotRouter, ExitError } from 'yemot-router2';
import { GoogleGenerativeAI } from '@google/generative-ai';
import YemotApi from 'yemot-api';
import fs from 'fs';
import path from 'path';
import nodemailer from 'nodemailer';

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

const apiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',').map(k => k.trim()).filter(Boolean);

if (!apiKeys.length) {
  console.warn('Gemini is not configured yet. Set GEMINI_API_KEYS.');
}

// התיקון לשגיאת החיבור ב-Render (IPv6 ENETUNREACH)
const transporter = nodemailer.createTransport({
  host: 'smtp.gmail.com',
  port: 465,
  secure: true,
  auth: { user: process.env.EMAIL_USER || '', pass: process.env.EMAIL_PASS || '' }
});

const MODEL_NAMES = (process.env.GEMINI_MODELS || 'gemini-3.5-flash-lite,gemini-3.5-flash')
  .split(',').map(x => x.trim()).filter(Boolean);

const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 40000); 
const GEMINI_FAILURE_COOLDOWN_MS = Number(process.env.GEMINI_FAILURE_COOLDOWN_MS || 30000);
const geminiCooldownUntil = new Map();

function geminiTargetKey(modelIndex, keyIndex) { return `${modelIndex}:${keyIndex}`; }
function isGeminiTargetCoolingDown(modelIndex, keyIndex) { return (geminiCooldownUntil.get(geminiTargetKey(modelIndex, keyIndex)) || 0) > Date.now(); }
function markGeminiTargetFailure(modelIndex, keyIndex, error) {
  if ([404, 503, 429, 500, 408].includes(error?.status) || error?.message?.includes('429')) {
    geminiCooldownUntil.set(geminiTargetKey(modelIndex, keyIndex), Date.now() + GEMINI_FAILURE_COOLDOWN_MS);
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
    ...options, headers: { apikey: SUPABASE_KEY, Authorization: 'Bearer ' + SUPABASE_KEY, 'Content-Type': 'application/json', ...(options.headers || {}) }
  });
  if (!response.ok) throw new Error('Supabase HTTP ' + response.status + ': ' + await response.text());
  return response;
}

function normalizePhone(value) { return String(value || '').trim() || 'לא מזוהה'; }
function getCallerNumber(call) {
  return normalizePhone(call?.values?.ApiPhone ?? call?.req?.query?.ApiPhone ?? call?.req?.body?.ApiPhone ?? call?.query?.ApiPhone);
}

function getCompressedHistory(phone, limit = 3) {
  return conversationLog
    .filter(x => x.phone === phone)
    .slice(-limit)
    .map(x => {
      let aiResponse = x.gemini || '';
      if (aiResponse.length > 200) aiResponse = aiResponse.substring(0, 200) + '... [התוכן קוצר כדי לחסוך באסימונים]';
      return `Q:${x.user}\nA:${aiResponse}`;
    }).join('\n');
}

async function loadConversationLog() {
  if (!SUPABASE_ENABLED) return;
  try {
    const r = await supabaseRequest('/rest/v1/conversations?select=id,created_at,phone,call_id,user_text,gemini_text&call_id=not.like.%5F%5Freminder%5F%5F%3A*&order=created_at.desc&limit=' + MAX_CONVERSATION_LOG);
    const rows = await r.json();
    const filteredRows = rows.filter(row => !(row.call_id && row.call_id.startsWith('__project__:')));
    conversationLog.splice(0, conversationLog.length, ...filteredRows.reverse().map(row => ({
      id: String(row.id), time: row.created_at, phone: normalizePhone(row.phone),
      callId: String(row.call_id || ''), user: row.user_text || '', gemini: row.gemini_text || ''
    })));
  } catch (e) {}
}

async function loadRemindersFromSupabase() {
  if (!SUPABASE_ENABLED) return;
  try {
    const r = await supabaseRequest('/rest/v1/conversations?select=id,phone,call_id,user_text&call_id=like.__reminder__%3A*&order=created_at.asc&limit=1000');
    const rows = await r.json();
    remindersList.splice(0, remindersList.length);
    for (const row of rows) {
      try { const data = JSON.parse(row.user_text || '{}'); if (data.id) remindersList.push({ ...data, _supabaseId: row.id }); } catch (e) {}
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
      try { const data = JSON.parse(row.user_text || '{}'); if (data.id) projectsList.push({ id: data.id, phone: row.phone, title: data.title || 'ללא שם', content: data.content || '', time: row.created_at, _supabaseId: row.id }); } catch (e) {}
    }
  } catch (e) {}
}

async function persistReminder(reminder) {
  if (!SUPABASE_ENABLED) return;
  const payload = JSON.stringify({ id: reminder.id, phone: reminder.phone, time: reminder.time, text: reminder.text, type: reminder.type, status: reminder.status, triggered: reminder.triggered, consumed: reminder.consumed, lastTriggeredDate: reminder.lastTriggeredDate || null });
  try {
    if (reminder._supabaseId) {
      await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(reminder._supabaseId), { method: 'PATCH', headers: { Prefer: 'return=minimal' }, body: JSON.stringify({ phone: reminder.phone, call_id: '__reminder__:' + reminder.id, user_text: payload, gemini_text: '' }) });
    } else {
      const r = await supabaseRequest('/rest/v1/conversations', { method: 'POST', headers: { Prefer: 'return=representation' }, body: JSON.stringify({ phone: reminder.phone, call_id: '__reminder__:' + reminder.id, user_text: payload, gemini_text: '' }) });
      const created = await r.json(); if (Array.isArray(created) && created[0]?.id != null) reminder._supabaseId = created[0].id;
    }
  } catch (e) {}
}

async function deletePersistedReminder(reminder) {
  if (!SUPABASE_ENABLED || !reminder?._supabaseId) return;
  try { await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(reminder._supabaseId), { method: 'DELETE' }); } catch (e) {}
}

async function persistProject(project) {
  if (!SUPABASE_ENABLED) return;
  const payload = JSON.stringify({ id: project.id, title: project.title, content: project.content });
  try {
    const r = await supabaseRequest('/rest/v1/conversations', { method: 'POST', headers: { Prefer: 'return=representation' }, body: JSON.stringify({ phone: project.phone, call_id: '__project__:' + project.id, user_text: payload, gemini_text: 'PROJECT_SAVED' }) });
    const created = await r.json(); if (Array.isArray(created) && created[0]?.id != null) project._supabaseId = created[0].id;
  } catch (e) {}
}

async function deletePersistedProject(project) {
  if (!SUPABASE_ENABLED || !project?._supabaseId) return;
  try { await supabaseRequest('/rest/v1/conversations?id=eq.' + encodeURIComponent(project._supabaseId), { method: 'DELETE' }); } catch (e) {}
}

async function persistConversationEntry(entry) {
  if (!SUPABASE_ENABLED) return;
  try { await supabaseRequest('/rest/v1/conversations', { method: 'POST', headers: { Prefer: 'return=minimal' }, body: JSON.stringify({ phone: entry.phone, call_id: entry.callId || null, user_text: entry.user, gemini_text: entry.gemini }) }); } catch (e) {}
}

async function addConversationEntry({phone, callId, userText, geminiText}) {
  const entry = { id: Date.now() + '-' + conversationLog.length, time: new Date().toISOString(), phone: normalizePhone(phone), callId: String(callId || ''), user: userText || '', gemini: geminiText || '' };
  conversationLog.push(entry);
  if (conversationLog.length > MAX_CONVERSATION_LOG) conversationLog.splice(0, conversationLog.length - MAX_CONVERSATION_LOG);
  void persistConversationEntry(entry);
}

function sanitizeForYemot(text) { return String(text||'').replace(/[."“”‘’']/g, ' ').replace(/[-–—]/g, ' ').replace(/\s+/g, ' ').trim(); }

function extractJsonSafely(raw) {
  try { return JSON.parse(raw.trim().replace(/^\x60\x60\x60(?:json)?\s*/i, '').replace(/\s*\x60\x60\x60$/i, '').trim()); } catch { return null; }
}

function withTimeout(promise, ms, label) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => { timeoutId = setTimeout(() => { const e = new Error(`Timeout after ${ms}ms: ${label}`); e.status = 408; e.isTimeout = true; reject(e); }, ms); });
  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutId));
}

function logDetailedError(context, err) {
  if (err instanceof ExitError || err?.name === 'ExitError') return;
  addSystemLog(`[${context}] ${err?.message || err}`, 'error');
}

const genAIClients = apiKeys.map(key => new GoogleGenerativeAI(key));
const jsonConfig = { thinkingConfig: { thinkingLevel: 'minimal' }, responseMimeType: 'application/json' };
const modelsJson = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: jsonConfig})));
const textConfig = { thinkingConfig: { thinkingLevel: 'minimal' } };
const modelsText = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: textConfig})));
const modelsWeb = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, tools:[{googleSearch:{}}], generationConfig: textConfig})));

function getExclusiveInstruction() { return [CONTENT_FILTER_INSTRUCTION, appSettings.systemInstruction].filter(Boolean).join('\n'); }

async function generateWithRetry(contents, mode = 'json') {
  if (!genAIClients.length || !modelsJson.length || !modelsJson[0]?.length) throw Object.assign(new Error('Gemini is not configured'), {status:400});
  let groups = modelsJson; if (mode === 'text') groups = modelsText; if (mode === 'web') groups = modelsWeb;
  const deadline = Date.now() + REQUEST_TIMEOUT_MS; let lastError;
  for (let mi = 0; mi < groups.length; mi++) {
    for (let ki = 0; ki < genAIClients.length; ki++) {
      if (isGeminiTargetCoolingDown(mi, ki)) continue;
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0) { const e = new Error(`Timeout after ${REQUEST_TIMEOUT_MS}ms`); e.status = 408; e.isTimeout = true; throw e; }
      try {
        const modelInstance = groups[mi][ki]; if (!modelInstance) continue;
        return await withTimeout(modelInstance.generateContent(contents), remainingMs, `${MODEL_NAMES[mi]} key #${ki + 1}`);
      } catch(e) {
        lastError = e; markGeminiTargetFailure(mi, ki, e);
        if ([404, 503, 429, 500, 408].includes(e.status) || e.message?.includes('429')) continue;
        throw e;
      }
    }
  }
  throw lastError || Object.assign(new Error('Gemini request failed'), {status:502});
}

const yemotApi = new YemotApi(process.env.YEMOT_API_USERNAME, process.env.YEMOT_API_PASSWORD);
const router = YemotRouter({ printLog: true, defaults: { removeInvalidChars: true }, uncaughtErrorHandler: e => logDetailedError('call handler', e) });

function audioParts(audioBase64) { return [{inlineData:{mimeType:process.env.YEMOT_AUDIO_MIME_TYPE || 'audio/wav', data:audioBase64}}]; }

async function processAudioTurn(audioBase64, callerPhone) {
  const history = getCompressedHistory(callerPhone, 3);
  const prompt = `${getExclusiveInstruction()}
הקשר קודם:
${history}

הקלטה חדשה. החזר אך ורק JSON תקני:
{"transcript":"תמלול מדויק","answer":"תשובה קצרה","needsWebSearch":false,"wantsProject":false,"sendEmailTo":null}
wantsProject=true לבקשת קוד/אתר/מאמר. needsWebSearch=true למידע עדכני. 
sendEmailTo="כתובת אימייל" - רק אם המשתמש הזכיר במפורש לאיזו כתובת לשלוח. אחרת השאר null.`;

  const result = await generateWithRetry([...audioParts(audioBase64), {text:prompt}], 'json');
  const parsed = extractJsonSafely(result.response.text());
  if (!parsed) throw Object.assign(new Error('Gemini returned invalid JSON'), {status:502});
  return { transcript: sanitizeForYemot(parsed.transcript || ''), answer: String(parsed.answer || '').trim(), needsWebSearch: parsed.needsWebSearch === true, wantsProject: parsed.wantsProject === true, sendEmailTo: parsed.sendEmailTo || null };
}

async function callHandler(call) {
  const callerPhone = getCallerNumber(call);
  const callId = call?.callId || call?.values?.ApiCallId || '';
  const activeKey = String(callId || (Date.now()+'-'+callerPhone));
  
  let pendingReminderText = null;
  const remIndex = remindersList.findIndex(r => r.phone === callerPhone && r.triggered && !r.consumed);
  if (remIndex !== -1) { pendingReminderText = remindersList[remIndex].text; remindersList[remIndex].consumed = true; }

  const activeCallObj = { id: activeKey, phone: callerPhone, callId: String(callId||''), startedAt: new Date().toISOString(), status: 'ממתין להקלטה', killRequested: false };
  activeCalls.set(activeKey, activeCallObj);
  addSystemLog(`שיחה חדשה התחילה מהמספר: ${callerPhone}`, 'info');

  let firstTurn = true;
  let openingPrompt = pendingReminderText ? sanitizeForYemot(`שלום, זוהי תזכורת עבורך: ${pendingReminderText}. על מה תרצה לדבר כעת לאחר הצפצוף?`) :
    (conversationLog.some(x=>x.phone===callerPhone) ? 'שלום שוב שמח לשמוע ממך על מה תרצה לדבר עכשיו' : appSettings.firstCallMessage);

  try {
    while(true) {
      if (activeCallObj.killRequested) { try { await call.id_list_message([{type: 'text', data: 'השיחה נותקה על ידי מנהל המערכת להתראות'}]); } catch(_){} break; }
      const prompt = firstTurn ? openingPrompt : 'אמור שאלה נוספת ולסיום הקש סולמית';
      firstTurn=false;

      let recordPath;
      try { recordPath=await call.read([{type:'text',data:prompt}],'record', {min_length:1,max_length:60,no_confirm_menu:true}); } 
      catch(readErr) { if (readErr instanceof ExitError || readErr?.name === 'ExitError') break; throw readErr; }

      if (activeCallObj.killRequested) break;
      if(!recordPath || recordPath==='None') { try { await call.id_list_message([{type: 'text', data: 'לא נקלט דבר להתראות'}]); } catch(_){} break; }

      activeCallObj.status = 'הקלטה התקבלה — מפענח פקודה';
      let audioBuffer;
      try { const response=await withTimeout(yemotApi.download_file('ivr2:'+recordPath), REQUEST_TIMEOUT_MS,'yemotApi.download_file'); audioBuffer=response.data; } 
      catch(e) { try { await call.id_list_message([{type:'text',data:'אירעה שגיאה בהורדת ההקלטה נסה שוב'}], {prependToNextAction:true}); } catch(_){} continue; }

      const audioBase64=Buffer.isBuffer(audioBuffer)?audioBuffer.toString('base64'):Buffer.from(audioBuffer).toString('base64');
      let replyText, transcript='';
      
      try {
        const turnResult = await processAudioTurn(audioBase64, callerPhone);
        transcript = turnResult.transcript || 'לא ניתן היה לתמלל';
        replyText = turnResult.answer;

        if (turnResult.wantsProject || turnResult.sendEmailTo) {
            activeCallObj.status = 'בונה פרויקט/מאמר...';
            const projectPrompt = `${getExclusiveInstruction()}
המשתמש מבקש: "${transcript}"
אם המשתמש ביקש לכתוב מאמר או טקסט, כתוב אותו בתוך [ANSWER] כדי שיוקרא, וגם בתוך [CONTENT] לשמירה. אם ביקש קוד, שים קוד (Tailwind CSS) רק ב-[CONTENT].
[ANSWER] הטקסט להקראה באוזן [/ANSWER]
[TITLE] כותרת עד 4 מילים [/TITLE]
[CONTENT] הקוד המלא או התוכן לשמירה [/CONTENT]`;

            const projRes = await generateWithRetry([{text: projectPrompt}], 'text');
            const rawOutput = projRes.response.text();
            
            const ansMatch = rawOutput.match(/\[ANSWER\]([\s\S]*?)\[\/ANSWER\]/i);
            const titleMatch = rawOutput.match(/\[TITLE\]([\s\S]*?)\[\/TITLE\]/i);
            const contentMatch = rawOutput.match(/\[CONTENT\]([\s\S]*?)\[\/CONTENT\]/i);

            if (ansMatch) replyText = sanitizeForYemot(ansMatch[1]);
            else replyText = "הפרויקט מוכן וממתין בדאשבורד.";

            if (titleMatch && contentMatch) {
                let cleanContent = contentMatch[1].trim().replace(/^\x60\x60\x60[a-z]*\n?/i, '').replace(/\n?\x60\x60\x60$/i, '').trim();
                const title = titleMatch[1].trim() || 'פרויקט חדש';
                const newProject = { id: Date.now().toString(), phone: callerPhone, title: title, content: cleanContent, time: new Date().toISOString() };
                projectsList.unshift(newProject);
                void persistProject(newProject);
                addSystemLog(`פרויקט עשיר נוצר עבור ${callerPhone}`, 'success');

                if (turnResult.sendEmailTo) {
                    if (process.env.EMAIL_USER) {
                        try {
                            const mailOptions = {
                                from: process.env.EMAIL_USER,
                                to: turnResult.sendEmailTo,
                                subject: `ימות המשיח AI - ${title}`,
                                html: `<div dir="rtl" style="font-family:sans-serif;"><h2>${title}</h2><hr/><pre style="white-space: pre-wrap; font-family:inherit;">${cleanContent}</pre></div>`
                            };
                            await transporter.sendMail(mailOptions);
                            replyText += " והתוכן נשלח לכתובת המייל שביקשת.";
                            addSystemLog(`נשלח מייל לכתובת ${turnResult.sendEmailTo} בהצלחה!`, 'success');
                        } catch(e) {
                            replyText += " המערכת ניסתה לשלוח אימייל אך נתקלה בתקלה טכנית.";
                            addSystemLog(`שגיאה בשליחת מייל: ${e.message}`, 'error');
                        }
                    } else {
                        replyText += " אך מערכת האימיילים טרם הוגדרה בשרת.";
                        addSystemLog(`בקשת אימייל נדחתה - חסר משתנה סביבה EMAIL_USER`, 'error');
                    }
                }
            }
        } 
        else if (turnResult.needsWebSearch) {
            activeCallObj.status = 'מחפש ברשת...';
            const webPrompt = `${getExclusiveInstruction()} \n שאלה: "${transcript}" \n חפש ברשת מידע עדכני ומדויק. החזר תשובה מפורטת להקראה טלפונית ללא קישורים.`;
            const webRes = await generateWithRetry([{text: webPrompt}], 'web');
            replyText = sanitizeForYemot(webRes.response.text());
        }

      } catch(e) {
        logDetailedError('Gemini processing',e);
        replyText = (e.status === 429 || e.message?.includes('429')) 
          ? 'מצטערים, הגענו למכסת הפניות היומית מגוגל. נסה שוב מאוחר יותר.'
          : (e.status === 503 ? 'אני קצת עמוס כרגע נסה שוב' : 'תקלה בעיבוד אפשר לנסות שוב');
      }

      replyText=sanitizeForYemot(replyText)||'מצטער לא הצלחתי לנסח תשובה';
      await addConversationEntry({phone:callerPhone,callId,userText:transcript,geminiText:replyText});

      if (activeCallObj.killRequested) break;
      try { await call.id_list_message([{type:'text',data:replyText}],{prependToNextAction:true}); } catch(e) { if (e instanceof ExitError || e?.name === 'ExitError') break; }
    }
  } catch (err) {
    if (!(err instanceof ExitError || err?.name === 'ExitError')) logDetailedError('fatal call loop error', err);
  } finally { activeCalls.delete(activeKey); addSystemLog(`שיחה הסתיימה עבור מספר: ${callerPhone}`, 'info'); }
}

router.all('/yemot',callHandler);
app.use(router);

// ==== נתיב מיוחד לסימולטור אינטרנטי ====
app.post('/api/simulate', async (req, res) => {
  const { text } = req.body;
  if(!text) return res.status(400).json({error: 'Missing text'});
  const callerPhone = 'Web-Simulator';
  
  try {
    const history = getCompressedHistory(callerPhone, 3);
    const prompt = `${getExclusiveInstruction()}\nהקשר קודם:\n${history}\n\nהודעה טקסטואלית חדשה. החזר אך ורק JSON תקני:\n{"transcript":"${text}","answer":"תשובה קצרה","needsWebSearch":false,"wantsProject":false,"sendEmailTo":null}\nwantsProject=true לבקשת קוד/אתר/מאמר. needsWebSearch=true למידע עדכני. sendEmailTo="כתובת אימייל" - רק אם המשתמש הזכיר במפורש לאיזו כתובת לשלוח. אחרת השאר null.`;
    
    const result = await generateWithRetry([{text:prompt}], 'json');
    const parsed = extractJsonSafely(result.response.text());
    if (!parsed) throw new Error('Invalid JSON');
    
    let replyText = parsed.answer || '';
    let transcript = text;
    
    if (parsed.wantsProject || parsed.sendEmailTo) {
        const projectPrompt = `${getExclusiveInstruction()}\nהמשתמש מבקש: "${transcript}"\nאם המשתמש ביקש לכתוב מאמר או טקסט, כתוב אותו בתוך [ANSWER] כדי שיוקרא, וגם בתוך [CONTENT] לשמירה. אם ביקש קוד, שים קוד רק ב-[CONTENT].\n[ANSWER] הטקסט [/ANSWER]\n[TITLE] כותרת [/TITLE]\n[CONTENT] התוכן [/CONTENT]`;
        const projRes = await generateWithRetry([{text: projectPrompt}], 'text');
        const rawOutput = projRes.response.text();
        
        const ansMatch = rawOutput.match(/\[ANSWER\]([\s\S]*?)\[\/ANSWER\]/i);
        const titleMatch = rawOutput.match(/\[TITLE\]([\s\S]*?)\[\/TITLE\]/i);
        const contentMatch = rawOutput.match(/\[CONTENT\]([\s\S]*?)\[\/CONTENT\]/i);

        if (ansMatch) replyText = ansMatch[1].trim();
        else replyText = "הפרויקט מוכן וממתין בדאשבורד.";

        if (titleMatch && contentMatch) {
            let cleanContent = contentMatch[1].trim().replace(/^\x60\x60\x60[a-z]*\n?/i, '').replace(/\n?\x60\x60\x60$/i, '').trim();
            const title = titleMatch[1].trim() || 'פרויקט חדש';
            const newProject = { id: Date.now().toString(), phone: callerPhone, title: title, content: cleanContent, time: new Date().toISOString() };
            projectsList.unshift(newProject);
            void persistProject(newProject);
            addSystemLog(`פרויקט עשיר נוצר מהסימולטור`, 'success');

            if (parsed.sendEmailTo && process.env.EMAIL_USER) {
                try {
                    await transporter.sendMail({ from: process.env.EMAIL_USER, to: parsed.sendEmailTo, subject: `ימות המשיח AI - ${title}`, html: `<div dir="rtl"><h2>${title}</h2><hr/><pre>${cleanContent}</pre></div>` });
                    replyText += " והתוכן נשלח לכתובת המייל שביקשת.";
                    addSystemLog(`נשלח מייל לכתובת ${parsed.sendEmailTo} בהצלחה!`, 'success');
                } catch(e) {
                    replyText += " אירעה שגיאה טכנית בשליחת האימייל.";
                    addSystemLog(`שגיאה בשליחת מייל: ${e.message}`, 'error');
                }
            } else if (parsed.sendEmailTo) {
                 addSystemLog(`בקשת אימייל נדחתה - חסר משתנה סביבה EMAIL_USER`, 'error');
            }
        }
    } else if (parsed.needsWebSearch) {
        const webRes = await generateWithRetry([{text: `${getExclusiveInstruction()} \n שאלה: "${transcript}" \n חפש ברשת מידע עדכני ומדויק. החזר תשובה מפורטת להקראה.`}], 'web');
        replyText = webRes.response.text().trim();
    }

    replyText = replyText || 'מצטער לא הצלחתי לנסח תשובה';
    await addConversationEntry({phone:callerPhone, callId:'web-sim', userText:transcript, geminiText:replyText});
    
    res.json({ reply: replyText });
  } catch(e) {
    res.status(500).json({ reply: 'שגיאה בעת העיבוד מול השרת או גוגל.' });
  }
});

app.get('/api/conversations',(req,res)=>res.json({
  conversations:conversationLog, activeCalls:Array.from(activeCalls.values()), reminders: remindersList, projects: projectsList,
  systemLogs: systemLogs, settings: appSettings, totalMessages:conversationLog.length, totalCallers:new Set(conversationLog.map(x=>x.phone)).size, serverTime:new Date().toISOString()
}));
app.get('/api/settings', (req, res) => res.json(appSettings));

app.post('/api/reminders', (req, res) => {
  const { phone, time, text, type } = req.body;
  if (!phone || !time || !text) return res.status(400).json({ error: 'Missing data' });
  const reminder = { id: Date.now().toString(), phone: String(phone).trim(), time: String(time).trim(), text: String(text).trim(), type: type || 'שיחה קולית מלאה', status: 'ממתין', triggered: false, consumed: false, lastTriggeredDate: null };
  remindersList.push(reminder); void persistReminder(reminder);
  addSystemLog(`תזכורת חדשה למספר ${phone}`, 'success'); res.json({ ok: true, reminder });
});
app.delete('/api/reminders/:id', (req, res) => {
  const idx = remindersList.findIndex(r => r.id === req.params.id);
  if (idx !== -1) { void deletePersistedReminder(remindersList.splice(idx, 1)[0]); res.json({ ok: true }); } else res.status(404).json({ error: 'Not found' });
});
app.delete('/api/projects/:id', (req, res) => {
  const idx = projectsList.findIndex(p => p.id === req.params.id);
  if (idx !== -1) { void deletePersistedProject(projectsList.splice(idx, 1)[0]); res.json({ ok: true }); } else res.status(404).json({ error: 'Not found' });
});
app.post('/api/calls/:id/kill', (req, res) => {
  const callObj = activeCalls.get(req.params.id);
  if (!callObj) return res.status(404).json({ error: 'Not found' });
  callObj.killRequested = true; callObj.status = 'התבקש ניתוק'; res.json({ ok: true });
});
app.post('/api/settings', (req, res) => {
  const { firstCallMessage, systemInstruction } = req.body;
  if (typeof firstCallMessage === 'string') appSettings.firstCallMessage = firstCallMessage;
  if (typeof systemInstruction === 'string') appSettings.systemInstruction = systemInstruction;
  addSystemLog('הגדרות המערכת עודכנו', 'success'); res.json({ ok: true, settings: appSettings });
});

app.get('/health',(req,res)=>res.json({ok:true}));
app.get('/',(req,res)=>res.type('html').send(fs.readFileSync(path.resolve('dashboard.html'), 'utf8')));

let reminderCheckRunning = false; let lastReminderCheckMs = Date.now();
setInterval(async () => {
  if (reminderCheckRunning) return;
  reminderCheckRunning = true;
  try {
    const nowIsrael = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Jerusalem" }));
    const nowMs = nowIsrael.getTime();
    const currentDateStr = [nowIsrael.getFullYear(), String(nowIsrael.getMonth() + 1).padStart(2, '0'), String(nowIsrael.getDate()).padStart(2, '0')].join('-');
    for (const r of remindersList) {
      if (r.triggered && r.status !== 'שגיאה') continue;
      let isTimeToRun = false;
      if (r.time.includes('T')) {
        const targetDate = new Date(r.time);
        isTimeToRun = Number.isFinite(targetDate.getTime()) && targetDate.getTime() <= nowMs && targetDate.getTime() > lastReminderCheckMs - 24 * 60 * 60 * 1000;
      } else {
        const match = r.time.match(/^(\d{1,2}):(\d{2})$/);
        if (match) {
          const scheduledMinutes = Number(match[1]) * 60 + Number(match[2]);
          const currentMinutes = nowIsrael.getHours() * 60 + nowIsrael.getMinutes();
          const alreadyTriggeredToday = r.lastTriggeredDate === currentDateStr;
          isTimeToRun = !alreadyTriggeredToday && currentMinutes >= scheduledMinutes;
        }
      }
      if (isTimeToRun) {
        r.triggered = true; r.status = 'מפעיל';
        if (!r.time.includes('T')) r.lastTriggeredDate = currentDateStr;
        void persistReminder(r);
        try {
          const apiKey = process.env.YEMOT_API_KEY || process.env.YEMOT_API_PASSWORD;
          const campId = process.env.REMINDER_KAMPAIN_ID?.trim() || '2';
          const cleanPhone = r.phone.replace(/\D/g, '');
          const url = `https://www.call2all.co.il/ym/api/RunCampaign?token=${encodeURIComponent(apiKey)}&campId=${encodeURIComponent(campId)}&phones=${encodeURIComponent(cleanPhone)}`;
          const result = await (await fetch(url)).json();
          r.status = result.responseStatus === 'OK' ? 'בוצע' : 'שגיאה';
          if(r.status==='שגיאה') r.triggered=false;
          void persistReminder(r);
        } catch (err) { r.status = 'שגיאה'; r.triggered = false; void persistReminder(r); }
      }
    }
    lastReminderCheckMs = nowMs;
  } catch (err) {} finally { reminderCheckRunning = false; }
}, 30000);

async function configureYemotStructure() {
  const apiKey=process.env.YEMOT_API_KEY?.trim(); if(!apiKey) return;
  const base='https://www.call2all.co.il/ym/api';
  async function updateExtension(path,params) { await fetch(`${base}/UpdateExtension?${new URLSearchParams({token:apiKey,path,...params})}`); }
  const publicUrl=(process.env.PUBLIC_BASE_URL||'').replace(/\/$/,''); if(!publicUrl) return;
  try {
      await updateExtension('ivr2:/1',{type:'api',api_link:publicUrl+'/yemot'});
      const voiceMap=(process.env.YEMOT_VOICE_OPTIONS||'1:Elik_2100,2:Jacob,3:ymMale').split(',');
      for(const item of voiceMap){
        const [extension,voice]=item.split(':'); if(!extension||!voice) continue;
        await updateExtension(`ivr2:/2/${extension}`,{ type:'add_id_to_list',add_id_to_list_location_list:'/ivr', add_id_to_list_key:'voice',add_id_to_list_value:voice, add_id_to_list_value_change:'yes',add_id_to_list_end_goto:'/1', add_id_to_list_error_end_goto:'/2' });
      }
  } catch(e) {}
}

process.on('unhandledRejection',(reason)=>{if(!(reason instanceof ExitError)) logDetailedError('Unhandled Rejection',reason)});
process.on('uncaughtException',(err)=>{if(!(err instanceof ExitError)) logDetailedError('Uncaught Exception',err)});
const port=process.env.PORT||3000;
app.listen(port,async()=>{
  console.log('server running on port '+port); addSystemLog('השרת עלה בהצלחה על פורט ' + port, 'success');
  await loadConversationLog(); await loadRemindersFromSupabase(); await loadProjectsFromSupabase(); await configureYemotStructure();
});