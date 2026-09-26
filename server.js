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

// זמן המתנה הוארך כדי לאפשר למודל לכתוב אתרים ענקיים
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

const CONTENT_FILTER_INSTRUCTION = `כלל סינון תוכן מחייב: אין לספק, לעודד או לפרט תוכן שאינו תואם ערכי צניעות וחינוך.

פירוט אסור, אין לענות על התוכן האסור ויש להחזיר בדיוק את הודעת הסינון הבאה:
"היי עצור הקו מסונן ולא ניתן לדבר איתו על תוכן שאינו מתאים לערכי הצניעות והחינוך"`;

const conversationLog = [];
const activeCalls = new Map();
const remindersList = [];
const projectsList = []; 
const systemLogs = [];

const appSettings = {
  firstCallMessage: process.env.FIRST_CALL_MESSAGE || 'שלום איך אפשר לעזור לך היום ? ולסיום ההקלטה הקש סולמית',
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


// ============== פיצול המודלים לפתרון בעיות ה-JSON והחיפוש ==============
const genAIClients = apiKeys.map(key => new GoogleGenerativeAI(key));

// 1. הגדרת מודל לפיענוח (חייב להחזיר JSON מסודר)
const jsonConfig = { thinkingConfig: { thinkingLevel: 'minimal' }, responseMimeType: 'application/json' };
const modelsJson = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: jsonConfig})));

// 2. הגדרת מודל ליצירת קוד מלא וטקסט ארוך (חופשי, ללא כבלי JSON)
const textConfig = { thinkingConfig: { thinkingLevel: 'minimal' } };
const modelsText = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, generationConfig: textConfig})));

// 3. הגדרת מודל ייעודי לחיפוש באינטרנט
const modelsWeb = MODEL_NAMES.map(name => genAIClients.map(ai => ai.getGenerativeModel({model:name, tools:[{googleSearch:{}}], generationConfig: textConfig})));


function getExclusiveInstruction() {
  return [CONTENT_FILTER_INSTRUCTION, appSettings.systemInstruction].filter(Boolean).join('\n\n');
}

// פונקציית העבודה החכמה - בוחרת את המודל הנכון לפי המשימה
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

// שלב 1: האזנה ופענוח ה-Intent של המתקשר (באמצעות JSON טהור ומהיר)
async function processAudioTurn(audioBase64) {
  const prompt = `${getExclusiveInstruction()}
זו הקלטה של שאלה או בקשה מהמתקשר. עבד את ההקלטה פעם אחת והחזר JSON בלבד במבנה הבא:
{
  "transcript": "תמלול מדויק בעברית של מה שהמתקשר אמר",
  "answer": "תשובה קצרה וברורה להקראה קולית. אם המשתמש ביקש לבנות משהו או לחפש משהו באינטרנט, אל תענה על השאלה כאן אלא פשוט תגיד משהו כמו: 'כמה שניות, אני מכין את זה...'",
  "needsWebSearch": false,
  "wantsProject": false
}

כללים:
1. needsWebSearch: true אם חובה לחפש באינטרנט (חדשות, נתונים עדכניים, או בקשה מפורשת לחיפוש רשת).
2. wantsProject: true אם המשתמש מבקש לבנות קוד, אתר, מערכת, או לנסח טקסט/מאמר ארוך במיוחד שצריך להישמר.
- אל תכניס JSON בתוך markdown.`;

  // משתמשים במודל ה-JSON לזיהוי מהיר
  const result = await generateWithRetry([...audioParts(audioBase64), {text:prompt}], 'json');
  const raw = result.response.text().trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
  try {
    const parsed = JSON.parse(raw);
    return {
      transcript: sanitizeForYemot(parsed.transcript || ''),
      answer: String(parsed.answer || '').trim(),
      needsWebSearch: parsed.needsWebSearch === true,
      wantsProject: parsed.wantsProject === true
    };
  } catch {
    throw Object.assign(new Error('Gemini returned invalid turn JSON'), {status:502});
  }
}

async function buildOpeningForCaller(phone, reminderText = null) {
  if (reminderText) return sanitizeForYemot(`שלום, זוהי תזכורת עבורך: ${reminderText}. על מה תרצה לדבר כעת לאחר הצפצוף?`);
  const previous = conversationLog.filter(x=>x.phone===normalizePhone(phone)).slice(-8);
  if (!previous.length) return appSettings.firstCallMessage;
  const history = previous.map(x=>'המתקשר: '+x.user+'\nAI: '+x.gemini).join('\n\n');
  try {
    // השתמשתי במודל טקסט רגיל כדי לקבל פתיח זורם
    const r = await generateWithRetry([{text:`${getExclusiveInstruction()}
הנה קטעים משיחות קודמות:
${history}
צור פתיח קצר בעברית שמזכיר בקצרה את הנושא האחרון ושואל על מה המתקשר רוצה לדבר עכשיו. בלי נקודות ובלי מרכאות.`}], 'text');
    return sanitizeForYemot(r.response.text()) || 'שלום שוב שמח לשמוע ממך על מה תרצה לדבר עכשיו';
  } catch { return 'שלום שוב שמח לשמוע ממך על מה תרצה לדבר עכשיו'; }
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

  let firstTurn=true;
  let openingPrompt=null;
  if (pendingReminderText) {
    openingPrompt = await buildOpeningForCaller(callerPhone, pendingReminderText);
  } else if (conversationLog.some(x=>x.phone===callerPhone)) {
    openingPrompt=await buildOpeningForCaller(callerPhone);
  }

  try {
    while(true) {
      if (activeCallObj.killRequested) {
        try { await call.id_list_message([{type: 'text', data: 'השיחה נותקה על ידי מנהל המערכת להתראות'}]); } catch(_){}
        break;
      }

      const prompt = firstTurn ? (openingPrompt || appSettings.firstCallMessage) : 'אמור שאלה נוספת ולסיום הקש סולמית';
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
        // שלב 1: זיהוי הצרכים
        const turnResult = await processAudioTurn(audioBase64);
        transcript = turnResult.transcript || 'לא ניתן היה לתמלל';
        replyText = turnResult.answer;

        // שלב 2 (אופציונלי): הפעלת סוכני ביצוע כבדים לפי הצורך (טקסט חופשי / רשת)
        if (turnResult.wantsProject) {
            activeCallObj.status = 'בונה פרויקט וקוד (פרימיום)...';
            const projectPrompt = `${getExclusiveInstruction()}
המשתמש ביקש לבנות את הפרויקט / האתר / התוכן הבא: "${transcript}"

הוראות ייצור קפדניות:
1. עליך לייצר קוד מודרני, ארוך, עשיר, מפורט ומקצועי לחלוטין ברמת Production! בשום אופן אל תייצר רק "שלד" או תבנית בסיסית.
2. אם התבקשת לבנות אתר - חובה להשתמש ב-Tailwind CSS דרך CDN (<script src="https://cdn.tailwindcss.com"></script>), לעצב בצורה מרהיבה ורספונסיבית, לכלול אלמנטים מציאותיים (כפתורים, כרטיסיות, פוטר, אנימציות CSS) ופונטים יפים (למשל Heebo מ-Google Fonts).

מבנה התשובה שלך חייב להיות אך ורק בפורמט הבא (ללא שום תוספת מסביב):

[ANSWER]
כאן תכתוב משפט אחד בעברית שיוקרא למתקשר באוזן (למשל: "מצוין, סיימתי לבנות את האתר המעוצב שלך והוא נשמר במערכת").
[/ANSWER]

[TITLE]
כותרת קצרה של הפרויקט (עד 5 מילים)
[/TITLE]

[CONTENT]
כאן תכניס את כל הקוד המלא, ברמת פירוט מקסימלית.
[/CONTENT]`;

            // קורא למודל הטקסט (המשוחרר מ-JSON) ליצירת פרויקט ענק
            const projRes = await generateWithRetry([{text: projectPrompt}], 'text');
            const rawOutput = projRes.response.text();
            
            // שליפת המידע מתוך הטקסט
            const ansMatch = rawOutput.match(/\[ANSWER\]([\s\S]*?)\[\/ANSWER\]/i);
            const titleMatch = rawOutput.match(/\[TITLE\]([\s\S]*?)\[\/TITLE\]/i);
            const contentMatch = rawOutput.match(/\[CONTENT\]([\s\S]*?)\[\/CONTENT\]/i);

            if (ansMatch) replyText = ansMatch[1].trim();
            else replyText = "הפרויקט המלא שלך מוכן וממתין במערכת.";

            if (titleMatch && contentMatch) {
                // ניקוי עטיפות ה-Markdown (```html) אם המודל הוסיף אותן בטעות
                let cleanContent = contentMatch[1].trim();
                cleanContent = cleanContent.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/i, '').trim();

                const newProject = {
                    id: Date.now().toString(),
                    phone: callerPhone,
                    title: titleMatch[1].trim() || 'פרויקט חדש',
                    content: cleanContent,
                    time: new Date().toISOString()
                };
                projectsList.unshift(newProject);
                void persistProject(newProject);
                addSystemLog(`פרויקט פרימיום נוצר עבור ${callerPhone}: ${newProject.title}`, 'success');
            }
        } 
        else if (turnResult.needsWebSearch) {
            activeCallObj.status = 'מחפש מידע בזמן אמת באינטרנט...';
            const webPrompt = `${getExclusiveInstruction()}
המתקשר שאל: "${transcript}"

חפש מידע עדכני ומדויק באינטרנט.
לאחר מכן, נסח תשובה קצרה, ברורה וקולעת בעברית שתוקרא למתקשר באוזן. 
אל תכלול קישורים, כתובות אינטרנט או תווים מיוחדים.`;
            
            // קורא למודל הייעודי שמחובר למנוע החיפוש של גוגל
            const webRes = await generateWithRetry([{text: webPrompt}], 'web');
            replyText = webRes.response.text().trim();
        }

      } catch(e) {
        logDetailedError('Gemini processing',e);
        replyText = (e.status === 429 || e.message?.includes('429')) 
          ? 'מצטערים, הגענו למכסת הפניות היומית. אנא נסה שוב מאוחר יותר.'
          : (e.status === 503 ? 'מצטערים אני עמוס כרגע נסה שוב עוד מעט' : 'מצטער הייתה תקלה בעיבוד השאלה אפשר לנסות שוב');
      }

      replyText=sanitizeForYemot(replyText)||'מצטער לא הצלחתי לנסח תשובה נסה שוב';
      await addConversationEntry({phone:callerPhone,callId,userText:transcript,geminiText:replyText});

      if (activeCallObj.killRequested) break;

      try {
        await call.id_list_message([{type:'text',data:replyText}],{prependToNextAction:true});
      } catch(e) {
        if (e instanceof ExitError || e?.name === 'ExitError') break;
      }
    }
  } catch (err) {
    if (!(err instanceof ExitError || err?.name === 'ExitError')) logDetailedError('fatal call loop error', err);
  } finally {
    activeCalls.delete(activeKey);
    addSystemLog(`שיחה הסתיימה עבור מספר: ${callerPhone}`, 'info');
  }
}

router.all('/yemot',callHandler);
app.use(router);

app.get('/api/conversations',(req,res)=>res.json({
  conversations:conversationLog,
  activeCalls:Array.from(activeCalls.values()),
  reminders: remindersList,
  projects: projectsList,
  systemLogs: systemLogs,
  settings: appSettings,
  totalMessages:conversationLog.length,
  totalCallers:new Set(conversationLog.map(x=>x.phone)).size,
  serverTime:new Date().toISOString()
}));

app.get('/api/settings', (req, res) => res.json(appSettings));

app.post('/api/reminders', (req, res) => {
  const { phone, time, text, type } = req.body;
  if (!phone || !time || !text) return res.status(400).json({ error: 'Missing phone, time or text' });
  const reminder = {
    id: Date.now().toString(), phone: String(phone).trim(), time: String(time).trim(), text: String(text).trim(),
    type: type || 'שיחה קולית מלאה', status: 'ממתין', triggered: false, consumed: false, lastTriggeredDate: null
  };
  remindersList.push(reminder);
  void persistReminder(reminder);
  addSystemLog(`תזכורת חדשה נוספה למספר ${phone} לשעה ${time}`, 'success');
  res.json({ ok: true, reminder });
});

app.delete('/api/reminders/:id', (req, res) => {
  const id = req.params.id;
  const idx = remindersList.findIndex(r => r.id === id);
  if (idx !== -1) {
    const removed = remindersList.splice(idx, 1)[0];
    void deletePersistedReminder(removed);
    res.json({ ok: true });
  } else { res.status(404).json({ error: 'Reminder not found' }); }
});

app.delete('/api/projects/:id', (req, res) => {
  const id = req.params.id;
  const idx = projectsList.findIndex(p => p.id === id);
  if (idx !== -1) {
    const removed = projectsList.splice(idx, 1)[0];
    void deletePersistedProject(removed);
    addSystemLog(`פרויקט נמחק: ${removed.title}`, 'info');
    res.json({ ok: true });
  } else { res.status(404).json({ error: 'Project not found' }); }
});

app.post('/api/calls/:id/kill', (req, res) => {
  const callId = req.params.id;
  const callObj = activeCalls.get(callId);
  if (!callObj) return res.status(404).json({ error: 'Call not found' });
  callObj.killRequested = true;
  callObj.status = 'התבקש ניתוק';
  res.json({ ok: true, message: 'Kill signal sent to call' });
});

app.post('/api/settings', (req, res) => {
  const { firstCallMessage, systemInstruction } = req.body;
  if (typeof firstCallMessage === 'string') appSettings.firstCallMessage = firstCallMessage;
  if (typeof systemInstruction === 'string') appSettings.systemInstruction = systemInstruction;
  addSystemLog('הגדרות המערכת עודכנו', 'success');
  res.json({ ok: true, settings: appSettings });
});

app.get('/health',(req,res)=>res.json({ok:true}));
app.get('/',(req,res)=>res.type('html').send(fs.readFileSync(path.resolve('dashboard.html'), 'utf8')));

let reminderCheckRunning = false;
let lastReminderCheckMs = Date.now();

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
          const url = `[https://www.call2all.co.il/ym/api/RunCampaign?token=$](https://www.call2all.co.il/ym/api/RunCampaign?token=$){encodeURIComponent(apiKey)}&campId=${encodeURIComponent(campId)}&phones=${encodeURIComponent(cleanPhone)}`;
          const apiRes = await fetch(url);
          const result = await apiRes.json();
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
  const apiKey=process.env.YEMOT_API_KEY?.trim();
  if(!apiKey) return;
  const base='[https://www.call2all.co.il/ym/api](https://www.call2all.co.il/ym/api)';
  async function updateExtension(path,params) {
    const qs=new URLSearchParams({token:apiKey,path,...params});
    await fetch(`${base}/UpdateExtension?${qs}`);
  }
  const publicUrl=(process.env.PUBLIC_BASE_URL||'').replace(/\/$/,'');
  if(!publicUrl) return;
  try {
      await updateExtension('ivr2:/1',{type:'api',api_link:publicUrl+'/yemot'});
      const voiceMap=(process.env.YEMOT_VOICE_OPTIONS||'1:Elik_2100,2:Jacob,3:ymMale').split(',');
      for(const item of voiceMap){
        const [extension,voice]=item.split(':');
        if(!extension||!voice) continue;
        await updateExtension(`ivr2:/2/${extension}`,{
          type:'add_id_to_list',add_id_to_list_location_list:'/ivr', add_id_to_list_key:'voice',add_id_to_list_value:voice,
          add_id_to_list_value_change:'yes',add_id_to_list_end_goto:'/1', add_id_to_list_error_end_goto:'/2'
        });
      }
  } catch(e) {}
}

process.on('unhandledRejection',(reason)=>{if(!(reason instanceof ExitError)) logDetailedError('Unhandled Rejection',reason)});
process.on('uncaughtException',(err)=>{if(!(err instanceof ExitError)) logDetailedError('Uncaught Exception',err)});
const port=process.env.PORT||3000;
app.listen(port,async()=>{
  console.log('server running on port '+port);
  addSystemLog('השרת עלה בהצלחה על פורט ' + port, 'success');
  await loadConversationLog();
  await loadRemindersFromSupabase();
  await loadProjectsFromSupabase();
  await configureYemotStructure();
});