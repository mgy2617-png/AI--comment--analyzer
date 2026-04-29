let chartInstance = null;
let globalData = [];
let abortController = null;
let currentCounts = null;
let currentWordCloudKeywords = [];
let selectedChartCategory = 'All';

const API_BASE = 'http://127.0.0.1:8000';

const CONFIG = {
    concurrency: 2,
    maxTextLength: 1000,
    requestTimeout: 90000,
    themeColor: '#1e6f3f'
};

const PROMPT_PRESETS = {
    general: '通用情感分析：判断文本整体情绪倾向。不要因为文本很短就强行判断为正向或负向。',
    service: '服务/体验评价：更关注服务态度、响应速度、流程体验、满意度和投诉意图。',
    product: '产品反馈：更关注功能好坏、质量问题、易用性、价格感受和复购/推荐意愿。',
    academic: '访谈/开放题文本：用更中性的研究视角判断态度，不要把描述事实自动当作情绪。'
};

function init() {
    loadExample();
    const slider = document.getElementById('batchSizeSlider');
    const sizeSpan = document.getElementById('batchSizeValue');
    if (slider && sizeSpan) {
        sizeSpan.innerText = slider.value;
        slider.addEventListener('input', () => {
            sizeSpan.innerText = slider.value;
        });
    }
    checkBackendHealth();
}

document.addEventListener('DOMContentLoaded', init);

function loadExample() {
    const preset = document.getElementById('promptPreset');
    if (preset) preset.value = 'general';
    applyPromptPreset();
}

function applyPromptPreset() {
    const preset = document.getElementById('promptPreset');
    const promptBox = document.getElementById('systemPrompt');
    if (!promptBox) return;
    const key = preset?.value || 'general';
    promptBox.value = PROMPT_PRESETS[key] || PROMPT_PRESETS.general;
}

function cleanText(text) {
    return String(text || '').replace(/\s+/g, ' ').trim().slice(0, CONFIG.maxTextLength);
}

function normalizeCategory(cat = 'Neutral') {
    const s = String(cat).trim().toLowerCase();
    if (['positive', '正面', '积极', '好评'].includes(s)) return 'Positive';
    if (['negative', '负面', '消极', '差评'].includes(s)) return 'Negative';
    return 'Neutral';
}

function safeScore(score) {
    const n = Number(score);
    if (!Number.isFinite(n)) return 0.5;
    return Math.max(0, Math.min(1, n));
}

function chunkArray(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
}

function escapeHTML(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function escapeCSV(value) {
    const val = String(value ?? '');
    return `"${val.replace(/"/g, '""')}"`;
}

function showResultSection() {
    const sec = document.getElementById('resultSection');
    sec.classList.remove('hidden');
}

function destroyChartIfExists() {
    if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
    }
}

function hideAllVisualOutputs() {
    document.getElementById('summaryCards').classList.add('hidden');
    document.getElementById('batchResultArea').classList.add('hidden');
    document.getElementById('wordCloudArea').classList.add('hidden');
    document.getElementById('outputContent').classList.add('hidden');
}

function prepareSingleModeOutput() {
    showResultSection();
    hideAllVisualOutputs();
    // 单句分析界面必须只显示文字结果，不保留批量分析图表实例。
    destroyChartIfExists();
}

function setOutputMessage(html) {
    const output = document.getElementById('outputContent');
    output.classList.remove('hidden');
    output.innerHTML = html;
}

function switchMode(mode, event) {
    document.querySelectorAll('.mode-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    const target = document.getElementById(`${mode}Mode`);
    if (target) target.classList.remove('hidden');
    if (event?.currentTarget) event.currentTarget.classList.add('active');

    document.getElementById('resultSection').classList.add('hidden');
    if (mode === 'single') destroyChartIfExists();
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
}

async function checkBackendHealth() {
    const el = document.getElementById('apiStatus');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        el.innerText = `后端已连接：${API_BASE}，模型：${data.model || 'deepseek-chat'}`;
    } catch (e) {
        el.innerText = `后端未连接：请先启动 ${API_BASE}`;
    }
}

async function testAIConnection() {
    showResultSection();
    hideAllVisualOutputs();
    setOutputMessage('正在测试 DeepSeek AI 接口...');
    try {
        const res = await fetch(`${API_BASE}/test-ai`);
        const raw = await res.text();
        let data = null;
        try { data = raw ? JSON.parse(raw) : null; } catch (e) { data = raw; }
        if (!res.ok) throw new Error(data?.detail || data || `HTTP ${res.status}`);
        setOutputMessage(`<h3>✅ AI 接口测试成功</h3><pre>${escapeHTML(JSON.stringify(data.parsed || data, null, 2))}</pre>`);
    } catch (e) {
        setOutputMessage(`<h3>❌ AI 接口测试失败</h3><p>${escapeHTML(e.message)}</p>`);
    }
}

async function callAPI(systemPrompt, text, signal, expectArray = false) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.requestTimeout);
    const abortHandler = () => controller.abort();
    if (signal) signal.addEventListener('abort', abortHandler, { once: true });

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ system_prompt: systemPrompt, text, expect_array: expectArray }),
            signal: controller.signal
        });
        const raw = await response.text();
        let payload = null;
        try { payload = raw ? JSON.parse(raw) : null; } catch (e) { payload = raw; }
        if (!response.ok) throw new Error(payload?.detail || payload || `HTTP ${response.status}`);
        return payload;
    } finally {
        clearTimeout(timeoutId);
        if (signal) signal.removeEventListener('abort', abortHandler);
    }
}

function getCategoryLabel(category) {
    const normalized = normalizeCategory(category);
    if (normalized === 'Positive') return '正向';
    if (normalized === 'Negative') return '负向';
    return '中性';
}

function renderSingleAnalysisResult(score, category) {
    const normalized = normalizeCategory(category);
    const categoryLabel = getCategoryLabel(normalized);
    setOutputMessage(`
        <div class="single-result-card">
            <h3>📌 单句分析结果</h3>
            <p class="hint">单句分析只展示文字结论，不显示图表或统计面板。</p>
            <div class="single-result-grid">
                <div class="single-result-item">
                    <div class="label">情感类别</div>
                    <div class="value">${getCategoryPill(normalized)} ${escapeHTML(categoryLabel)}</div>
                </div>
                <div class="single-result-item">
                    <div class="label">情感得分</div>
                    <div class="value">${escapeHTML(score)}</div>
                    <div class="hint">0 更消极，1 更积极</div>
                </div>
            </div>
        </div>
    `);
}

async function sendRequest() {
    const text = document.getElementById('userInput').value;
    const sys = document.getElementById('systemPrompt').value;
    if (!text.trim()) return alert('请输入文本');

    prepareSingleModeOutput();
    setOutputMessage('分析中...');

    try {
        const res = await callAPI(sys, cleanText(text), null, false);
        const score = res.score !== undefined ? safeScore(res.score) : 'N/A';
        const category = normalizeCategory(res.category);
        prepareSingleModeOutput();
        renderSingleAnalysisResult(score, category);
    } catch (e) {
        prepareSingleModeOutput();
        setOutputMessage(`<h3>❌ 分析失败</h3><p>${escapeHTML(e.message)}</p>`);
    }
}

async function processBatchFile() {
    const file = document.getElementById('batchFileInput').files[0];
    if (!file) return alert('请选择文件');

    abortController = new AbortController();
    showResultSection();
    hideAllVisualOutputs();
    setOutputMessage('📦 正在读取文件并开始批量分析...');

    document.getElementById('progressContainer').classList.remove('hidden');
    document.getElementById('stopBtn').classList.remove('hidden');

    let records = [];
    const ext = file.name.split('.').pop().toLowerCase();
    try {
        if (ext === 'csv') records = await parseCSV(file);
        else if (ext === 'xlsx' || ext === 'xls') records = await parseExcel(file);
        else if (ext === 'docx') records = await parseDocx(file);
        else throw new Error('不支持的文件类型');
    } catch (e) {
        resetProgressUI();
        return alert(`文件解析失败：${e.message}`);
    }

    records = records.map((r, idx) => ({ ...r, rowIndex: idx + 1, text: cleanText(r.text) })).filter(r => r.text.length > 0);
    if (!records.length) {
        resetProgressUI();
        return alert('文件中没有有效文本内容');
    }

    globalData = [];
    currentCounts = null;
    selectedChartCategory = 'All';

    const batchSize = Number(document.getElementById('batchSizeSlider').value || 20);
    const batches = chunkArray(records, batchSize);
    const resultsByBatch = new Array(batches.length);
    const sysPromptBase = document.getElementById('systemPrompt').value;
    const sysPromptBatch = `${sysPromptBase}\n\n请按输入顺序逐条分析，返回 JSON 对象：{"results":[{"score":0.8,"category":"Positive"}]}`;

    let nextBatchIndex = 0;
    let completedRecords = 0;
    const batchErrors = [];

    async function worker() {
        while (nextBatchIndex < batches.length) {
            if (abortController.signal.aborted) return;
            const currentIndex = nextBatchIndex++;
            const batch = batches[currentIndex];
            const combinedText = batch.map((r, idx) => `文本${idx + 1}: ${r.text}`).join('\n');

            try {
                const res = await callAPI(sysPromptBatch, combinedText, abortController.signal, true);
                let resultArray = [];
                if (Array.isArray(res)) resultArray = res;
                else if (Array.isArray(res?.results)) resultArray = res.results;

                while (resultArray.length < batch.length) {
                    resultArray.push({ score: 0.5, category: 'Neutral' });
                }

                resultsByBatch[currentIndex] = batch.map((r, idx) => {
                    const item = resultArray[idx] || {};
                    return {
                        Row_Number: r.rowIndex,
                        ...r.originalRow,
                        original_text: r.text,
                        AI_Score: safeScore(item.score),
                        AI_Category: normalizeCategory(item.category)
                    };
                });
            } catch (e) {
                batchErrors.push(`批次 ${currentIndex + 1}: ${e.message}`);
                resultsByBatch[currentIndex] = batch.map(r => ({
                    Row_Number: r.rowIndex,
                    ...r.originalRow,
                    original_text: r.text,
                    AI_Score: 0.5,
                    AI_Category: 'Neutral',
                    error: e.message
                }));
            }

            completedRecords += batch.length;
            updateProgress(Math.min(completedRecords, records.length), records.length);
        }
    }

    const workerCount = Math.min(CONFIG.concurrency, batches.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));

    globalData = resultsByBatch.filter(Boolean).flat();
    const counts = { Positive: 0, Neutral: 0, Negative: 0 };
    for (const row of globalData) counts[row.AI_Category] = (counts[row.AI_Category] || 0) + 1;
    currentCounts = counts;

    document.getElementById('summaryCards').classList.remove('hidden');
    document.getElementById('batchResultArea').classList.remove('hidden');
    updateSummaryCards(counts, globalData.length);

    if (abortController?.signal.aborted) {
        setOutputMessage(`⏹️ 分析已停止。已完成 ${globalData.length} 条，仍可导出当前结果。`);
    } else if (batchErrors.length) {
        setOutputMessage(`
            <h3>⚠️ 批量分析完成，但有 ${batchErrors.length} 个批次失败</h3>
            <p>失败批次已暂时标记为 Neutral。第一条错误如下：</p>
            <pre>${escapeHTML(batchErrors[0])}</pre>
        `);
    } else {
        setOutputMessage(`<h3>✅ 批量分析完成</h3><p>共处理 <strong>${globalData.length}</strong> 条评论。</p>`);
    }

    renderChart(counts, document.getElementById('chartType').value);
    renderDetailTable();
    document.getElementById('downloadBtn').classList.remove('hidden');
    document.getElementById('downloadExcelBtn').classList.remove('hidden');
    document.getElementById('downloadChartBtn').classList.remove('hidden');
    resetProgressUI();
}

function updateSummaryCards(counts, total) {
    document.getElementById('totalCountValue').innerText = total;
    document.getElementById('positiveCountValue').innerText = counts.Positive || 0;
    document.getElementById('neutralCountValue').innerText = counts.Neutral || 0;
    document.getElementById('negativeCountValue').innerText = counts.Negative || 0;
}

function getChartColors() {
    return {
        Positive: document.getElementById('positiveColor').value,
        Neutral: document.getElementById('neutralColor').value,
        Negative: document.getElementById('negativeColor').value
    };
}

function renderChart(counts, type) {
    const canvas = document.getElementById('myChart');
    if (chartInstance) chartInstance.destroy();

    const labels = ['Positive', 'Neutral', 'Negative'];
    const data = labels.map(label => counts[label] || 0);
    const colors = getChartColors();
    const backgroundColor = labels.map(label => colors[label]);

    chartInstance = new Chart(canvas, {
        type,
        data: {
            labels,
            datasets: [{
                label: '评论数量',
                data,
                backgroundColor,
                borderColor: backgroundColor,
                borderWidth: 2,
                fill: false,
                tension: 0.25
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            onClick: (_, elements) => {
                if (!elements?.length) return;
                const index = elements[0].index;
                const category = labels[index];
                selectedChartCategory = category;
                document.getElementById('detailCategoryFilter').value = category;
                renderDetailTable();
            },
            plugins: {
                legend: { display: true },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.parsed}`
                    }
                }
            },
            scales: (type === 'pie' || type === 'doughnut') ? {} : {
                y: { beginAtZero: true, ticks: { precision: 0 } }
            }
        }
    });
}

function refreshChart() {
    if (!currentCounts) return;
    renderChart(currentCounts, document.getElementById('chartType').value);
}

function getCategoryPill(category) {
    const normalized = normalizeCategory(category);
    const label = normalized === 'Positive' ? '正向' : normalized === 'Negative' ? '负向' : '中性';
    const cls = normalized.toLowerCase();
    return `<span class="pill ${cls}">${label}</span>`;
}

function getFilteredDetailData() {
    const searchValue = document.getElementById('detailSearchInput').value.trim().toLowerCase();
    const categoryValue = document.getElementById('detailCategoryFilter').value;
    return globalData.filter(row => {
        const matchesCategory = categoryValue === 'All' || row.AI_Category === categoryValue;
        const matchesText = !searchValue || String(row.original_text || '').toLowerCase().includes(searchValue);
        return matchesCategory && matchesText;
    });
}

function renderDetailTable() {
    const tbody = document.getElementById('detailTableBody');
    if (!tbody) return;

    const filtered = getFilteredDetailData();
    document.getElementById('detailInfo').innerText = `显示 ${filtered.length} / ${globalData.length} 条`;

    if (!filtered.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="hint">没有符合条件的评论。</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(row => `
        <tr>
            <td>${escapeHTML(row.Row_Number)}</td>
            <td><div class="truncate-text">${escapeHTML(row.original_text)}</div></td>
            <td>${escapeHTML(safeScore(row.AI_Score))}</td>
            <td>${getCategoryPill(row.AI_Category)}</td>
        </tr>
    `).join('');
}

function resetDetailFilters() {
    selectedChartCategory = 'All';
    document.getElementById('detailSearchInput').value = '';
    document.getElementById('detailCategoryFilter').value = 'All';
    renderDetailTable();
}

function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
                const rows = results.data || [];
                resolve(rows.map(row => {
                    const firstValue = Object.values(row)[0];
                    return { text: firstValue ? String(firstValue) : '', originalRow: row };
                }).filter(r => r.text.trim()));
            },
            error: reject
        });
    });
}

function parseExcel(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const workbook = XLSX.read(new Uint8Array(e.target.result), { type: 'array' });
                const sheet = workbook.Sheets[workbook.SheetNames[0]];
                const rows = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: '' });
                const header = rows[0] || [];
                const records = rows.slice(1).map(row => {
                    const originalRow = {};
                    header.forEach((h, idx) => {
                        originalRow[h || `col${idx + 1}`] = row[idx] ?? '';
                    });
                    return {
                        text: row[0] ? String(row[0]) : '',
                        originalRow
                    };
                }).filter(r => r.text.trim());
                resolve(records);
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

function parseDocx(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            mammoth.extractRawText({ arrayBuffer: e.target.result })
                .then(result => {
                    const paragraphs = (result.value || '')
                        .split(/\r?\n/)
                        .map(p => p.trim())
                        .filter(p => p.length > 10);
                    resolve(paragraphs.map((para, idx) => ({
                        text: para,
                        originalRow: { paragraph: idx + 1 }
                    })));
                })
                .catch(reject);
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

function extractKeywordsLocal(text, options = {}) {
    const topN = Number(options.topN || 60);
    const minWordLength = Number(options.minWordLength || 2);
    const minWordCount = Number(options.minWordCount || 2);

    const stopWords = new Set([
        '的', '了', '和', '是', '在', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
        '我们', '你们', '他们', '她们', '它们', '以及', '这个', '那个', '一种', '进行', '可以',
        '需要', '通过', '因为', '所以', '如果', '但是', '然后', '已经', '自己', '不是', '就是',
        '还是', '一些', '这些', '那些', '非常', '比较', '觉得', '真的', '还是', '一下', '一个'
    ]);

    const freq = new Map();
    if (typeof Intl !== 'undefined' && Intl.Segmenter) {
        const segmenter = new Intl.Segmenter('zh-CN', { granularity: 'word' });
        for (const item of segmenter.segment(text)) {
            const word = item.segment.trim().toLowerCase();
            if (!item.isWordLike || word.length < minWordLength || stopWords.has(word)) continue;
            freq.set(word, (freq.get(word) || 0) + 1);
        }
    } else {
        const tokens = text.match(/[\u4e00-\u9fa5]{2,}|[a-zA-Z]{2,}/g) || [];
        for (const token of tokens) {
            const word = token.toLowerCase();
            if (word.length < minWordLength || stopWords.has(word)) continue;
            freq.set(word, (freq.get(word) || 0) + 1);
        }
    }

    return [...freq.entries()]
        .filter(([, count]) => count >= minWordCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topN)
        .map(([word, count]) => ({ word, count, weight: count }));
}

async function generateWordCloud() {
    const file = document.getElementById('kwFileInput').files[0];
    if (!file) return alert('请上传文件');

    let fullText = '';
    const ext = file.name.split('.').pop().toLowerCase();
    try {
        if (ext === 'csv') fullText = (await parseCSV(file)).map(r => r.text).join(' ');
        else if (ext === 'docx') fullText = (await parseDocx(file)).map(r => r.text).join(' ');
        else if (ext === 'txt') fullText = await file.text();
        else return alert('仅支持 .txt、.csv、.docx');
    } catch (e) {
        return alert(`读取文件失败：${e.message}`);
    }

    if (!fullText.trim()) return alert('文档内容为空');

    const options = {
        topN: Number(document.getElementById('keywordTopN').value || 60),
        minWordLength: Number(document.getElementById('minWordLength').value || 2),
        minWordCount: Number(document.getElementById('minWordCount').value || 2)
    };

    showResultSection();
    hideAllVisualOutputs();
    setOutputMessage('🔍 正在本地提取关键词并生成词云图...');

    const keywords = extractKeywordsLocal(fullText, options);
    currentWordCloudKeywords = keywords;
    if (!keywords.length) return setOutputMessage('没有提取到满足条件的关键词。你可以把最小词频调低一点。');

    document.getElementById('wordCloudArea').classList.remove('hidden');
    document.getElementById('outputContent').classList.add('hidden');
    document.getElementById('wordCloudInfo').innerText = `${keywords.length} 个关键词`;
    renderWordCloud(keywords);
}

function renderWordCloud(keywords) {
    const canvas = document.getElementById('wordCloudCanvas');
    const width = canvas.parentElement.clientWidth - 12;
    const height = 480;
    canvas.width = Math.max(800, width);
    canvas.height = height;

    const list = keywords.map(item => [item.word, item.count]);
    const palette = ['#1e40af', '#0f766e', '#7c3aed', '#ea580c', '#be123c', '#15803d'];

    WordCloud(canvas, {
        list,
        gridSize: Math.max(8, Math.round(canvas.width / 80)),
        weightFactor: (size) => Math.max(18, size * 2.6),
        fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
        color: function(word, weight) {
            const idx = Math.abs((word.length + weight)) % palette.length;
            return palette[idx];
        },
        backgroundColor: '#f8fbff',
        rotateRatio: 0.18,
        rotationSteps: 2,
        minSize: 14,
        drawOutOfBound: false,
        shrinkToFit: true,
        shape: 'circle'
    });
}

function downloadWordCloud() {
    const canvas = document.getElementById('wordCloudCanvas');
    if (!currentWordCloudKeywords.length) return alert('请先生成词云图');
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'wordcloud.png';
    link.click();
}

function downloadChartImage() {
    const canvas = document.getElementById('myChart');
    if (!chartInstance) return alert('请先生成图表');
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'sentiment_chart.png';
    link.click();
}

function downloadCSV() {
    if (!globalData.length) return alert('没有可导出的数据');
    const headers = Object.keys(globalData[0]);
    const rows = [headers.join(',')];
    for (const row of globalData) rows.push(headers.map(h => escapeCSV(row[h])).join(','));
    const blob = new Blob([`\ufeff${rows.join('\n')}`], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentiment_analysis_result.csv';
    a.click();
    URL.revokeObjectURL(url);
}

function downloadExcel() {
    if (!globalData.length) return alert('没有可导出的数据');
    const worksheet = XLSX.utils.json_to_sheet(globalData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'AnalysisResult');
    XLSX.writeFile(workbook, 'sentiment_analysis_result.xlsx');
}

function updateProgress(current, total) {
    const percent = total > 0 ? Math.round((current / total) * 100) : 0;
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('progressText').innerText = `${current}/${total}`;
}

function resetProgressUI() {
    document.getElementById('progressContainer').classList.add('hidden');
    document.getElementById('stopBtn').classList.add('hidden');
    abortController = null;
}

function stopAnalysis() {
    if (abortController) {
        abortController.abort();
        setOutputMessage('⏹️ 正在停止，已完成的批次会保留。');
    }
}
