// 全局变量
let currentMethod = 'semantic';
let users = [];

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadUsers();
    initMethodButtons();
});

// 加载统计数据
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        document.getElementById('totalNews').textContent = data.total_news;
        document.getElementById('totalUsers').textContent = data.total_users;
    } catch (error) {
        console.error('加载统计失败:', error);
    }
}

// 加载用户列表
async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        const data = await response.json();
        users = data.users;

        // 填充用户选择器
        const userSelect = document.getElementById('userId');
        const fusionUserSelect = document.getElementById('fusionUserId');

        userSelect.innerHTML = '<option value="">-- 请选择用户 --</option>';
        fusionUserSelect.innerHTML = '<option value="">-- 选择用户 --</option>';

        users.forEach(user => {
            const option = document.createElement('option');
            option.value = user;
            option.textContent = user;
            userSelect.appendChild(option);

            const option2 = document.createElement('option');
            option2.value = user;
            option2.textContent = user;
            fusionUserSelect.appendChild(option2);
        });

        console.log(`加载了 ${users.length} 个用户`);
    } catch (error) {
        console.error('加载用户列表失败:', error);
    }
}

// 初始化方法按钮
function initMethodButtons() {
    const buttons = document.querySelectorAll('.method-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMethod = btn.dataset.method;
        });
    });
}

// 检索
async function doSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    if (!query) {
        alert('请输入查询内容');
        return;
    }

    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.innerHTML = '<div class="empty-state">⏳ 检索中...</div>';

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                method: currentMethod,
                top_k: 10
            })
        });

        const data = await response.json();
        if (data.error) {
            resultsDiv.innerHTML = `<div class="empty-state">❌ ${data.error}</div>`;
            return;
        }

        displayResults(data.results, 'searchResults');
        document.getElementById('searchResultCount').textContent = `(${data.results.length}条)`;

    } catch (error) {
        resultsDiv.innerHTML = '<div class="empty-state">❌ 检索失败，请重试</div>';
    }
}

// 推荐
async function doRecommend() {
    const userId = document.getElementById('userId').value;
    if (!userId) {
        alert('请选择用户');
        return;
    }

    const resultsDiv = document.getElementById('recommendResults');
    resultsDiv.innerHTML = '<div class="empty-state">⏳ 推荐中...</div>';

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                top_k: 10
            })
        });

        const data = await response.json();
        if (data.error) {
            resultsDiv.innerHTML = `<div class="empty-state">❌ ${data.error}</div>`;
            return;
        }

        displayResults(data.results, 'recommendResults');
        document.getElementById('recommendResultCount').textContent = `(${data.results.length}条)`;

    } catch (error) {
        resultsDiv.innerHTML = '<div class="empty-state">❌ 推荐失败，请重试</div>';
    }
}

// 融合检索
async function doFusion() {
    const query = document.getElementById('fusionQuery').value.trim();
    const userId = document.getElementById('fusionUserId').value;

    if (!query) {
        alert('请输入查询内容');
        return;
    }
    if (!userId) {
        alert('请选择用户');
        return;
    }

    const resultsDiv = document.getElementById('fusionResults');
    resultsDiv.innerHTML = '<div class="empty-state">⏳ 融合检索中...</div>';

    try {
        const response = await fetch('/api/fusion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                user_id: userId,
                top_k: 10
            })
        });

        const data = await response.json();
        if (data.error) {
            resultsDiv.innerHTML = `<div class="empty-state">❌ ${data.error}</div>`;
            return;
        }

        displayResults(data.results, 'fusionResults');
        document.getElementById('fusionResultCount').textContent = `(${data.results.length}条)`;

    } catch (error) {
        resultsDiv.innerHTML = '<div class="empty-state">❌ 融合检索失败，请重试</div>';
    }
}

// 加载用户信息
async function loadUserInfo() {
    const userId = document.getElementById('userId').value;
    if (!userId) {
        document.getElementById('userProfile').style.display = 'none';
        return;
    }

    // 同时更新融合选择的用户
    const fusionSelect = document.getElementById('fusionUserId');
    if (fusionSelect.querySelector(`option[value="${userId}"]`)) {
        fusionSelect.value = userId;
    }

    try {
        const response = await fetch(`/api/user_info?user_id=${userId}`);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        document.getElementById('userName').textContent = data.username || data.user_id;
        document.getElementById('userActivity').textContent = `活跃度: ${data.activity_level}`;

        const prefsDiv = document.getElementById('userPrefs');
        prefsDiv.innerHTML = '';
        const sortedPrefs = Object.entries(data.preferences)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8);

        if (sortedPrefs.length === 0) {
            prefsDiv.innerHTML = '<span class="pref-tag">暂无偏好数据</span>';
        } else {
            sortedPrefs.forEach(([cat, score]) => {
                const tag = document.createElement('span');
                tag.className = 'pref-tag';
                tag.textContent = `${cat} ${(score * 100).toFixed(0)}%`;
                prefsDiv.appendChild(tag);
            });
        }

        document.getElementById('userProfile').style.display = 'block';

    } catch (error) {
        console.error('加载用户信息失败:', error);
    }
}

// 显示结果
function displayResults(results, containerId) {
    const container = document.getElementById(containerId);
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="empty-state">📭 未找到相关结果</div>';
        return;
    }

    container.innerHTML = results.map(item => `
        <div class="result-item">
            <div class="result-title">${escapeHtml(item.title)}</div>
            <div class="result-meta">
                <span class="result-category">📁 ${item.category || '未分类'}</span>
                <span>👁️ ${formatNumber(item.views)}</span>
                <span>❤️ ${formatNumber(item.likes)}</span>
                <span class="result-score">⭐ 相关度: ${(item.score * 100).toFixed(1)}%</span>
            </div>
            ${item.keywords && item.keywords.length ? `
                <div class="result-keywords">
                    ${item.keywords.slice(0, 5).map(kw => `<span class="keyword-tag">#${escapeHtml(kw)}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
}

// 辅助函数
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 10000) return (num / 10000).toFixed(1) + 'w';
    return num.toString();
}