<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { useI18n } from '../composables/useI18n'
import ChatMessage from '../components/ChatMessage.vue'
import RunStep from '../components/RunStep.vue'

const { t } = useI18n()

interface Message { role: 'user' | 'assistant'; content: string }
interface ToolCall {
  function: string
  module_id?: string
  ok?: boolean
  error?: string
  _ems_fix_hint?: string
}

const messages = ref<Message[]>([])
const input = ref('')
const sending = ref(false)
const liveSteps = ref<ToolCall[]>([])
const chatContainer = ref<HTMLElement>()
const streamContent = ref('')
const configured = ref(true)
const lastCost = ref('')
const lastTokens = ref(0)
const lastDuration = ref(0)

onMounted(async () => {
  // Check configuration
  try {
    const res = await fetch('/console/api/setup/status')
    const data = await res.json()
    configured.value = data.configured
  } catch {}

  // Load chat history from SQLite
  try {
    const res = await fetch('/console/api/chat/history')
    const data = await res.json()
    if (data.messages && data.messages.length > 0) {
      messages.value = data.messages.map((m: any) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content || '',
      }))
      scrollToBottom()
    }
  } catch {}
})

async function send() {
  const text = input.value.trim()
  if (!text || sending.value) return

  input.value = ''
  messages.value.push({ role: 'user', content: text })
  sending.value = true
  liveSteps.value = []
  streamContent.value = ''
  lastCost.value = ''
  lastTokens.value = 0
  lastDuration.value = 0

  await nextTick()
  scrollToBottom()

  try {
    const history = messages.value.slice(0, -1).map(m => ({
      role: m.role, content: m.content,
    }))

    // Try batch mode directly (more reliable than SSE for showing full results)
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        history: history.length > 0 ? history : undefined,
        mode: 'execute',
      }),
    })

    const data = await res.json()

    // Show assistant message
    const msg = data.message || data.error || 'No response'
    messages.value.push({ role: 'assistant', content: msg })

    // Show execution steps in the panel
    // Prefer execution_results (actual module runs) over tool_calls (all calls)
    if (data.execution_results && data.execution_results.length > 0) {
      liveSteps.value = data.execution_results.map((er: any) => ({
        function: 'execute_module',
        module_id: er.module_id || '',
        ok: er.ok ?? false,
        error: er.error || '',
        _ems_fix_hint: er._ems_fix_hint || '',
      }))
    } else if (data.tool_calls && data.tool_calls.length > 0) {
      liveSteps.value = data.tool_calls
        .filter((tc: any) => tc.function === 'execute_module')
        .map((tc: any) => ({
          function: tc.function,
          module_id: tc.module_id || tc.arguments?.module_id || '',
          ok: tc.ok ?? false,
          error: tc.error || '',
          _ems_fix_hint: tc._ems_fix_hint || '',
        }))
    }

    // Show cost/usage stats
    if (data.usage) {
      lastTokens.value = data.usage.total_tokens || 0
    }
    if (data.cost) {
      const c = data.cost.session_total_usd || 0
      lastCost.value = c < 0.01 ? '$' + c.toFixed(4) : '$' + c.toFixed(2)
    }
    if (data.rounds_used) {
      lastDuration.value = data.rounds_used
    }

    // Check for not-configured
    if (data.error === 'no_api_key') {
      configured.value = false
    }

  } catch (e: any) {
    messages.value.push({ role: 'assistant', content: 'Connection error: ' + e.message })
  }

  sending.value = false
  streamContent.value = ''
  scrollToBottom()
}

function scrollToBottom() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

function newChat() {
  messages.value = []
  liveSteps.value = []
  input.value = ''
  lastCost.value = ''
  lastTokens.value = 0
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="h-full flex">
    <!-- Chat area -->
    <div class="flex-1 flex flex-col min-w-0">
      <!-- Top bar -->
      <div class="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <span class="text-sm text-gray-400">{{ t('nav.chat') }}</span>
        <div class="flex items-center gap-3">
          <span v-if="lastCost" class="text-xs text-green-400">{{ lastCost }}</span>
          <span v-if="lastTokens" class="text-xs text-gray-500">{{ lastTokens.toLocaleString() }} tokens</span>
          <button
            @click="newChat"
            class="text-xs text-gray-500 hover:text-gray-300 px-2 py-1 rounded hover:bg-gray-800"
          >
            {{ t('chat.newChat') }}
          </button>
        </div>
      </div>

      <!-- Not configured warning -->
      <div v-if="!configured" class="mx-4 mt-3 p-3 bg-amber-900/20 border border-amber-700/50 rounded-lg text-sm text-amber-300 flex items-center gap-2">
        <span class="i-carbon-warning text-lg" />
        {{ t('setup.connectionFailed') }} — Go to Settings → AI Provider
      </div>

      <!-- Messages -->
      <div ref="chatContainer" class="flex-1 overflow-y-auto p-4 space-y-3">
        <div v-if="messages.length === 0 && configured" class="flex items-center justify-center h-full">
          <div class="text-center text-gray-600">
            <img src="../assets/logo.png" alt="" class="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p class="text-sm">{{ t('chat.placeholder') }}</p>
          </div>
        </div>
        <ChatMessage
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
        />
        <div v-if="sending" class="flex gap-3">
          <div class="bg-gray-800 rounded-xl px-4 py-2.5 text-sm text-gray-400">
            <span class="animate-pulse">{{ t('chat.executing') }}</span>
          </div>
        </div>
      </div>

      <!-- Input -->
      <div class="p-3 border-t border-gray-800">
        <div class="flex gap-2 max-w-4xl mx-auto">
          <textarea
            v-model="input"
            @keydown="handleKeydown"
            :placeholder="t('chat.placeholder')"
            :disabled="sending"
            rows="1"
            class="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-200 placeholder-gray-600 resize-none focus:outline-none focus:border-brand-500 disabled:opacity-50"
          />
          <button
            @click="send"
            :disabled="!input.trim() || sending"
            class="px-5 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 rounded-xl text-sm font-medium transition-colors"
          >
            <span v-if="sending" class="i-carbon-stop-filled" />
            <span v-else class="i-carbon-send" />
          </button>
        </div>
      </div>
    </div>

    <!-- Execution panel -->
    <div class="w-72 border-l border-gray-800 bg-gray-900/50 flex flex-col shrink-0">
      <div class="px-3 py-2 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wide">
        {{ t('run.toolCalls') }}
      </div>
      <div class="flex-1 overflow-y-auto p-2 space-y-1">
        <template v-if="liveSteps.length">
          <RunStep
            v-for="(step, i) in liveSteps"
            :key="i"
            :module-id="step.module_id || step.function"
            :status="step.ok === undefined ? 'running' : step.ok ? 'ok' : 'failed'"
            :error="step.error"
            :fix-hint="step._ems_fix_hint"
          />
        </template>
        <div v-else class="text-xs text-gray-600 p-2 text-center">
          {{ t('run.noRuns') }}
        </div>
      </div>

      <!-- Stats -->
      <div class="p-3 border-t border-gray-800 text-xs text-gray-500 space-y-1">
        <div class="flex justify-between">
          <span>{{ t('run.toolCalls') }}</span>
          <span class="text-gray-400">{{ liveSteps.length }}</span>
        </div>
        <div class="flex justify-between">
          <span>{{ t('status.ok') }}</span>
          <span class="text-green-400">{{ liveSteps.filter(s => s.ok === true).length }}</span>
        </div>
        <div class="flex justify-between">
          <span>{{ t('status.error') }}</span>
          <span class="text-red-400">{{ liveSteps.filter(s => s.ok === false).length }}</span>
        </div>
        <div v-if="lastCost" class="flex justify-between pt-1 border-t border-gray-800">
          <span>{{ t('run.cost') }}</span>
          <span class="text-green-400">{{ lastCost }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
