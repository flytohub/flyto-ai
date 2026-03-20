<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { post } from '../composables/useApi'

const { t } = useI18n()
const emit = defineEmits<{ done: [] }>()

const providers = [
  { id: 'openai', name: 'OpenAI', model: 'gpt-4o', icon: 'i-carbon-machine-learning-model', color: 'text-green-400', needsKey: true },
  { id: 'anthropic', name: 'Anthropic', model: 'Claude', icon: 'i-carbon-bot', color: 'text-amber-400', needsKey: true },
  { id: 'ollama', name: 'Ollama', model: '', icon: 'i-carbon-home', color: 'text-blue-400', needsKey: false },
  { id: 'deepseek', name: 'DeepSeek', model: 'deepseek-chat', icon: 'i-carbon-search', color: 'text-cyan-400', needsKey: true },
  { id: 'custom', name: 'Custom', model: '', icon: 'i-carbon-settings-adjust', color: 'text-gray-400', needsKey: true },
]

const selected = ref('')
const apiKey = ref('')
const baseUrl = ref('')
const testing = ref(false)
const testResult = ref<'ok' | 'fail' | ''>('')

function selectProvider(id: string) {
  selected.value = id
  testResult.value = ''
  apiKey.value = ''
  baseUrl.value = ''
}

async function testConnection() {
  testing.value = true
  testResult.value = ''
  try {
    const res = await post('/keys', {
      provider: selected.value + '_login',
      api_key: apiKey.value,
    })
    testResult.value = res.ok ? 'ok' : 'fail'
  } catch {
    testResult.value = 'fail'
  }
  testing.value = false
}

async function finish() {
  // Send provider + key to server
  const provider = providers.find(p => p.id === selected.value)
  if (selected.value) {
    const res = await post('/setup', {
      provider: selected.value,
      api_key: apiKey.value,
      base_url: baseUrl.value,
    })
    if (!res.ok) {
      testResult.value = 'fail'
      return
    }
  }
  localStorage.setItem('flyto-ai-setup-done', '1')
  emit('done')
}

function skip() {
  localStorage.setItem('flyto-ai-setup-done', '1')
  emit('done')
}
</script>

<template>
  <div class="min-h-screen bg-gray-950 flex items-center justify-center">
    <div class="max-w-xl w-full p-8">
      <!-- Header with logo -->
      <div class="text-center mb-10">
        <img src="../assets/logo.png" alt="" class="w-16 h-16 mx-auto mb-4" />
        <h1 class="text-3xl font-bold text-white">{{ t('setup.welcome') }}</h1>
        <p class="text-gray-400 mt-2 text-lg">{{ t('setup.chooseProvider') }}</p>
      </div>

      <!-- Provider cards -->
      <div class="grid grid-cols-3 gap-3 mb-6">
        <button
          v-for="p in providers"
          :key="p.id"
          @click="selectProvider(p.id)"
          class="border rounded-xl p-5 text-center transition-all"
          :class="selected === p.id
            ? 'border-brand-500 bg-brand-600/15 shadow-lg shadow-brand-500/10'
            : 'border-gray-700 bg-gray-900 hover:border-gray-500 hover:bg-gray-800/80'"
        >
          <span :class="[p.icon, p.color]" class="text-3xl block mb-2" />
          <div class="font-semibold text-white text-sm">{{ p.name }}</div>
          <div class="text-xs text-gray-500 mt-1">
            {{ p.id === 'ollama' ? t('setup.ollamaFree') : p.id === 'custom' ? t('setup.customBaseUrl') : p.model }}
          </div>
        </button>
      </div>

      <!-- API key input -->
      <div v-if="selected" class="space-y-3">
        <template v-if="providers.find(p => p.id === selected)?.needsKey">
          <input
            v-model="apiKey"
            type="password"
            :placeholder="t('setup.pasteKey')"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
          />
          <input
            v-if="selected === 'custom'"
            v-model="baseUrl"
            placeholder="https://api.example.com/v1"
            class="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
          />
        </template>

        <div class="flex gap-3 pt-1">
          <button
            v-if="providers.find(p => p.id === selected)?.needsKey"
            @click="testConnection"
            :disabled="!apiKey || testing"
            class="flex-1 py-3 border border-gray-600 rounded-lg text-sm text-gray-300 hover:bg-gray-800 disabled:opacity-40 transition-colors"
          >
            {{ testing ? '...' : t('setup.testConnection') }}
          </button>
          <button
            @click="finish"
            class="flex-1 py-3 bg-brand-600 hover:bg-brand-500 rounded-lg text-sm font-semibold text-white transition-colors"
          >
            {{ t('setup.letsGo') }}
          </button>
        </div>

        <p v-if="testResult === 'ok'" class="text-sm text-green-400 text-center">
          {{ t('setup.connectionOk') }}
        </p>
        <p v-if="testResult === 'fail'" class="text-sm text-red-400 text-center">
          {{ t('setup.connectionFailed') }}
        </p>
      </div>

      <button @click="skip" class="w-full mt-6 text-sm text-gray-500 hover:text-gray-300 transition-colors">
        {{ t('setup.skip') }}
      </button>
    </div>
  </div>
</template>
