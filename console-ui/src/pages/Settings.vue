<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi, post } from '../composables/useApi'

const keys = useApi<any[]>('/keys')
const license = useApi<any>('/license')
const budget = useApi<any>('/budget')
const modules = useApi<any>('/modules')
const setupStatus = useApi<any>('/setup/status')
const channels = useApi<any>('/channels')

// Provider switch
const providerOptions = [
  { id: 'openai', name: 'OpenAI', color: 'text-green-400' },
  { id: 'anthropic', name: 'Anthropic', color: 'text-amber-400' },
  { id: 'ollama', name: 'Ollama (Local)', color: 'text-blue-400' },
  { id: 'deepseek', name: 'DeepSeek', color: 'text-cyan-400' },
  { id: 'custom', name: 'Custom', color: 'text-gray-400' },
]
const switchProvider = ref('')
const switchKey = ref('')
const switchUrl = ref('')
const switching = ref(false)
const switchMsg = ref('')

// Add key form
const newProvider = ref('')
const newKey = ref('')
const saving = ref(false)

// License activation
const licenseKey = ref('')
const activating = ref(false)
const activateMsg = ref('')

// Budget setting
const budgetSession = ref('')
const budgetGlobal = ref('')
const budgetMsg = ref('')

// Channel setting
const channelName = ref('')
const channelToken = ref('')
const channelMsg = ref('')

onMounted(() => {
  keys.fetch()
  license.fetch()
  budget.fetch()
  modules.fetch()
  setupStatus.fetch()
  channels.fetch()
})

async function saveBudget() {
  const body: any = {}
  if (budgetSession.value) body.session_budget_usd = parseFloat(budgetSession.value)
  if (budgetGlobal.value) body.global_budget_usd = parseFloat(budgetGlobal.value)
  const res = await post('/budget/set', body)
  budgetMsg.value = res.ok ? 'Budget updated' : 'Error: ' + res.error
  budget.fetch()
}

async function resetBudget() {
  const res = await post('/budget/reset', {})
  budgetMsg.value = res.ok ? 'Session reset' : 'Error: ' + res.error
  budget.fetch()
}

async function saveChannel() {
  if (!channelName.value || !channelToken.value) return
  const res = await post('/channels/set', {
    channel: channelName.value,
    token: channelToken.value,
  })
  channelMsg.value = res.ok ? channelName.value + ' configured' : 'Error: ' + res.error
  channelToken.value = ''
  channels.fetch()
}

async function changeProvider() {
  if (!switchProvider.value) return
  switching.value = true
  switchMsg.value = ''
  const res = await post('/setup', {
    provider: switchProvider.value,
    api_key: switchKey.value,
    base_url: switchUrl.value,
  })
  switching.value = false
  if (res.ok) {
    switchMsg.value = 'Switched to ' + res.model
    switchKey.value = ''
    switchUrl.value = ''
    keys.fetch()
    setupStatus.fetch()
  } else {
    switchMsg.value = 'Error: ' + res.error
  }
}

async function addKey() {
  if (!newProvider.value || !newKey.value) return
  saving.value = true
  await post('/keys', { provider: newProvider.value, api_key: newKey.value })
  newProvider.value = ''
  newKey.value = ''
  saving.value = false
  keys.fetch()
}

async function deleteKey(provider: string) {
  await post('/keys/delete', { provider })
  keys.fetch()
}

async function activateLicense() {
  if (!licenseKey.value) return
  activating.value = true
  activateMsg.value = ''
  const res = await post('/license/activate', { key: licenseKey.value })
  activateMsg.value = res.ok ? `Activated: ${res.tier}` : `Error: ${res.error}`
  activating.value = false
  if (res.ok) {
    licenseKey.value = ''
    license.fetch()
  }
}
</script>

<template>
  <div class="p-6 max-w-4xl mx-auto space-y-8">
    <h2 class="text-xl font-semibold">Settings</h2>

    <!-- Active Provider -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 class="font-medium text-sm text-gray-300">AI Provider</h3>
      <div v-if="setupStatus.data.value?.configured" class="flex items-center gap-3">
        <span class="text-sm text-green-400 i-carbon-checkmark-filled" />
        <span class="text-sm">{{ setupStatus.data.value.provider }}</span>
        <span class="text-xs text-gray-500">{{ setupStatus.data.value.model }}</span>
      </div>
      <div v-else class="text-sm text-amber-400 flex items-center gap-2">
        <span class="i-carbon-warning" />
        Not configured
      </div>

      <div class="flex gap-2 pt-2 border-t border-gray-800">
        <select
          v-model="switchProvider"
          class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
        >
          <option value="">Switch provider...</option>
          <option v-for="p in providerOptions" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
        <input
          v-if="switchProvider && switchProvider !== 'ollama'"
          v-model="switchKey"
          type="password"
          placeholder="API key"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600"
        />
        <input
          v-if="switchProvider === 'custom' || switchProvider === 'deepseek'"
          v-model="switchUrl"
          placeholder="Base URL"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600"
        />
        <button
          @click="changeProvider"
          :disabled="!switchProvider || switching"
          class="px-4 py-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
        >{{ switching ? '...' : 'Switch' }}</button>
      </div>
      <p v-if="switchMsg" class="text-xs" :class="switchMsg.startsWith('Error') ? 'text-red-400' : 'text-green-400'">
        {{ switchMsg }}
      </p>
    </section>

    <!-- API Keys -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 class="font-medium text-sm text-gray-300">API Keys</h3>

      <div v-if="keys.data.value?.length" class="space-y-2">
        <div
          v-for="key in keys.data.value"
          :key="key.provider"
          class="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-2.5"
        >
          <div>
            <span class="font-medium text-sm">{{ key.provider }}</span>
            <span class="text-xs text-gray-500 ml-2">{{ key.source }}</span>
          </div>
          <div class="flex items-center gap-3">
            <code class="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">{{ key.masked }}</code>
            <button
              v-if="key.source === 'vault'"
              @click="deleteKey(key.provider)"
              class="text-xs text-red-400 hover:text-red-300"
            >Remove</button>
          </div>
        </div>
      </div>
      <p v-else class="text-sm text-gray-500">No API keys configured</p>

      <!-- Add key form -->
      <form @submit.prevent="addKey" class="flex gap-2 pt-2 border-t border-gray-800" autocomplete="off">
        <select
          v-model="newProvider"
          class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200"
        >
          <option value="">Provider...</option>
          <option value="openai_login">OpenAI</option>
          <option value="anthropic_login">Anthropic</option>
          <option value="custom">Custom</option>
        </select>
        <input
          v-model="newKey"
          type="password"
          placeholder="API key"
          autocomplete="new-password"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600"
        />
        <button
          type="submit"
          :disabled="saving || !newProvider || !newKey"
          class="px-4 py-1.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
        >Save</button>
      </form>
    </section>

    <!-- License -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 class="font-medium text-sm text-gray-300">License</h3>

      <div v-if="license.data.value" class="flex items-center gap-4">
        <span
          class="px-3 py-1 rounded-full text-xs font-bold uppercase"
          :class="{
            'bg-gray-800 text-gray-400': license.data.value.tier === 'free',
            'bg-brand-600/20 text-brand-400': license.data.value.tier === 'pro',
            'bg-amber-600/20 text-amber-400': license.data.value.tier === 'enterprise',
          }"
        >{{ license.data.value.tier }}</span>
        <div class="text-sm text-gray-400 space-x-4">
          <span>Core: {{ license.data.value.core_available ? '✓' : '✗' }}</span>
          <span>Premium: {{ license.data.value.premium_available ? '✓' : '✗' }}</span>
        </div>
      </div>

      <!-- Activate -->
      <div class="flex gap-2 pt-2 border-t border-gray-800">
        <input
          v-model="licenseKey"
          placeholder="Enter license key"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600"
        />
        <button
          @click="activateLicense"
          :disabled="activating || !licenseKey"
          class="px-4 py-1.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors"
        >Activate</button>
      </div>
      <p v-if="activateMsg" class="text-xs" :class="activateMsg.startsWith('Error') ? 'text-red-400' : 'text-green-400'">
        {{ activateMsg }}
      </p>
    </section>

    <!-- Budget -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="font-medium text-sm text-gray-300">Budget</h3>
        <button @click="resetBudget" class="text-xs text-gray-500 hover:text-gray-300">Reset Session</button>
      </div>

      <!-- Current usage -->
      <div v-if="budget.data.value" class="grid grid-cols-2 gap-3 text-sm">
        <div class="bg-gray-800/50 rounded-lg p-3">
          <p class="text-xs text-gray-500">Session Budget</p>
          <p class="font-mono text-lg">{{ budget.data.value.session_budget_usd ? '$' + budget.data.value.session_budget_usd : '∞' }}</p>
        </div>
        <div class="bg-gray-800/50 rounded-lg p-3">
          <p class="text-xs text-gray-500">Global Budget</p>
          <p class="font-mono text-lg">{{ budget.data.value.global_budget_usd ? '$' + budget.data.value.global_budget_usd : '∞' }}</p>
        </div>
      </div>

      <!-- Set budget -->
      <div class="flex gap-2 pt-2 border-t border-gray-800">
        <input v-model="budgetSession" type="number" step="0.1" min="0" placeholder="Session $ limit"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600" />
        <input v-model="budgetGlobal" type="number" step="1" min="0" placeholder="Global $ limit"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600" />
        <button @click="saveBudget" class="px-4 py-2 bg-brand-600 hover:bg-brand-700 rounded-lg text-sm font-medium transition-colors">Set</button>
      </div>
      <p v-if="budgetMsg" class="text-xs" :class="budgetMsg.startsWith('Error') ? 'text-red-400' : 'text-green-400'">{{ budgetMsg }}</p>
    </section>

    <!-- Channels (Telegram, Slack, Discord) -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <h3 class="font-medium text-sm text-gray-300">Channels</h3>

      <div v-if="channels.data.value" class="space-y-2">
        <div v-for="(ch, name) in channels.data.value" :key="name"
          class="flex items-center justify-between bg-gray-800/50 rounded-lg px-4 py-2.5">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full" :class="ch.configured ? 'bg-green-500' : 'bg-gray-600'" />
            <span class="text-sm font-medium capitalize">{{ name }}</span>
          </div>
          <span class="text-xs text-gray-500">{{ ch.configured ? 'Connected' : ch.env_var }}</span>
        </div>
      </div>

      <div class="flex gap-2 pt-2 border-t border-gray-800">
        <select v-model="channelName" class="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200">
          <option value="">Channel...</option>
          <option value="telegram">Telegram</option>
          <option value="slack">Slack</option>
          <option value="discord">Discord</option>
          <option value="webhook">Webhook</option>
        </select>
        <input v-model="channelToken" :placeholder="channelName === 'webhook' ? 'Webhook URL' : 'Bot Token'"
          class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600" />
        <button @click="saveChannel" :disabled="!channelName || !channelToken"
          class="px-4 py-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-40 rounded-lg text-sm font-medium transition-colors">Connect</button>
      </div>
      <p v-if="channelMsg" class="text-xs text-green-400">{{ channelMsg }}</p>
    </section>

    <!-- Modules -->
    <section class="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
      <h3 class="font-medium text-sm text-gray-300">
        Modules
        <span v-if="modules.data.value" class="text-gray-500 font-normal ml-1">
          ({{ modules.data.value.total }} total)
        </span>
      </h3>
      <div v-if="modules.data.value?.categories" class="flex flex-wrap gap-2">
        <span
          v-for="(cat, name) in modules.data.value.categories"
          :key="name"
          class="text-xs bg-gray-800 text-gray-300 px-2.5 py-1 rounded-full"
        >
          {{ name }}
          <span class="text-gray-500 ml-1">{{ cat.count }}</span>
        </span>
      </div>
    </section>
  </div>
</template>
