<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from './composables/useI18n'
import { useApi } from './composables/useApi'
import Chat from './pages/Chat.vue'
import Dashboard from './pages/Dashboard.vue'
import Executions from './pages/Executions.vue'
import Settings from './pages/Settings.vue'
import Setup from './pages/Setup.vue'

const { t, locale, setLocale, localeName, availableLocales } = useI18n()

type Page = 'chat' | 'runs' | 'data' | 'settings'
const currentPage = ref<Page>('chat')
const showSetup = ref(false)

const navItems: { id: Page; key: string; icon: string }[] = [
  { id: 'chat', key: 'nav.chat', icon: 'i-carbon-chat' },
  { id: 'runs', key: 'nav.runs', icon: 'i-carbon-activity' },
  { id: 'data', key: 'nav.data', icon: 'i-carbon-dashboard' },
  { id: 'settings', key: 'nav.settings', icon: 'i-carbon-settings' },
]

const overview = useApi<any>('/overview')

onMounted(() => {
  if (!localStorage.getItem('flyto-ai-setup-done')) {
    showSetup.value = true
  }
  overview.fetch()
})

function onSetupDone() {
  showSetup.value = false
  overview.fetch()
}

function fmt(n: number) {
  return n < 0.01 ? '$' + n.toFixed(4) : '$' + n.toFixed(2)
}
</script>

<template>
  <!-- Setup wizard -->
  <Setup v-if="showSetup" @done="onSetupDone" />

  <!-- Main app -->
  <div v-else class="h-screen bg-gray-950 text-gray-100 flex flex-col overflow-hidden">
    <!-- Top bar -->
    <header class="h-11 bg-gray-900/80 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
      <img src="./assets/logo.png" alt="" class="w-6 h-6" />

      <div class="flex-1" />

      <!-- Quick stats -->
      <template v-if="overview.data.value">
        <span class="text-xs text-gray-500">
          {{ overview.data.value.model || '' }}
        </span>
        <span v-if="overview.data.value.cost?.session_total_usd > 0"
              class="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400">
          {{ fmt(overview.data.value.cost.session_total_usd) }}
        </span>
        <span class="text-xs text-gray-600">
          {{ overview.data.value.module_count }} {{ t('data.modules') }}
        </span>
        <span
          class="text-xs px-2 py-0.5 rounded font-bold uppercase"
          :class="{
            'bg-gray-800 text-gray-500': overview.data.value.license_tier === 'free',
            'bg-brand-600/20 text-brand-400': overview.data.value.license_tier === 'pro',
            'bg-amber-600/20 text-amber-400': overview.data.value.license_tier === 'enterprise',
          }"
        >{{ t('license.' + overview.data.value.license_tier, overview.data.value.license_tier) }}</span>
      </template>

      <!-- Language -->
      <select
        :value="locale"
        @change="setLocale(($event.target as HTMLSelectElement).value)"
        class="bg-transparent border-none text-xs text-gray-500 cursor-pointer focus:outline-none"
      >
        <option v-for="loc in availableLocales" :key="loc" :value="loc" class="bg-gray-900">
          {{ localeName(loc) }}
        </option>
      </select>
    </header>

    <div class="flex flex-1 overflow-hidden">
      <!-- Sidebar (icon-only) -->
      <aside class="w-14 bg-gray-900 border-r border-gray-800 flex flex-col items-center py-3 gap-1 shrink-0">
        <button
          v-for="item in navItems"
          :key="item.id"
          @click="currentPage = item.id"
          :title="t(item.key)"
          class="w-10 h-10 flex items-center justify-center rounded-lg transition-colors relative group"
          :class="currentPage === item.id
            ? 'bg-brand-600/20 text-brand-400'
            : 'text-gray-500 hover:bg-gray-800 hover:text-gray-300'"
        >
          <span :class="item.icon" class="text-lg" />
          <!-- Tooltip -->
          <span class="absolute left-12 px-2 py-1 bg-gray-800 text-xs text-gray-200 rounded shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
            {{ t(item.key) }}
          </span>
        </button>
      </aside>

      <!-- Content -->
      <main class="flex-1 overflow-hidden">
        <Chat v-if="currentPage === 'chat'" class="h-full" />
        <div v-else class="h-full overflow-y-auto">
          <Executions v-if="currentPage === 'runs'" />
          <Dashboard v-else-if="currentPage === 'data'" />
          <Settings v-else-if="currentPage === 'settings'" />
        </div>
      </main>
    </div>
  </div>
</template>
