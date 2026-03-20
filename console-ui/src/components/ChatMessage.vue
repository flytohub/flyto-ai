<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  role: 'user' | 'assistant'
  content: string
}>()

// Simple markdown rendering — code blocks, bold, links
const rendered = computed(() => {
  if (props.role === 'user') return escapeHtml(props.content)

  let html = escapeHtml(props.content)

  // Code blocks: ```yaml ... ``` → <pre><code>
  html = html.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_m, lang, code) =>
      `<pre class="bg-gray-900 border border-gray-700 rounded-lg p-3 my-2 overflow-x-auto text-xs font-mono"><code class="language-${lang}">${code.trim()}</code></pre>`
  )

  // Inline code: `text` → <code>
  html = html.replace(
    /`([^`]+)`/g,
    '<code class="bg-gray-700 px-1.5 py-0.5 rounded text-xs font-mono text-brand-400">$1</code>'
  )

  // Bold: **text** → <strong>
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')

  // Links: [text](url) → <a>
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,
    '<a href="$2" target="_blank" class="text-brand-400 hover:underline">$1</a>'
  )

  // Bare URLs
  html = html.replace(
    /(?<!")(?<!=)(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" class="text-brand-400 hover:underline">$1</a>'
  )

  // Numbered lists: 1. text → proper list styling
  html = html.replace(
    /^(\d+)\.\s+(.+)$/gm,
    '<div class="flex gap-2 my-0.5"><span class="text-gray-500 shrink-0">$1.</span><span>$2</span></div>'
  )

  // Line breaks (but not inside <pre>)
  html = html.replace(/\n(?![^<]*<\/pre>)/g, '<br>')

  return html
})

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
</script>

<template>
  <div class="flex gap-3" :class="role === 'user' ? 'justify-end' : ''">
    <!-- User message -->
    <div
      v-if="role === 'user'"
      class="max-w-[70%] rounded-2xl rounded-br-md px-4 py-2.5 text-sm bg-brand-600 text-white"
    >
      {{ content }}
    </div>

    <!-- Assistant message -->
    <div v-else class="max-w-[85%] flex gap-2.5">
      <div class="w-7 h-7 rounded-full bg-gray-800 flex items-center justify-center shrink-0 mt-0.5">
        <img src="../assets/logo.png" alt="" class="w-4 h-4" />
      </div>
      <div
        class="rounded-2xl rounded-tl-md px-4 py-3 text-sm bg-gray-800/80 text-gray-200 leading-relaxed message-content"
        v-html="rendered"
      />
    </div>
  </div>
</template>

<style scoped>
.message-content :deep(pre) {
  margin: 0.5rem 0;
}
.message-content :deep(a) {
  word-break: break-all;
}
</style>
