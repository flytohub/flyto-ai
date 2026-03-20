import { ref, computed } from 'vue'

// Locale code → native display name
const LOCALE_NAMES: Record<string, string> = {
  de: 'Deutsch',
  en: 'English',
  es: 'Español',
  fr: 'Français',
  hi: 'हिन्दी',
  id: 'Bahasa Indonesia',
  it: 'Italiano',
  ja: '日本語',
  ko: '한국어',
  pl: 'Polski',
  'pt-BR': 'Português (BR)',
  th: 'ไทย',
  tr: 'Türkçe',
  vi: 'Tiếng Việt',
  'zh-CN': '简体中文',
  'zh-TW': '繁體中文',
}

// All locale files are imported at build time (eager = bundled inline)
const localeModules = import.meta.glob('/locales/*/ai.console.json', { eager: true }) as Record<string, any>

// Parse available locales from file paths
const availableLocales: Record<string, Record<string, string>> = {}
for (const [path, mod] of Object.entries(localeModules)) {
  const match = path.match(/\/locales\/([^/]+)\//)
  if (!match) continue
  const locale = match[1]
  // Vite eager import: mod is the JSON object directly, or mod.default
  const raw = mod.default || mod
  const translations = raw.translations || raw
  if (typeof translations === 'object') {
    availableLocales[locale] = translations
  }
}

function detectLocale(): string {
  const saved = localStorage.getItem('flyto-ai-locale')
  if (saved && availableLocales[saved]) return saved

  const browserLang = navigator.language
  if (availableLocales[browserLang]) return browserLang

  // zh-TW, zh-CN fallback
  const base = browserLang.split('-')[0]
  const match = Object.keys(availableLocales).find(k => k.startsWith(base))
  if (match) return match

  return 'en'
}

const currentLocale = ref(detectLocale())
const translations = computed(() => availableLocales[currentLocale.value] || availableLocales['en'] || {})

export function useI18n() {
  function t(key: string, fallback?: string): string {
    return translations.value[key] || availableLocales['en']?.[key] || fallback || key
  }

  function setLocale(locale: string) {
    if (availableLocales[locale]) {
      currentLocale.value = locale
      localStorage.setItem('flyto-ai-locale', locale)
    }
  }

  function localeName(code: string): string {
    return LOCALE_NAMES[code] || code
  }

  return {
    t,
    locale: currentLocale,
    setLocale,
    localeName,
    availableLocales: Object.keys(availableLocales),
  }
}
