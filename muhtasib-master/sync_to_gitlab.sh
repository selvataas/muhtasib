#!/bin/bash
# GitHub'dan GitLab'a senkronizasyon scripti

echo "🔄 GitHub → GitLab Senkronizasyonu Başlıyor..."

# GitLab bağlantısını test et
echo "📡 GitLab bağlantısı test ediliyor..."
if curl -s --connect-timeout 10 https://gitlabs.mecellem.com > /dev/null; then
    echo "✅ GitLab erişilebilir!"
    
    # GitHub'dan son değişiklikleri çek
    echo "📥 GitHub'dan son değişiklikler çekiliyor..."
    git fetch github
    
    # GitLab'a push et
    echo "📤 GitLab'a push yapılıyor..."
    git push origin master
    
    echo "🎉 Senkronizasyon tamamlandı!"
    echo "📍 GitLab: https://gitlabs.mecellem.com/newmind/mursit/research/muhtasib"
    echo "📍 GitHub: https://github.com/selvataas/muhtesib"
else
    echo "❌ GitLab'a erişim yok!"
    echo "🔧 VPN bağlantısını kontrol edin"
    echo "🔧 Alternatif: GitLab web arayüzünden import yapın"
fi
