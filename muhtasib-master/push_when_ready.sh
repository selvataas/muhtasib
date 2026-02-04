#!/bin/bash
# VPN bağlantısı kurulduktan sonra çalıştırın

echo "GitLab bağlantısı test ediliyor..."
if curl -s --connect-timeout 5 https://gitlabs.mecellem.com > /dev/null; then
    echo "✅ GitLab erişilebilir, push yapılıyor..."
    git push -u origin master
    echo "✅ Push tamamlandı!"
else
    echo "❌ GitLab'a hala erişim yok. VPN kontrolü yapın."
fi
