import pandas as pd

# 1) Excel dosyasını oku
data = pd.read_excel("default of credit card clients.xls", header=1)

# 2) Veri yapısını incele
print("Satır x Sütun:", data.shape)
print("\nİlk 5 kayıt:")
print(data.head())
print("\nSütun tipleri:")
print(data.dtypes)

# ID sütununu kaldıralım
data = data.drop('ID', axis=1)

# Performans için 5 000 satırlık rastgele örnek alıyoruz
data_small = data.sample(5000, random_state=42)

discrete_cols = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'default payment next month'
]

from ctgan import CTGAN

model = CTGAN(epochs=10)
model.fit(data_small, discrete_columns=discrete_cols)
print("CTGAN ile model eğitildi")

# 5 000 sentetik kayıt üretelim
synthetic_credit = model.sample(5000)

# İlk 5 kaydı görelim
print("Sentetik kredi kartı verisi (ilk 5 kayıt):")
print(synthetic_credit.head())

# İsterseniz CSV'ye de kaydedelim
synthetic_credit.to_csv("synthetic_credit_data.csv", index=False)
print("Sentetik veri CSV olarak kaydedildi: synthetic_credit_data.csv")

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy

# 1) Veriyi yükle
data = pd.read_excel("default of credit card clients.xls", header=1).drop('ID', axis=1)
data_small = data.sample(5000, random_state=42)
synthetic_credit = pd.read_csv("synthetic_credit_data.csv")

# 2) Sütun gruplarını tanımla
continuous_cols = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]
discrete_cols = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'default payment next month'
]

# 3) Özet istatistikleri karşılaştır
real_stats      = data_small.describe().T[['mean','std','min','max']]
synthetic_stats = synthetic_credit.describe().T[['mean','std','min','max']]
comparison = real_stats.join(synthetic_stats, lsuffix='_real', rsuffix='_synth')
print("=== Özet İstatistik Karşılaştırması ===")
print(comparison)

# 4) K–S testi (continuous)
print("\n=== K–S Testi (Continuous) ===")
for col in continuous_cols:
    stat, p_value = ks_2samp(data_small[col], synthetic_credit[col])
    print(f"{col:15s} stat={stat:.4f}, p-value={p_value:.4f}")

# 5) KL Divergence (continuous) — histogram bazlı
print("\n=== KL Divergence (Continuous) ===")
for col in continuous_cols:
    # 10 eşit aralıklı bin, density=True normalize eder
    real_hist, bins = np.histogram(data_small[col], bins=10, density=True)
    syn_hist, _     = np.histogram(synthetic_credit[col], bins=bins, density=True)
    # sıfırları önlemek için küçük eps ekliyoruz
    kl = entropy(real_hist + 1e-8, syn_hist + 1e-8)
    print(f"{col:15s} KL={kl:.4f}")

# 6) Total Variation Distance (discrete)
print("\n=== Total Variation Distance (Discrete) ===")
for col in discrete_cols:
    p_real = data_small[col].value_counts(normalize=True)
    p_syn  = synthetic_credit[col].value_counts(normalize=True)
    idx = p_real.index.union(p_syn.index)
    p_real = p_real.reindex(idx, fill_value=0)
    p_syn  = p_syn.reindex(idx, fill_value=0)
    tvd = 0.5 * np.abs(p_real - p_syn).sum()
    print(f"{col:25s} TVD={tvd:.4f}")




import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# 1) Excel dosyasının tam yolunu verin
file_path = "/Users/uygarsirin/Library/CloudStorage/OneDrive-Personal/Bilgi Computer Engineering/Tez/pycharm/default of credit card clients.xls"

# 2) Veriyi oku ve ID sütununu at
data = pd.read_excel(file_path, header=1).drop('ID', axis=1)

# 3) 5 000 örnek al
data_small = data.sample(5000, random_state=42)

# 4) Metadata oluştur ve otomatik çıkarım yap
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_small)  # :contentReference[oaicite:0]{index=0}

# 5) GaussianCopulaSynthesizer'ı metadata ile oluştur ve eğit
gc_model = GaussianCopulaSynthesizer(metadata)
gc_model.fit(data_small)

# 6) 5 000 kayıt üret, ilk 5’ini yazdır ve CSV’ye kaydet
synthetic_gc = gc_model.sample(5000)
print("Sentetik GC veri (ilk 5 kayıt):")
print(synthetic_gc.head())

synthetic_gc.to_csv("synthetic_credit_gc.csv", index=False)
print("Sentetik GC veri CSV olarak kaydedildi: synthetic_credit_gc.csv")


import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy

# 1) Orijinal ve sentetik GC veriyi yükleyelim
file_path = "/Users/uygarsirin/Library/CloudStorage/OneDrive-Personal/Bilgi Computer Engineering/Tez/pycharm/default of credit card clients.xls"
data = pd.read_excel(file_path, header=1).drop('ID', axis=1)
data_small = data.sample(5000, random_state=42)
synthetic_gc = pd.read_csv("synthetic_credit_gc.csv")

# 2) Sürekli ve kesikli sütunları tanımlayalım
continuous_cols = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]
discrete_cols = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'default payment next month'
]

# 3) Özet istatistik karşılaştırması
real_stats      = data_small.describe().T[['mean','std','min','max']]
gc_stats        = synthetic_gc.describe().T[['mean','std','min','max']]
comparison_gc   = real_stats.join(gc_stats, lsuffix='_real', rsuffix='_gc')
print("=== Özet İstatistik Karşılaştırması (GC) ===")
print(comparison_gc)

# 4) K–S testi (continuous)
print("\n=== K–S Testi (Continuous, GC) ===")
for col in continuous_cols:
    stat, p = ks_2samp(data_small[col], synthetic_gc[col])
    print(f"{col:15s} stat={stat:.4f}, p-value={p:.4f}")

# 5) KL Divergence (continuous)
print("\n=== KL Divergence (Continuous, GC) ===")
for col in continuous_cols:
    real_hist, bins = np.histogram(data_small[col], bins=10, density=True)
    gc_hist, _      = np.histogram(synthetic_gc[col], bins=bins, density=True)
    kl = entropy(real_hist + 1e-8, gc_hist + 1e-8)
    print(f"{col:15s} KL={kl:.4f}")

# 6) TVD (discrete)
print("\n=== Total Variation Distance (Discrete, GC) ===")
for col in discrete_cols:
    p_real = data_small[col].value_counts(normalize=True)
    p_gc   = synthetic_gc[col].value_counts(normalize=True)
    idx = p_real.index.union(p_gc.index)
    p_real = p_real.reindex(idx, fill_value=0)
    p_gc   = p_gc.reindex(idx, fill_value=0)
    tvd = 0.5 * np.abs(p_real - p_gc).sum()
    print(f"{col:25s} TVD={tvd:.4f}")
