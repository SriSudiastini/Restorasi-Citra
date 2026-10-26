# # BAB IV
# HASIL DAN PEMBAHASAN

## 4.1 Gambaran Umum Pengujian

Pada eksperimen ini dilakukan penerapan pendekatan BIR-Adapter sederhana untuk restorasi citra CCTV yang mengalami motion blur. Tujuan pengujian adalah mengetahui apakah penggunaan Restoring Attention pada model diffusion dapat membantu memperbaiki kualitas citra yang mengalami penurunan ketajaman akibat efek gerakan.

Implementasi dilakukan menggunakan Google Colab dengan dukungan GPU NVIDIA Tesla T4. Model utama yang digunakan adalah Stable Diffusion 1.5, khususnya komponen Variational Autoencoder (VAE) dan U-Net sebagai backbone diffusion. Untuk mengurangi kebutuhan komputasi, eksperimen dibatasi pada penggunaan satu Restoring Attention dari 16 self-attention layer yang ditemukan pada U-Net.

Citra yang digunakan berupa satu gambar CCTV asli dan satu gambar CCTV yang telah diberikan efek motion blur secara sintetis. Kedua gambar diproses pada resolusi 256 × 256 piksel sebelum digunakan dalam proses training dan evaluasi.

Implementasi ini merupakan eksperimen sederhana yang mengadaptasi konsep Restoring Attention dari BIR-Adapter, bukan reproduksi lengkap seluruh arsitektur dan konfigurasi penelitian aslinya.

## 4.2 Implementasi Restoring Attention

Tahap awal implementasi dilakukan dengan mengidentifikasi self-attention layer pada U-Net Stable Diffusion 1.5. Berdasarkan pemeriksaan struktur model, ditemukan 16 self-attention layer yang dapat menjadi lokasi integrasi adapter.

Dalam eksperimen ini, hanya satu layer yang digunakan agar proses pengembangan, pengujian, dan training dapat dilakukan dengan sumber daya komputasi yang terbatas.

Restoring Attention dikembangkan dengan memanfaatkan dua jalur fitur, yaitu degraded branch dan diffusion branch. Degraded branch menerima informasi dari citra CCTV yang mengalami motion blur, sedangkan diffusion branch memproses latent pada tahapan diffusion.

Query diperoleh melalui proyeksi fitur degraded menggunakan bobot Query U-Net. Sementara itu, Key dan Value dihitung melalui proyeksi adapter terhadap fitur diffusion. Output attention digunakan untuk menghasilkan koreksi terhadap fitur yang diproses pada U-Net.

Pengujian awal menunjukkan bahwa adapter berhasil diintegrasikan dalam proses forward U-Net. Pada resolusi latent 64 × 64, kedua jalur menghasilkan fitur berukuran `[1, 4096, 320]`. Pengujian juga menunjukkan bahwa output adapter memiliki nilai numerik yang valid dan dapat digabungkan dengan output self-attention.

Selain itu, hasil pengujian backpropagation memperlihatkan bahwa parameter Key, Value, dan Output Adapter dapat menerima gradien. Temuan ini menunjukkan bahwa mekanisme pembaruan parameter pada adapter telah berjalan.

## 4.3 Hasil Training BIR-Adapter Sederhana

Training dilakukan menggunakan satu pasangan citra CCTV asli dan citra motion blur. Kedua gambar dikonversi menjadi representasi latent menggunakan VAE.

Selama training, parameter U-Net dibekukan sehingga pembaruan bobot hanya dilakukan pada modul Restoring Attention. Proses training menggunakan optimizer AdamW dengan learning rate 0,00001 dan weight decay 0,01. Training dilakukan sebanyak 20 langkah menggunakan empty prompt.

**Tabel 4.1 Hasil Training Restoring Attention**

| Training Step | Loss | Gradient |
|---|---:|---:|
| 1 | 0,377895 | 0,407524 |
| 5 | 0,487711 | 0,295718 |
| 10 | 0,560003 | 0,270696 |
| 15 | 0,537829 | 0,276137 |
| 20 | 0,791780 | 0,236951 |

Berdasarkan Tabel 4.1, nilai loss selama proses training mengalami fluktuasi. Kondisi tersebut berkaitan dengan penggunaan timestep dan noise yang dipilih secara acak pada setiap langkah. Oleh karena itu, peningkatan atau penurunan loss pada satu langkah belum cukup untuk menyimpulkan keberhasilan training secara keseluruhan.

Meskipun loss tidak mengalami penurunan secara konsisten, nilai gradien menunjukkan bahwa proses pembelajaran pada parameter adapter tetap berlangsung. Seluruh proses training berhasil diselesaikan tanpa kesalahan komputasi dan bobot adapter berhasil disimpan.

## 4.4 Evaluasi Noise Prediction

Setelah training, dilakukan evaluasi untuk membandingkan kemampuan prediksi noise antara U-Net tanpa adapter dan U-Net dengan Restoring Attention satu layer.

Pengujian menggunakan 10 variasi noise dan timestep. Kedua model menerima input yang sama pada setiap pengujian agar perbandingan lebih terkontrol.

**Tabel 4.2 Hasil Evaluasi Noise Prediction**

| Komponen | Hasil |
|---|---:|
| Rata-rata loss tanpa adapter | 0,187704 |
| Rata-rata loss dengan adapter | 0,186590 |
| Penurunan rata-rata loss | 0,5932% |
| Jumlah pengujian dengan loss lebih rendah | 9 dari 10 |

Berdasarkan Tabel 4.2, model dengan Restoring Attention menghasilkan rata-rata loss yang lebih rendah dibandingkan U-Net tanpa adapter. Dari 10 pengujian, sebanyak 9 pengujian menunjukkan penurunan loss setelah menggunakan adapter.

Hasil tersebut memberikan indikasi awal bahwa Restoring Attention dapat memengaruhi prediksi noise pada model diffusion. Namun, peningkatannya masih relatif kecil. Selain itu, evaluasi masih menggunakan gambar yang sama dengan data training, sehingga hasil tersebut belum dapat digunakan untuk menyimpulkan kemampuan generalisasi model terhadap gambar CCTV lainnya.

## 4.5 Hasil Restorasi Citra CCTV

Setelah tahap training dan evaluasi noise prediction, dilakukan pengujian inferensi untuk menghasilkan citra restorasi. Inferensi menggunakan DDIM sampling dengan citra CCTV blur sebagai input.

Pada percobaan awal, proses diffusion menghasilkan citra dengan struktur objek yang mengalami perubahan tidak wajar. Untuk mengurangi kerusakan struktur, dilakukan penyesuaian konfigurasi DDIM dan penurunan strength menjadi 0,10.

Setelah penyesuaian, struktur jalan, bangunan, dan objek pada citra CCTV dapat dipertahankan dengan lebih baik. Akan tetapi, hasil restorasi masih terlihat buram dan belum menunjukkan peningkatan ketajaman secara signifikan.

Selanjutnya, dilakukan pengujian dengan menambahkan Sobel Guidance serta membandingkan beberapa referensi restorasi awal.

**Tabel 4.3 Perbandingan Kualitas Citra Menggunakan PSNR dan SSIM**

| Metode | PSNR (dB) | SSIM |
|---|---:|---:|
| CCTV Blur (Input) | **23,2515** | **0,6579** |
| DDIM + Adapter | 21,8057 | 0,5814 |
| DDIM + Adapter + Sobel Guidance | 22,4533 | 0,6153 |
| DDIM + Adapter + Best Blend Guidance | 22,4090 | 0,6130 |

Berdasarkan Tabel 4.3, hasil DDIM dengan Restoring Attention menghasilkan PSNR sebesar 21,8057 dB dan SSIM sebesar 0,5814. Nilai tersebut lebih rendah dibandingkan citra CCTV blur awal yang memiliki PSNR sebesar 23,2515 dB dan SSIM sebesar 0,6579.

Penambahan Sobel Guidance meningkatkan kualitas hasil dibandingkan DDIM tanpa guidance. Nilai PSNR meningkat menjadi 22,4533 dB dan SSIM menjadi 0,6153. Namun, kedua nilai tersebut masih lebih rendah daripada kualitas citra blur awal berdasarkan gambar referensi.

Dengan demikian, penggunaan guidance memberikan manfaat dalam mempertahankan struktur citra, tetapi belum berhasil mengembalikan detail yang hilang akibat motion blur.

## 4.6 Pengujian Initial Restoration

Untuk mendukung proses guidance, dilakukan pengujian metode Richardson–Lucy sebagai restorasi awal. Metode ini digunakan untuk mencoba mengurangi efek motion blur sebelum hasilnya dijadikan referensi tambahan bagi proses diffusion.

Pengujian dilakukan dengan beberapa jumlah iterasi, yaitu 3, 5, 10, dan 15. Berdasarkan hasil pengukuran, iterasi ke-3 menghasilkan nilai terbaik di antara kandidat Richardson–Lucy yang diuji, dengan PSNR sebesar 20,6268 dB dan SSIM sebesar 0,5387.

Namun, hasil tersebut masih lebih rendah dibandingkan citra blur awal. Penambahan iterasi juga menghasilkan artefak visual dan penurunan kualitas berdasarkan metrik pengujian.

Selanjutnya, dilakukan penggabungan citra blur dengan hasil Richardson–Lucy melalui metode blending untuk mengurangi perubahan struktur yang terlalu besar.

**Tabel 4.4 Hasil Pengujian Blended Initial Restoration**

| Nilai Alpha | PSNR (dB) | SSIM |
|---|---:|---:|
| 0,1 | **23,1307** | **0,6513** |
| 0,2 | 22,9698 | 0,6431 |
| 0,3 | 22,7634 | 0,6331 |
| 0,4 | 22,5220 | 0,6218 |

Berdasarkan Tabel 4.4, nilai alpha 0,1 menghasilkan hasil terbaik dari empat kandidat blending. Akan tetapi, hasilnya masih belum melampaui kualitas citra blur awal.

Penggunaan hasil blending tersebut sebagai referensi guidance pada DDIM juga belum menghasilkan peningkatan kualitas yang lebih baik dibandingkan penggunaan citra blur sebagai referensi.

Hal ini menunjukkan bahwa kualitas citra yang digunakan sebagai referensi guidance turut memengaruhi hasil restorasi. Referensi yang masih mengalami blur atau memiliki artefak belum mampu memberikan informasi detail yang cukup untuk meningkatkan hasil akhir.

## 4.7 Pembahasan Hasil Eksperimen

Berdasarkan keseluruhan pengujian, implementasi Restoring Attention satu layer telah berhasil dijalankan pada U-Net Stable Diffusion 1.5. Hal tersebut dibuktikan melalui keberhasilan proses forward, penerimaan gradien, training, serta perubahan nilai prediksi noise ketika adapter digunakan.

Meskipun evaluasi noise prediction menunjukkan penurunan rata-rata loss sebesar 0,5932%, peningkatan tersebut belum diikuti oleh perbaikan kualitas citra berdasarkan PSNR dan SSIM.

Perbedaan ini menunjukkan bahwa penurunan noise-prediction loss tidak secara langsung menjamin peningkatan kualitas restorasi citra. Model dapat menghasilkan prediksi noise yang sedikit lebih baik pada kondisi pengujian tertentu, tetapi proses sampling tetap dapat mengubah struktur maupun detail citra.

Beberapa faktor yang diduga memengaruhi hasil eksperimen adalah jumlah data training yang sangat terbatas, penggunaan hanya satu Restoring Attention layer, jumlah langkah training yang masih sedikit, serta mekanisme guided sampling yang belum sepenuhnya mengikuti metode penelitian BIR-Adapter.

Selain itu, pendekatan Richardson–Lucy menggunakan estimasi kernel blur tertentu, sementara karakteristik blur pada citra yang telah diproses dapat dipengaruhi oleh perubahan resolusi dan proses interpolasi. Hal ini berpotensi menyebabkan restorasi awal menghasilkan artefak.

Oleh karena itu, hasil eksperimen ini lebih tepat dipandang sebagai pengujian awal kelayakan integrasi Restoring Attention pada proses diffusion, bukan sebagai pembuktian keberhasilan metode BIR-Adapter untuk menghilangkan motion blur CCTV secara efektif.

## 4.8 Keterbatasan dan Rencana Pengembangan

Eksperimen ini memiliki beberapa keterbatasan. Pertama, dataset hanya terdiri dari satu pasangan citra sehingga belum cukup untuk melatih model yang dapat bekerja pada berbagai kondisi CCTV. Kedua, eksperimen hanya menggunakan satu dari 16 self-attention layer U-Net. Ketiga, durasi training masih terbatas pada 20 langkah.

Selain itu, metode guidance yang diterapkan merupakan penyederhanaan dan belum mereproduksi seluruh prosedur inferensi pada penelitian asli. Pemilihan beberapa konfigurasi restorasi awal juga dilakukan berdasarkan metrik terhadap gambar referensi yang sama, sehingga masih diperlukan pemisahan data pengembangan dan data pengujian.

Untuk pengembangan selanjutnya, disarankan menggunakan dataset CCTV yang lebih banyak dan beragam, melakukan variasi motion blur, memperpanjang proses training, serta membandingkan konfigurasi satu, dua, dan empat layer Restoring Attention. Mekanisme guided sampling juga perlu disesuaikan lebih lanjut dengan arsitektur BIR-Adapter yang dijadikan acuan.

Evaluasi selanjutnya sebaiknya dilakukan pada gambar yang tidak digunakan selama proses training maupun pemilihan parameter sehingga hasil pengujian dapat memberikan gambaran kemampuan restorasi yang lebih objektif.

---

# BAB V
# PENUTUP

## 5.1 Kesimpulan

Berdasarkan eksperimen yang telah dilakukan, dapat disimpulkan bahwa implementasi sederhana Restoring Attention yang mengadaptasi konsep BIR-Adapter berhasil diintegrasikan ke dalam U-Net Stable Diffusion 1.5 menggunakan satu self-attention layer. Proses training dapat dijalankan dan seluruh parameter adapter mampu menerima gradien.

Hasil evaluasi noise prediction menunjukkan penurunan rata-rata loss sebesar 0,5932%, dengan 9 dari 10 pengujian menghasilkan nilai loss yang lebih rendah setelah menggunakan adapter. Namun, hasil tersebut belum menunjukkan peningkatan kualitas restorasi citra CCTV.

Berdasarkan evaluasi PSNR dan SSIM, citra CCTV blur awal masih menghasilkan nilai terbaik, yaitu PSNR 23,2515 dB dan SSIM 0,6579. Metode DDIM dengan Restoring Attention dan Sobel Guidance menghasilkan kualitas yang lebih baik dibandingkan DDIM tanpa guidance, tetapi belum mampu melampaui kualitas citra input.

Dengan demikian, eksperimen ini berhasil menunjukkan kelayakan teknis integrasi Restoring Attention pada proses diffusion, tetapi belum berhasil membuktikan efektivitasnya dalam menghilangkan motion blur pada citra CCTV.

## 5.2 Saran

Untuk meningkatkan kualitas hasil pada penelitian selanjutnya, diperlukan penggunaan dataset yang lebih besar, variasi kondisi motion blur, dan proses training yang lebih panjang. Selain itu, jumlah Restoring Attention layer dapat ditingkatkan secara bertahap untuk mengamati pengaruhnya terhadap kualitas hasil restorasi.

Pengembangan juga dapat difokuskan pada penggunaan model restorasi awal yang lebih baik serta implementasi mekanisme guided sampling yang lebih mendekati metode BIR-Adapter asli.

Dengan pengembangan tersebut, diharapkan model mampu menghasilkan citra CCTV yang lebih jelas, mempertahankan struktur objek, dan mencapai peningkatan nilai PSNR serta SSIM dibandingkan citra blur awal.
