# **CharlieTeam - Hotel Bookings**
Felicya Antoni - Patricio Kresnatama

# **Table of Content**
- Introduction
- Problem Statement
    - Problem Statement for Machine Learning
    - Problem Statement for Analytics
- EDA
    - Data Cleaning
    - Feature Engineering
- Insight
- Modeling
- Conclusion and Suggestion

# **Introduction**
Data menjelaskan mengenai 2 dataset dengan hotel demand masing-masing. Kedua hotel berlokasi di Portugal. Hotel
pertama (H1) adalah Resort Hotel yang berlokasi di daerah Algarve sedangkan hotel kedua (H2) adalah City Hotel yang berlokasi di kota Lisbon.\
Kedua dataset memiliki struktur yang sama yaitu 31 variabel yang menggambarkan 40.060 pengamatan Resort Hotel dan 79.330 pengamatan City Hotel. Setiap pengamatan mewakili pemesanan hotel.  
Kedua set data terdiri dari pemesanan hotel antara 1 Juli 2015 dan 31 Agustus 2017. 
Melalui dataset ini diharapkan dapat memberikan insight baru untuk penelitian pendidikan dalam manajemen pendapatan, pembelajaran mesin, dan pengembangan data.

# **Problem Statement**
Berdasarkan dataset yang diperoleh, ada beberapa permasalahan yang akan dipecahkan, yaitu: \
**Problem Statement for Machine Learning** :
1. Bagaimana cara memprediksi apakah customer akan cancel booking atau tidak sehingga kita dapat memikirkan solusi untuk meminimalisir cancel booking?

**Problem Statement for Analytics** :
1. Apa saja variabel yang memengaruhi customer akan membatalkan booking atau tidak?
2. Berapa besar persentase variabel tersebut terhadap keputusan customer membatalkan booking?
3. Customer seperti apa yang harus kita treat agar tidak membatalkan booking?


# **EDA**
Data yang tercatat pada CSV ini merupakan record pemesanan hotel yang sudah terjadi pada 2015 - 2017. Data ini terdiri dari 32 kolom fitur dengan 119.390 record.
### Metadata
1.	Hotel (object) : city (hotel yang bentuknya Gedung, berada di tengah kota) & resort (hotel di lahan luas, pinggir pantai/pegunungan)
2.	Is_canceled (int64) : value booking cancel (1) atau tidak cancel (0)
3.	Lead_time (int64) : Selisih hari antara tanggal booking dengan tanggal check in
4.	Arrival_date_year (int64) : Tahun check in
5.	Arrival_date_month (object) : Bulan check in
6.	Arrival_date_week_number (int64) : Minggu ke berapa (dalam 1 tahun) check in
7.	Arrival_date_of_month (int64) : Hari ke berapa (dalam 1 bulan) check in
8.	Stays_in_weekend_nights (int64) : Jumlah weekend (Sabtu&Minggu) dimana tamu menginap
9.	Stays_in_week_nights (int64) : Jumlah week (Senin-Jumat) dimana tamu menginap
10.	Adults (int64) : Jumlah orang dewasa
11.	Children (float64) : Jumlah anak-anak
12.	Babies (int64) : Jumlah bayi
13.	Meal (object) : Jenis makanan yang dipesan
-	Undefinded / SC : no meal
-	BB : bed & breakfast
-	HB (Half board) : breakfast & 1 other meal usually dinner
-	FB (Full board) : breakfast, lunch, dinner
14.	Country (object) : Negara asal tamu
15.	Market_segment (object) : 
-	Online TA : Travel agent
-	Offline TA/TO : Tour operators
-	Direct
-	Corporate
-	Complementary
-	Groups
-	Undefined
-	Aviation
16.	Distribution_channel (object) : 
-	Direct
-	Corporate
-	TA/TO
-	Undefined
-	GDS
17.	Is_repeated_guest (int64) : Tamu yang sudah pernah menginap sebelumnya di hotel tsb
18.	Previous_cancellations (int64) : Jumlah pembatalan sebelum booking yang sekarang
19.	Previous_booking_not_cancelled (int64) : Jumlah booking sebelumnya yang tidak dicancel sebelum booking yang sekarang
20.	Reserved_room_type (object) : Jenis kamar (sesuai request tamu) yang dibooking
21.	Assigned_room_type (object) : Jenis kamar yang diberikan pihak hotel kepada tamu
22.	Booking_changes (int64) : Jumlah perubahan booking sejak pembuatan bookingan sampai check in/cancel
23.	Deposit_type (object) : Uang jaminan untuk booking hotel
-	No depost
-	Non refund
-	Refundable
24.	Agent (float64) : ID travel agent yang membuat bookingan
25.	Company (float64) : ID Company yang membuat bookingan / bertanggung jawab atas pembayaran hotel tsb
26.	Days_in_waiting_list (int64) : Jumlah hari bookingan tsb dalam waiting list sebelum dikonfirmasi ke tamu
27.	Customer_type (object) : Jenis bookingan
28.	ADR (float64) : average daily rate, jumlah transaksi / jumlah hari menginap
29.	Required_car_parking_space (int64) : Jumlah parkir mobil (request by tamu)
30.	Total_of_special_request (int64) : Jumlah special request oleh tamu
31.	Reservation_status (object) : Status terakhir reservasi
32.	Reservation_status_date (object) : Tanggal dari status terakhir reservasi. Bisa tau kapan bookingan dicancel / tamu check in check out

# **Insight**
Berdasarkan proses understanding data kemudian EDA, ada insight yang bisa didapatkan. 

Kenaikan jumlah pelanggan di city hotel bertepatan dengan diadakannya sebuah event besar yang biasanya diadakan selama 2-3 hari di kota lisbon seperti konser musik, konferensi investasi, festival musik, dll. Hal ini juga terjadi di resort hotel walaupun tidak setinggi city hotel peningkatan jumlah pengunjungnya.

Hal ini didukung dengan beberapa data yang ada seperti tipe customer dan proporsi jumlah adults, children, dan babies yang datang ke hotel. 

1. Tipe customer yang datang adalah transient dengan rata-rata lama menginap sekitar 2-3 hari. 
2. Kebanyakan customer juga tidak membawa anak-anak dan bayi. Hal ini dapat dikarenakan acara yang diadakan di kota lisbon lebih diperuntukkan untuk orang dewasa. 
3. Customer juga kebanyakan hanya memesan kamar hanya dengan sarapan pagi. Hal ini dapat dikarenakan acara yang dilakukan pada siang atau malam hari sehingga pelanggan tidak makan di hotel.

Selain itu, ada juga insight lain seperti kebanyakan pelanggan hotel berasal dari negara Portugal yang berarti pelanggan hotel juga merupakan wisatawan lokal.
Kebanyakan pelanggan tidak memerlukan mobil dapat dikarenakan tempat acara yang dekat dengan hotel atau memang di kota tersebut transportasi umum sudah sangat memadai sehingga pengunjung lebih memilih untuk naik transportasi umum.

# **Modeling**
Model yang digunakan adalah Logistic Regression dan Decision Tree. Model akan memprediksi customer mana yang kemungkinan akan membatalkan bookingan dan tidak. Kedua model ini akan dikomparasi, model mana yang mendapatkan nilai akurasi yang tinggi. 
<br>
Evaluasi yang diharapkan adalah memperkecil nilai False Negatif untuk mengurangi kerugian yang lebih besar kepada pihak hotel. Sehingga model evaluasi yang digunakan adalah Precision.
