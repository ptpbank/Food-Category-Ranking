"""Finding Top Food Type"""

import GetOldTweets3 as got
import re
import string
import nltk

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from nltk.stem.porter import PorterStemmer

import pandas as pd
from stop_words import get_stop_words
from collections import Counter
import matplotlib.pyplot as plt

text_query = 'อยาก กิน หิว'
since_date = '2020-05-01'
until_date = '2020-05-31'
count = 5000
# สร้าง query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setSince(since_date).setUntil(
    until_date).setMaxTweets(count)
# สร้าง tweet ที่เข้า criteria
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
# สร้าง list ที่ได้จากการดึงทวิตมา
text_tweets = [tweet.text for tweet in tweets]

# คลีน message
def clean_msg(msg):
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>', '', msg)

    # ลบ hashtag
    msg = re.sub(r'#', '', msg)
    # ลบ ฿
    msg = re.sub(r'฿', '', msg)
    # ลบ -
    msg = re.sub(r'-', '', msg)
    # ลบ -
    msg = re.sub(r'—', '', msg)

    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c), '', msg)

    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())

    return msg


clean_text = [clean_msg(txt) for txt in text_tweets]

# โหลดตัว stop word  ทั้งไทยและอังกฤษ
nltk.download('words')
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()


# ตัดคำ
def split_word(text):
    # ตัดคำโดยใช้ dict ใน corpus ที่ผม edit ไป มันจะตัดเฉพาะเมนูอาหารที่ผมใส่ไปใน words.th.txt
    tokens = word_tokenize(text, engine='dict')

    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]

    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]

    # Thai
    tokens_temp = []
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn) > 0) and (len(w_syn[0].lemma_names('tha')) > 0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)

    tokens = tokens_temp

    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]

    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens


# สร้าง list มาเก็บ list ของ list คำที่ตัดมา
total_food = []
for i in text_tweets:
    total_food.append(split_word(clean_msg(i)))

# สร้าง list มาเก็บเฉพาะคำ
total_word = []
for i in total_food:
    for j in i:
        total_word.append(j)

print(total_word)

# สร้าง list ของเมนูอาหารแยกเป็นชนิดๆ ต้ม ผัด แกง ทอด ย่าง นึ่ง+อบ ยำ+ตำ อาหารทั่วไป อาหารต่างประเทศ อาหารหวาน

# list menmu อาหารประเภทต้ม
boiled_data = ['ต้ม', 'ต้มยำกุ้ง', 'ต้มยำรวมมิตร', 'ต้มแซบกระดูกอ่อน', 'ต้มแซบกระดูกหมู',
               'ต้มเค็ม', 'ต้มยำไก่', 'ต้มข่าไก่', 'ต้มส้ม', 'ต้มแซบกระดูกอ่อน', 'ต้มแซบ', 'ต้มยำ', 'ต้ำยำน้ำข้น',
               'ต้มยำปลากระป๋อง', 'กระเพราะปลา', 'หมูมะนาว', 'คะหน้าหมูกรอบ', 'น้ำพริก', 'ปลาร้าทรงเครื่อง', 'มาม่า',
               'พะโล้', 'สุกี้', 'ชาบู', 'หมูตุ๋น', 'โจ๊ก', 'ขาหมู']

# list menu อาหารประเภทผัด
puff_data = ['ผัด', 'ผัดไท', 'ผัดไทกุ้งสด', 'ข้าวผัดหมู', 'ข้าวผัด', 'ข้าวผัดกุ้ง', 'ข้าวผัดไก่', 'ข้าวผัดไส้กรอก',
             'ข้าวผัดกุนเชียง',
             'ข้าวผัดรถไฟ', 'ข้าวผัดทะเล', 'ข้าวผัดปู', 'เนื้อผัดน้ำมันหอย', 'เนื้อน้ำมันหอย', 'ผัดพริกแกง',
             'ผัดพริกแกงหมู',
             'ผัดพริกแกงหมูกรอบ', 'ผัดพริกแกงไก่', 'ผัดพริกแกงทะเล', 'ผัดพริกแกงปลาหมึก', 'กุ้งผัดพริกแกง', 'ไข่ลูกเขย',
             'ฉู่ฉี่', 'ผัดไททะเล', 'ผัดฉ่า', 'ผัดฉ่ารวมมิตร', 'ผัดขี้เมา', 'ผัดขี้เมาทะเล', 'หมูผัดปลาร้า',
             'หมูปลาร้า',
             'กะเพรา', 'กะเพราหมูกอรบ', 'กะเพราไก่', 'กะเพราทะเล', 'กะเพรารวมมิตร', 'กะเพราปลาหมึก', 'กะเพราะหมูสับ'
             ]

# list menu อาหารประเภทแกง
curry_data = ['แกง', 'แกงจืด', 'แกงกะหรี่', 'แกงจืดเต้าหู้', 'แกงจืดเต้าหู้หมูสับ', 'แกงจืดเต้าหู้ไข่สาหร่าย',
              'แกงจืดแตงกวา',
              'แกงจืดวุ้นเส้น', 'แกงจืดผักกาด', 'แกงจืดผักกาดดอง', 'แกงจืดมะระ', 'แกงจืดมะระหมูสับ',
              'แกงจืดวุ้นเส้นหมูสับ',
              'แกงเขียวหวาน', 'แกงส้ม', 'แกงเลียง', 'แกงเผ็ด', 'แกงรันจวญ', 'แกงไตปลา', 'แกงเทโพ',
              'แกงป่า', 'แกงเหลือง', 'แกงฟัก', 'แกงแดง', 'แกงคั่ว']

# list menu อาหารประเภททอด
fried_data = ['ทอด', 'ไข่เจียว', 'ไข่ดาว', 'หมูกรอบ', 'หมูทอด', 'ไก่ทอด', 'ปลาทอด', 'ปลาส้มทอด', 'ทอดมัน', 'กากหมู',
              'แคปหมู', 'แค็ปหมู', 'ลูกชิ้นทอด', 'ไส้กรอกทอด', 'หมูยอทอด', 'หอมเจียว', 'หนังไก่ทอด', 'ข้อไก่ทอด',
              'เอ็นไก่ทอด', 'คอหมูทอด', 'ปลากระพงทอด', 'ปลาเก๋าทอด', 'กุ้งทอดกระเทียม', 'หมูทอดกระเทียม',
              'หมูทอดน้ำปลา',
              'ไก่ทอดกระเทียม', 'ไก่กระเทียม', 'หมูกระเทียม', 'กุ้งกระเทียม']

# list menu อาหารประเภทต้ม
steam_data = ['ปลานึ่ง', 'ลูกชิ้นนึ่ง', 'หมูอบ', 'หมูนึ่งมะนาว', 'ปลานึ่งมะนาว', 'ปลาหมึกนึ่งมะนาว', 'ไก่อบ', 'ไข่ตุ๋น']

# list menu อาหารประเภทยำ+ตำ
yum_data = ['ยำ', 'ยำหมูยอ', 'ยำหมูยอไข่แดงเค็ม', 'ยำไหลบัว', 'ยำหมูยอกุ้งสด', 'ยำกุ้งสด', 'ยำไส้กรอก',
            'ยำลูกชิ้น', 'ยำปลาร้า', 'ยำแซลมอน', 'ยำวุ้นเส้น', 'ยำมาม่า', 'ยำสาหร่าย', 'ยำทะเล', 'ยำรวมมิตร', 'ส้มตำ',
            'ตำป่า', 'ตำไทย', 'ตำลาว', 'ตำโคราช', 'ตำแตง', 'ตำหมูยอ', 'ตำถาด', 'ตำไทยไข่เค็ม']

# list menu อาหารประเภทย่าง
grill_data = ['หมูกระทะ', 'บาบีก้อน', 'คอหมูย่าง', 'ไก่ย่าง', 'หมูย่าง',
              'เนื้อย่าง', 'ปลาย่าง', 'ปลาดุกย่าง', 'เสือร้องไห้', 'หม่าล่า']

# list menu อาหารประเภททั่วไป
general_data = ['ก๋วยจั๊บ', 'บะหมี่', 'ข้าวคลุกกะปิ', 'ขนมจีน', 'ขนมจีนน้ำยา', 'ข้าวมันไก่', 'ข้าวหมูแดง',
                'ข้าวขาหมู', 'ก๋วยเตี๋ยว', 'เกี๊ยวกุ้ง', 'เกี๊ยวหมู', 'เกี๊ยวปลา', 'ข้าวต้ม', 'น้ำตกหมู', 'น้ำตกเนื้อ',
                'น้ำพริก', 'น้ำพริกกากหมู', 'น้ำพริกกะปิ', 'เกี๊ยว']

# list menu อาหารประเภทอาหารต่างประเทศ
foreign_data = ['ซูชิ', 'ซาซิมิ', 'แซลมอน', 'วากิว', 'ทงคัตสึ', 'ทงคตสึ', 'เท็มปุระ', 'เนื้อวากิว', 'อุด้ง', 'ราเมง',
                'อูนิ', 'อิกุระ', 'มักกะโรนี', 'สปาเก็ตตี้', 'สลัด', 'พาสต้า', 'พิซซ่า', 'ทาโก้', 'ฟัวกราส์', 'ฟัวกรา',
                'รามยอน', 'ขนมจีบ',
                'จีบกุ้ง', 'ขนมจีบกุ้ง', 'ขนมจีบหมู', 'ซาลาเปา', 'เปาหมูสับ', 'เปาหมูแดง']

# list menu อาหารประเภทอาหารหวาน
dessert_data = ['เชื่อม', 'บวชชี', 'กะทิ', 'ลอดช่อง', 'แกงบวช', 'เฉาก๊วย', 'ลูกชุบ', 'ข้าวเหนียวมะม่วง',
                'ข้าวเหนียวทุเรียน',
                'ตะโก้', 'วุ้นมะพร้าว', 'หวานเย็น', 'บิงซู', 'เค้ก', 'ไอติม', 'ไอศกรีม', 'พาย', 'ช็อคโกแลต', 'ทาร์ตไข่',
                'มะม่วงน้ำปลาหวาน', 'โรตี', 'ขนมเบื้อง', 'ขนม']

boiled_food = []
puff_food = []
curry_food = []
fried_food = []
steam_food = []
yum_food = []
grill_food = []
general_food = []
foreign_food = []
dessert_food = []
other = []

for food in total_word:
    if food in boiled_data:
        boiled_food.append(food)
    elif food in puff_data:
        puff_food.append(food)
    elif food in curry_data:
        curry_food.append(food)
    elif food in fried_data:
        fried_food.append(food)
    elif food in steam_data:
        steam_food.append(food)
    elif food in yum_data:
        yum_food.append(food)
    elif food in grill_data:
        grill_food.append(food)
    elif food in general_data:
        general_food.append(food)
    elif food in foreign_data:
        foreign_food.append(food)
    elif food in dessert_data:
        dessert_food.append(food)
    else:
        other.append(food)

print(boiled_food)
print(Counter(boiled_food))

print("Count of boiled food is: %d", len(boiled_food))
print("Count of puff food is: %d", len(puff_food))
print("Count of curry food is: %d", len(curry_food))
print("Count of fried food is: %d", len(fried_food))
print("Count of steam food is: %d", len(steam_food))
print("Count of yum food is: %d", len(yum_food))
print("Count of grill food is: %d", len(grill_food))
print("Count of general food is: %d", len(general_food))
print("Count of foreign food is: %d", len(foreign_food))
print("Count of dessert food is: %d", len(dessert_food))

food_cat = {'boiled': len(boiled_food), 'puff': len(puff_food),
            'curry': len(curry_food), 'fried': len(fried_food),
            'steam': len(steam_food), 'yim': len(yum_food),
            'grill': len(grill_food), 'general': len(general_food),
            'foreign': len(foreign_food), 'dessert': len(dessert_food)}

print(Counter(food_cat))

food_cat_df = pd.DataFrame(Counter(food_cat).most_common(10), columns=['food_cat', 'count'])
print(food_cat_df.head(10))

fig, ax = plt.subplots(figsize=(8, 8))

# Plot กราฟเป็นแบบ horizontal bar ของ food category
food_cat_df.sort_values(by='count').plot.barh(x='food_cat',
                                              y='count',
                                              ax=ax,
                                              color="green")
ax.set_title("Top Food Category")

for tick in ax.get_xticklabels():
    tick.set_fontname("Sarabun")

for tick in ax.get_yticklabels():
    tick.set_fontname("Sarabun")

plt.show()

dessert_df = pd.DataFrame(Counter(dessert_food).most_common(5), columns=['dessert_food', 'dessert_count'])
print(dessert_df.head(5))

fig, ax = plt.subplots(figsize=(8, 8))

# Plot กราฟเป็นแบบ horizontal bar ของ food category
dessert_df.sort_values(by='dessert_count').plot.barh(x='dessert_food',
                                              y='dessert_count',
                                              ax=ax,
                                              color="red")
ax.set_title("Top Dessert Menu")

for tick in ax.get_xticklabels():
    tick.set_fontname("Sarabun")

for tick in ax.get_yticklabels():
    tick.set_fontname("Sarabun")

plt.show()

# ลองแสดงเมนูอาหารของอาหารประเภทต้ม ว่ามีอะไรบ้าง
boiled_df = pd.DataFrame(Counter(boiled_food).most_common(5), columns=['boiled_food', 'boiled_count'])
print(boiled_df.head(5))

fig, ax = plt.subplots(figsize=(8, 8))

# Plot กราฟเป็นแบบ horizontal bar
boiled_df.sort_values(by='boiled_count').plot.barh(x='boiled_food',
                                              y='boiled_count',
                                              ax=ax,
                                              color="blue")
ax.set_title("Top Boiled Food")

for tick in ax.get_xticklabels():
    tick.set_fontname("Sarabun")

for tick in ax.get_yticklabels():
    tick.set_fontname("Sarabun")

plt.show()


# ลองแสดงเมนูอาหารของอาหารประเภททอด ว่ามีอะไรบ้าง
fried_df = pd.DataFrame(Counter(fried_food).most_common(5), columns=['fried_food', 'fried_count'])
print(fried_df.head(5))

fig, ax = plt.subplots(figsize=(8, 8))

# Plot กราฟเป็นแบบ horizontal bar ของเมนูทอด
fried_df.sort_values(by='fried_count').plot.barh(x='fried_food',
                                              y='fried_count',
                                              ax=ax,
                                              color="purple")
ax.set_title("Top Fried Food")

for tick in ax.get_xticklabels():
    tick.set_fontname("Sarabun")

for tick in ax.get_yticklabels():
    tick.set_fontname("Sarabun")

plt.show()