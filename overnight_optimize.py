#!/usr/bin/env python3
"""
Overnight Autonomous Prompt Optimizer for VoiceInk
===================================================
Runs for ~6 hours, testing prompt × temperature combinations.
Saves results after EVERY round (crash-safe).
Does NOT modify production code — only finds the best config.

Strategy:
  Phase 1 (Rounds 1-40):  Test 40 prompt variations at temp=0.3
  Phase 2 (Rounds 41-50): Test top 5 prompts × 10 temperatures
  Phase 3 (Rounds 51+):   Generate hybrid prompts from top performers, repeat

Output: overnight_results.json (updated after every round)
"""

import json
import os
import re
import time
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "overnight_results.json"

# ══════════════════════════════════════════════════════════════════
# TEST CASES — 160 cases, real user inputs, bilingual, complex
# ══════════════════════════════════════════════════════════════════

TESTS = [
    # ─── Chinese Numbers (12) ───
    ("百分之三十二", "32%", "zh_num"),
    ("百分之九十九点九", "99.9%", "zh_num"),
    ("三百七十六个人", "376个人", "zh_num"),
    ("一千零二十四", "1024", "zh_num"),
    ("九千四百八十二", "9482", "zh_num"),
    ("两万五千三百", "25300", "zh_num"),
    ("零点五", "0.5", "zh_num"),
    ("三点一四一五九", "3.14159", "zh_num"),
    ("十二块五毛", "12.5元", "zh_num"),
    ("三百二十万美元", "320万美元", "zh_num"),
    ("第三个选项", "第3个选项", "zh_num"),
    ("大概三十二秒左右", "大概32秒左右", "zh_num"),

    # ─── Chinese Dates/Times — user prefers AM/PM, month/day/year (8) ───
    ("二零二六年四月三号", "2026年4月3号", "zh_date"),
    ("二零二五年十二月三十一日", "2025年12月31日", "zh_date"),
    ("下午两点三十分", "下午2:30", "zh_date"),
    ("凌晨两点三十七分", "凌晨2:37", "zh_date"),
    ("三个小时二十分钟", "3小时20分钟", "zh_date"),
    ("早上九点十五分开会", "早上9:15开会", "zh_date"),
    ("还有两分钟就结束了", "还有2分钟就结束了", "zh_date"),
    ("六月十五号星期天", "6月15号星期天", "zh_date"),

    # ─── Chinese Math/Symbols (12) ───
    ("三加五等于八", "3+5=8", "zh_math"),
    ("十减三等于七", "10-3=7", "zh_math"),
    ("六乘以七等于四十二", "6×7=42", "zh_math"),
    ("二十除以四等于五", "20÷4=5", "zh_math"),
    ("X大于零", "X>0", "zh_math"),
    ("Y小于一百", "Y<100", "zh_math"),
    ("A大于等于B", "A≥B", "zh_math"),
    ("A不等于B", "A≠B", "zh_math"),
    ("三点一四乘以R的平方", "3.14×R²", "zh_math"),
    ("X的平方加Y的平方等于Z的平方", "X²+Y²=Z²", "zh_math"),
    ("根号二约等于一点四一四", "√2≈1.414", "zh_math"),
    ("百分之三十二大于百分之二十五", "32%>25%", "zh_math"),

    # ─── Chinese Idioms — must NOT convert (8) ───
    ("三心二意", "三心二意", "zh_idiom"),
    ("不管三七二十一", "不管三七二十一", "zh_idiom"),
    ("七上八下", "七上八下", "zh_idiom"),
    ("五花八门", "五花八门", "zh_idiom"),
    ("一模一样", "一模一样", "zh_idiom"),
    ("九牛一毛", "九牛一毛", "zh_idiom"),
    ("四面楚歌", "四面楚歌", "zh_idiom"),
    ("百发百中", "百发百中", "zh_idiom"),

    # ─── Chinese Common Words — must NOT convert (8) ───
    ("一些东西", "一些东西", "zh_keep"),
    ("一下子", "一下子", "zh_keep"),
    ("一直在做", "一直在做", "zh_keep"),
    ("我有一个想法", "我有一个想法", "zh_keep"),
    ("一起去吧", "一起去吧", "zh_keep"),
    ("一定要做好", "一定要做好", "zh_keep"),
    ("一般来说", "一般来说", "zh_keep"),
    ("一会儿再说", "一会儿再说", "zh_keep"),

    # ─── Chinese Fillers (8) ───
    ("呃就是说这个东西嗯还不错", "这个东西还不错", "zh_filler"),
    ("嗯然后呢我们看看怎么办", "我们看看怎么办", "zh_filler"),
    ("呃我觉得嗯这个方案可以", "我觉得这个方案可以", "zh_filler"),
    ("就是说呃你帮我看看这个", "你帮我看看这个", "zh_filler"),
    ("然后呢嗯对就是这样", "对就是这样", "zh_filler"),
    ("呃呃呃我想说的是", "我想说的是", "zh_filler"),
    ("那个嗯你懂的", "你懂的", "zh_filler"),
    ("就是说嗯可以的", "可以的", "zh_filler"),

    # ─── Chinese Tone Preservation (5) ───
    ("可不可以帮我看一下", "可不可以帮我看一下", "zh_tone"),
    ("这到底是怎么回事", "这到底是怎么回事", "zh_tone"),
    ("你觉得怎么样", "你觉得怎么样", "zh_tone"),
    ("这个模型会不会太垃圾了呀", "这个模型会不会太垃圾了呀", "zh_tone"),
    ("能不能把它换成更好的呀", "能不能把它换成更好的呀", "zh_tone"),

    # ─── English Numbers (8) ───
    ("thirty eight percent", "38%", "en_num"),
    ("two hundred eighty four", "284", "en_num"),
    ("one thousand twenty four", "1024", "en_num"),
    ("three point one four", "3.14", "en_num"),
    ("five hundred thousand", "500000", "en_num"),
    ("twelve point five percent", "12.5%", "en_num"),
    ("zero point five", "0.5", "en_num"),
    ("twenty three percent growth", "23% growth", "en_num"),

    # ─── English Dates/Math (8) ───
    ("April third twenty twenty six", "April 3rd, 2026", "en_date"),
    ("December thirty first", "December 31st", "en_date"),
    ("three plus five equals eight", "3+5=8", "en_math"),
    ("X is greater than zero", "X>0", "en_math"),
    ("Y is less than one hundred", "Y<100", "en_math"),
    ("A is greater than or equal to B", "A≥B", "en_math"),
    ("three point one four times R squared", "3.14×R²", "en_math"),
    ("X squared plus Y squared equals Z squared", "X²+Y²=Z²", "en_math"),

    # ─── English Fillers/Preserve (8) ───
    ("um so basically the thing is pretty good right", "the thing is pretty good", "en_filler"),
    ("uh I think we should probably go ahead", "I think we should probably go ahead", "en_filler"),
    ("you know like it's kind of important", "it's kind of important", "en_filler"),
    ("can you help me take a look at this", "can you help me take a look at this", "en_keep"),
    ("we need to refactor this code before the deadline", "we need to refactor this code before the deadline", "en_keep"),
    ("what do you think about this approach", "what do you think about this approach", "en_keep"),
    ("I think the performance is around ninety five percent", "I think the performance is around 95%", "en_keep"),
    ("the total cost is twelve dollars and fifty cents", "the total cost is $12.50", "en_keep"),

    # ─── Mixed Code-switching (20) ───
    ("我觉得这个model的performance还不错", "我觉得这个model的performance还不错", "mixed"),
    ("这个feature的implementation有点complex", "这个feature的implementation有点complex", "mixed"),
    ("OK the performance looks quite good表现力还不错", "OK the performance looks quite good，表现力还不错", "mixed"),
    ("it works pretty well但是还有一些edge cases需要handle", "it works pretty well，但是还有一些edge cases需要handle", "mixed"),
    ("我想要keep原本的language不要translate", "我想要keep原本的language不要translate", "mixed"),
    ("just run the benchmark看看accuracy怎么样", "just run the benchmark，看看accuracy怎么样", "mixed"),
    ("百分之三十二的growth rate大于我们的target", "32%的growth rate大于我们的target", "mixed_num"),
    ("然后performance提升了百分之二十三点五", "performance提升了23.5%", "mixed_num"),
    ("这个API的response time大概两百毫秒左右", "这个API的response time大概200毫秒左右", "mixed_num"),
    ("我们需要optimize这个algorithm的efficiency", "我们需要optimize这个algorithm的efficiency", "mixed"),
    ("can you check一下这个code有没有issue", "can you check一下这个code有没有issue", "mixed"),
    ("我今天去了meeting然后discuss了一下project的timeline", "我今天去了meeting，然后discuss了一下project的timeline", "mixed"),
    ("Sometimes my sentences are purely based on English so you never know", "Sometimes my sentences are purely based on English, so you never know", "mixed_en"),
    ("关于这个你确定吗你确定它中文就是占大多数吗", "关于这个你确定吗？你确定它中文就是占大多数吗？", "mixed_zh"),
    ("呃这个use case里面我想要preserve我的中英文mixing", "这个use case里面我想要preserve我的中英文mixing", "mixed_filler"),
    ("嗯然后呢这个bug的root cause是什么我们需要debug一下", "这个bug的root cause是什么，我们需要debug一下", "mixed_filler"),
    ("um I think we should呃我们应该先看看data再decide", "I think we should，我们应该先看看data再decide", "mixed_filler"),
    ("三心二意的人有百分之三十二", "三心二意的人有32%", "mixed_idiom_num"),
    ("不管三七二十一先做了再说呃然后呢再看看", "不管三七二十一先做了再说，再看看", "mixed_idiom_filler"),
    ("七十六percent的growth", "76%的growth", "mixed_num"),

    # ─── Real User Inputs from Logs — complex, long (25) ───
    ("呃就是说如果呢这个百分之三十二的增长率大于我们的target的话嗯我觉得还是可以的",
     "如果32%的增长率大于我们的target的话，我觉得还是可以的", "real"),
    ("千问的这个ASR模型用的是三点四G的为什么我们这个language模型只用九百MB的",
     "千问的这个ASR模型用的是3.4G的，为什么我们这个language模型只用900MB的", "real"),
    ("所以我现在说的话可能会很长很丑一句话可能会非常ugly它将是多语言的",
     "所以我现在说的话可能会很长很丑，一句话可能会非常ugly。它将是多语言的", "real"),
    ("不管三七二十一这个肯定是还可以的表现力应该是可以的",
     "不管三七二十一，这个肯定是还可以的，表现力应该是可以的", "real"),
    ("我们可以直接把模型下载好啊这样子打包成一个package直接给user用",
     "我们可以直接把模型下载好，这样子打包成一个package直接给user用", "real"),
    ("Okay这个规则我觉得很好是一个good starting point",
     "Okay，这个规则我觉得很好，是一个good starting point", "real"),
    ("我们该怎么样选哪一个模型呢有哪些开源的模型它足够好足够快又能解决我们的这个问题",
     "我们该怎么样选哪一个模型呢？有哪些开源的模型，它足够好，足够快，又能解决我们的这个问题？", "real"),
    ("我觉得既然已经有了一个natural language model去帮我们process这些东西我们还需要另外一个hard code的section去把文字转换成数字吗Is it really necessary我不太知道这个到底需不需要",
     "我觉得既然已经有了一个natural language model去帮我们process这些东西，我们还需要另外一个hard code的section去把文字转换成数字吗？Is it really necessary？我不太知道这个到底需不需要", "real"),
    ("我刚刚给你发那个很长的direction的时候我遇到了一个问题就是我还按着recording button的时候它又有了一个error",
     "我刚刚给你发那个很长的direction的时候，我遇到了一个问题，就是我还按着recording button的时候，它又有了一个error", "real"),
    ("76%和3897除以forty seven", "76%和3897÷47", "real"),
    ("678 multiplied by 28", "678×28", "real"),
    ("today is April third 2026", "today is April 3rd, 2026", "real"),
    ("我觉得我们刚刚跑的benchmark还不够好我们把我们的test case给它扩大扩大到两百个吧",
     "我觉得我们刚刚跑的benchmark还不够好，我们把我们的test case给它扩大，扩大到200个吧", "real"),
    ("万一我还想说话只是我卡顿了这算是什么比如说我还在说话但是我在想接下来该怎么说所以我这一会儿没有发声音",
     "万一我还想说话，只是我卡顿了，这算是什么？比如说我还在说话，但是我在想接下来该怎么说，所以我这一会儿没有发声音", "real"),
    ("还有为什么我们要用VAD去自动停止我们的audio recording for double tap他万一我在思考然后已经思考三秒钟了他不就自动停止了",
     "还有为什么我们要用VAD去自动停止我们的audio recording for double tap？他万一我在思考，然后已经思考3秒钟了，他不就自动停止了？", "real"),
    ("关于这个你确定吗你确定它中文就是占大多数吗而英文是更小数吗你完全不知道你在瞎编而且我说的话也不是大多数都是用中文我还会说很多的英文Sometimes my sentences are purely based on English so you never know",
     "关于这个你确定吗？你确定它中文就是占大多数吗？而英文是更小数吗？你完全不知道你在瞎编，而且我说的话也不是大多数都是用中文，我还会说很多的英文。Sometimes my sentences are purely based on English, so you never know", "real"),
    ("Okay we agree on保留英文但是我在想我们现在的这些prompt会不会太复杂了一些especially for第六个preserve idioms and set phrases",
     "Okay we agree on保留英文，但是我在想我们现在的这些prompt会不会太复杂了一些，especially for第6个preserve idioms and set phrases", "real"),
    ("这个模型会不会太垃圾了呀我们换成更好的对我们也有好处",
     "这个模型会不会太垃圾了呀？我们换成更好的，对我们也有好处", "real"),
    ("我现在最主要看中的可能还是他帮我把我混乱的语言整理好但是他不能整理的特别抽象比如说可以帮我去掉一些废话之类的",
     "我现在最主要看中的可能还是他帮我把我混乱的语言整理好，但是他不能整理的特别抽象，比如说可以帮我去掉一些废话之类的", "real"),
    ("我觉得你测试的不是特别好首先我们的LLM的prompt就没有designed去handle这个东西然后呢我们的数字符号也没有让他去handle这个东西而且我们的test case有点太少了",
     "我觉得你测试的不是特别好，首先我们的LLM的prompt就没有designed去handle这个东西，然后我们的数字符号也没有让他去handle这个东西，而且我们的test case有点太少了", "real"),
    ("你能看到我说话的logging吗比如说我的original input是什么然后它又帮我转换成了什么样子我发现它现在会把我原本说的一些文字转换的有点奇怪",
     "你能看到我说话的logging吗？比如说我的original input是什么，然后它又帮我转换成了什么样子？我发现它现在会把我原本说的一些文字转换的有点奇怪", "real"),
    ("你做这个的同时你可以用Agent Teams去做然后你再开一个Sub Agent去看这个debug的问题",
     "你做这个的同时，你可以用Agent Teams去做，然后你再开一个Sub Agent去看这个debug的问题", "real"),
    ("我还是不要wetext了", "我还是不要wetext了", "real"),
    ("好啊开始做吧那我们要给它的input是什么呢", "好啊，开始做吧。那我们要给它的input是什么呢？", "real"),
    ("我觉得我们刚刚跑的benchmark还不够好我们把我们的test case给它扩大然后我们的prompt呢我觉得四种也是太少了",
     "我觉得我们刚刚跑的benchmark还不够好，我们把我们的test case给它扩大。然后我们的prompt呢，我觉得4种也是太少了", "real"),

    # ─── Translation bugs from real usage (should NOT translate) ───
    ("然后submit diff的时候不要忘记rebase到最新的branch上面因为我们现在这个branch点delay了",
     "然后submit diff的时候，不要忘记rebase到最新的branch上面，因为我们现在这个branch点delay了", "real_notranslate"),
    ("然后POC的话别忘记它是可能存在一个或者多个的有可能不存在我们写的方式就借鉴top user和top experiment在这个automation code里面",
     "然后POC的话，别忘记它是可能存在一个或者多个的，有可能不存在，我们写的方式就借鉴top user和top experiment在这个automation code里面", "real_notranslate"),
    ("这个diff就是去改这个SQL的然后给我再加一个column去加这些POC吧",
     "这个diff就是去改这个SQL的，然后给我再加一个column去加这些POC吧", "real_notranslate"),
    ("我们现在不是知道怎么样去找我们的universe的POC了吗有个很准确的方法然后我们有一个pipeline",
     "我们现在不是知道怎么样去找我们的universe的POC了吗？有个很准确的方法，然后我们有一个pipeline", "real_notranslate"),

    # ─── English number conversion in mixed context ───
    ("你刚刚又找出了一些不是active的experiment from the last twenty eight days不是active的universe",
     "你刚刚又找出了一些不是active的experiment from the last 28 days，不是active的universe", "real_notranslate"),

    # ─── Long bilingual real-world (code-switching heavy) ───
    ("虽然两个小孩没有办法真正的做到fifty percent那么一碗水端平but mom has tried my best",
     "虽然两个小孩没有办法真正的做到50%那么一碗水端平，but mom has tried my best", "real_long"),
    ("你真的长大了I'm so happy No we just feel so touched他们也喜欢被尊重的感觉",
     "你真的长大了，I'm so happy. No, we just feel so touched. 他们也喜欢被尊重的感觉", "real_long"),
    ("In my childhood I really hate my parents yelling at me in public It's so awkward and embarrassing",
     "In my childhood, I really hate my parents yelling at me in public. It's so awkward and embarrassing.", "real_long"),
    ("So妈妈说再也不可以这样子来对待我的小孩谢谢你Jacob",
     "So，妈妈说再也不可以这样子来对待我的小孩。谢谢你，Jacob", "real_long"),
    ("Come on babies come here give mommy一个抱一下那我们开心去吃饭OK Good let's go",
     "Come on babies, come here, give mommy一个抱一下。那我们开心去吃饭，OK? Good, let's go.", "real_long"),
    ("嗯你可以再选一个别的东西好吗吃完饭去一起去挑吧",
     "你可以再选一个别的东西，好吗？吃完饭一起去挑吧", "real_long"),
    ("OK你可以一整天逛完了之后我们再make a decision好吗",
     "OK，你可以一整天逛完了之后，我们再make a decision，好吗？", "real_long"),
    ("上次那件事情我在大家面前吼你的事情可不可以原谅妈咪",
     "上次那件事情，我在大家面前吼你的事情，可不可以原谅妈咪？", "real_long"),

    # ─── Edge Cases (10) ───
    ("呃嗯就是说然后呢那个", "", "edge"),
    ("好", "好", "edge"),
    ("OK", "OK", "edge"),
    ("hello你好hi嗨", "hello你好hi嗨", "edge"),
    ("A大于B B大于C所以A大于C", "A>B，B>C，所以A>C", "edge"),
    ("三心二意的人有百分之三十二的概率", "三心二意的人有32%的概率", "edge"),
    ("一些人大概有百分之五十的概率", "一些人大概有50%的概率", "edge"),
    ("X大于Y大于Z大于零", "X>Y>Z>0", "edge"),
    ("I have三百七十六个items and百分之五十are new", "I have 376个items and 50% are new", "edge"),
    ("七十六percent的growth", "76%的growth", "edge"),
]

# ══════════════════════════════════════════════════════════════════
# PROMPT POOL — 40 variations to test
# ══════════════════════════════════════════════════════════════════

def get_prompt_pool():
    """Generate 40 prompt variations systematically varying key dimensions."""
    pool = {}

    # ── Baseline: current winner ──
    pool["baseline_en_detailed"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL RULES:
1. PRESERVE the original language of every word exactly as spoken. If the speaker said an English word, keep it in English. If they said a Chinese word, keep it in Chinese. NEVER translate between languages.
2. Convert spoken numbers to digits (百分之三十二→32%, thirty-eight→38, 三百七十六→376)
3. Convert spoken math/symbols to notation (大于→>, 等于→=, 加→+, 乘以→×, 除以→÷, 大于等于→≥, 小于等于→≤, 不等于→≠, 的平方→², 根号→√)
4. Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢)
5. Add proper punctuation for readability
6. Preserve idioms and set phrases (三心二意, 不管三七二十一)
7. Do NOT rephrase, summarize, or add content

Example: 我今天去了meeting然后discuss了一下project的timeline
Output: 我今天去了meeting，然后discuss了一下project的timeline。"""

    # ── V1: Remove idiom rule (test if model knows natively) ──
    pool["v1_no_idiom_rule"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE the original language of every word. NEVER translate between languages.

Rules:
1. Convert spoken numbers to digits (百分之三十二→32%, thirty-eight→38)
2. Convert spoken math/symbols to notation (大于→>, 等于→=, 加→+, 乘以→×, 除以→÷)
3. Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢)
4. Add proper punctuation
5. Do NOT rephrase, summarize, or add content

Example: 我今天去了meeting然后discuss了一下project的timeline
Output: 我今天去了meeting，然后discuss了一下project的timeline。"""

    # ── V2: Reorder — language preservation last (recency effect) ──
    pool["v2_lang_last"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

Rules:
1. Convert spoken numbers to digits (百分之三十二→32%, thirty-eight→38)
2. Convert spoken math/symbols to notation (大于→>, 等于→=, 加→+, 乘以→×)
3. Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢)
4. Add proper punctuation
5. Preserve idioms (三心二意, 不管三七二十一)
6. Do NOT rephrase or add content
7. CRITICAL: PRESERVE original language of every word. NEVER translate between languages.

Example: 我今天去了meeting然后discuss了一下project的timeline
Output: 我今天去了meeting，然后discuss了一下project的timeline。"""

    # ── V3: Two examples (one ZH-dominant, one EN-dominant) ──
    pool["v3_two_examples"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE the original language of every word. NEVER translate.
Convert numbers to digits, math to symbols, remove fillers, add punctuation.
Preserve idioms. Do NOT rephrase.

Example 1: 呃百分之三十二的growth rate大于我们的target然后呢performance还不错
Output 1: 32%的growth rate大于我们的target，performance还不错

Example 2: um I think thirty eight percent is greater than our target right
Output 2: I think 38% is greater than our target."""

    # ── V4: Three examples ──
    pool["v4_three_examples"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE the original language of every word. NEVER translate.
Convert numbers to digits, math to symbols, remove fillers, add punctuation. Do NOT rephrase.

Example 1: 呃百分之三十二的growth rate大于我们的target
Output 1: 32%的growth rate大于我们的target

Example 2: um thirty eight percent is pretty good right
Output 2: 38% is pretty good.

Example 3: 不管三七二十一我们先做了一些testing看看results
Output 3: 不管三七二十一，我们先做了一些testing，看看results"""

    # ── V5: Ultra-concise + 1 example ──
    pool["v5_ultra_concise"] = """Voice transcription cleaner. Output only cleaned text.
Numbers→digits, math→symbols, remove fillers, add punctuation. Never translate. Never rephrase.

Example: 呃百分之三十二的growth rate → 32%的growth rate"""

    # ── V6: Role as "editor" ──
    pool["v6_editor_role"] = """You are a bilingual text editor. A user dictated the following text using voice input. Clean it up.

Your job: fix punctuation, convert spoken numbers to digits, convert math to symbols, remove verbal hesitations (呃/嗯/um/uh/就是说/然后呢).
Keep EVERY word in its original language. Never translate. Never rephrase. Output only the edited text."""

    # ── V7: Negative examples (what NOT to do) ──
    pool["v7_negative_examples"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

Rules: Convert numbers to digits, math to symbols, remove fillers, add punctuation.
NEVER translate between languages. NEVER rephrase content.

WRONG: Input "这个model的performance不错" → Output "这个模型的表现不错" (translated!)
RIGHT: Input "这个model的performance不错" → Output "这个model的performance不错"

WRONG: Input "三心二意" → Output "3心2意" (broke idiom!)
RIGHT: Input "三心二意" → Output "三心二意" """

    # ── V8: Structured with sections ──
    pool["v8_structured"] = """# Voice Transcription Post-Processor

## Task
Clean up raw voice transcription. Output ONLY the result.

## DO:
- Convert spoken numbers → digits (百分之三十二→32%)
- Convert spoken math → symbols (大于→>, 乘以→×)
- Remove fillers (呃, 嗯, um, uh, 就是说)
- Add punctuation

## DO NOT:
- Translate between languages
- Rephrase or reword
- Change idioms (三心二意, 不管三七二十一)
- Add or remove content"""

    # ── V9: Emphasize bilingual nature ──
    pool["v9_bilingual_emphasis"] = """You process bilingual (Chinese+English) voice transcriptions. The speaker freely mixes both languages in every sentence. This is INTENTIONAL code-switching — not errors.

Output ONLY the cleaned text. Rules:
1. Keep Chinese in Chinese, English in English — NO translation
2. Numbers → digits, math → symbols
3. Remove fillers (呃/嗯/um/uh/就是说/然后呢)
4. Add punctuation. Do not rephrase.

Example: 呃这个model的performance大概百分之九十五
Output: 这个model的performance大概95%"""

    # ── V10: Persona-based ──
    pool["v10_persona"] = """You are a professional transcriptionist who specializes in bilingual Chinese-English speech. Your clients frequently code-switch between languages. You clean up their dictation while respecting their language choices exactly.

Clean the following transcription: convert numbers to digits, math to symbols, remove filler words, add punctuation. Never translate or rephrase. Output only the cleaned text."""

    # ── V11: XML format ──
    pool["v11_xml"] = """<task>Voice transcription post-processing</task>
<output>Cleaned text only, no explanation</output>
<critical>NEVER translate between languages — preserve every word in its original language</critical>
<rules>
- Numbers to digits (百分之三十二→32%, thirty-eight→38)
- Math to symbols (大于→>, 等于→=, 加→+, 乘以→×)
- Remove fillers (呃/嗯/um/uh/就是说/然后呢)
- Add punctuation
- Preserve idioms (三心二意, 不管三七二十一)
- Do NOT rephrase or add content
</rules>"""

    # ── V12: Markdown format ──
    pool["v12_markdown"] = """**Role:** Voice transcription post-processor
**Output:** Cleaned text only

**CRITICAL:** Never translate between languages.

**Rules:**
- `numbers` → digits (百分之三十二→32%)
- `math` → symbols (大于→>, 乘以→×)
- `fillers` → remove (呃, 嗯, um, uh)
- `punctuation` → add
- `idioms` → preserve (三心二意)
- `content` → do NOT rephrase"""

    # ── V13: Minimal + bilingual example ──
    pool["v13_minimal_bi_example"] = """Clean up this voice transcription. Numbers→digits, remove fillers, add punctuation. NEVER translate.

Example: 呃百分之三十二的growth rate然后呢performance还不错
Output: 32%的growth rate，performance还不错"""

    # ── V14: Instruction-style (imperative) ──
    pool["v14_imperative"] = """Process the following voice transcription:
- Convert all spoken numbers to Arabic digits
- Convert mathematical expressions to symbols
- Remove all filler words and hesitations
- Add appropriate punctuation
- KEEP every word in its original language
- DO NOT translate, rephrase, or add anything
Output ONLY the processed text."""

    # ── V15: Q&A style ──
    pool["v15_qa_style"] = """Q: How should I clean up a bilingual voice transcription?
A: Convert numbers to digits, math to symbols, remove fillers (呃/嗯/um/uh), add punctuation. Never translate between languages. Never rephrase. Preserve idioms.

Now clean up the following transcription. Output only the result:"""

    # ── V16: Chain-of-thought suppressed ──
    pool["v16_no_thinking"] = """You are a voice transcription formatter. You must respond with ONLY the formatted text — no thinking, no explanation, no preamble.

Format rules: spoken numbers→digits, spoken math→symbols, remove fillers, add punctuation.
Language rule: NEVER translate. Keep English as English, Chinese as Chinese.
Content rule: NEVER rephrase, add, or remove meaningful content."""

    # ── V17: Priority-ordered ──
    pool["v17_priority_order"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

Priority 1 (MOST IMPORTANT): Keep every word in its original language. Never translate.
Priority 2: Convert spoken numbers to digits and math to symbols.
Priority 3: Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢).
Priority 4: Add punctuation for readability.
Priority 5 (LEAST IMPORTANT): Light grammar fixes if needed."""

    # ── V18: With constraints ──
    pool["v18_with_constraints"] = """You are a voice transcription post-processor.

INPUT: Raw voice transcription (may be Chinese, English, or mixed)
OUTPUT: Cleaned text only (no explanation)

CONSTRAINTS:
- Language: preserve original (never translate)
- Numbers: spoken → digits
- Math: spoken → symbols (>, <, =, +, -, ×, ÷, ≥, ≤, ², √)
- Fillers: remove (呃, 嗯, um, uh, 就是说, 然后呢)
- Punctuation: add where needed
- Content: do not modify meaning"""

    # ── V19: Baseline but no examples ──
    pool["v19_detailed_no_example"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL RULES:
1. PRESERVE the original language of every word exactly as spoken. NEVER translate between languages.
2. Convert spoken numbers to digits (百分之三十二→32%, thirty-eight→38, 三百七十六→376)
3. Convert spoken math/symbols to notation (大于→>, 等于→=, 加→+, 乘以→×, 除以→÷, 大于等于→≥, 小于等于→≤, 不等于→≠, 的平方→², 根号→√)
4. Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢)
5. Add proper punctuation for readability
6. Preserve idioms and set phrases (三心二意, 不管三七二十一)
7. Do NOT rephrase, summarize, or add content"""

    # ── V20-V25: Baseline variations with different example styles ──
    pool["v20_example_math"] = pool["baseline_en_detailed"].replace(
        "Example: 我今天去了meeting然后discuss了一下project的timeline\nOutput: 我今天去了meeting，然后discuss了一下project的timeline。",
        "Example: 呃X大于零并且Y小于百分之三十二然后呢我们看看\nOutput: X>0并且Y<32%，我们看看")

    pool["v21_example_filler_heavy"] = pool["baseline_en_detailed"].replace(
        "Example: 我今天去了meeting然后discuss了一下project的timeline\nOutput: 我今天去了meeting，然后discuss了一下project的timeline。",
        "Example: 呃就是说嗯这个百分之三十二的增长rate还不错然后呢嗯performance也可以\nOutput: 32%的增长rate还不错，performance也可以")

    pool["v22_example_pure_en"] = pool["baseline_en_detailed"].replace(
        "Example: 我今天去了meeting然后discuss了一下project的timeline\nOutput: 我今天去了meeting，然后discuss了一下project的timeline。",
        "Example: um thirty eight percent is greater than twenty five percent right\nOutput: 38% is greater than 25%.")

    pool["v23_example_idiom"] = pool["baseline_en_detailed"].replace(
        "Example: 我今天去了meeting然后discuss了一下project的timeline\nOutput: 我今天去了meeting，然后discuss了一下project的timeline。",
        "Example: 不管三七二十一我们先把百分之五十的task做完\nOutput: 不管三七二十一，我们先把50%的task做完")

    pool["v24_example_long"] = pool["baseline_en_detailed"].replace(
        "Example: 我今天去了meeting然后discuss了一下project的timeline\nOutput: 我今天去了meeting，然后discuss了一下project的timeline。",
        "Example: 呃就是说如果百分之三十二的growth rate大于我们的target的话嗯我觉得这个model的performance还是可以的\nOutput: 如果32%的growth rate大于我们的target的话，我觉得这个model的performance还是可以的")

    pool["v25_two_mixed_examples"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

CRITICAL: PRESERVE original language. NEVER translate. Convert numbers/math to digits/symbols. Remove fillers. Add punctuation. Preserve idioms. Do NOT rephrase.

Example 1: 呃百分之三十二的growth rate大于我们的target然后呢performance还不错
Output 1: 32%的growth rate大于我们的target，performance还不错

Example 2: 不管三七二十一我们先做了一些testing嗯看看这个model的accuracy怎么样
Output 2: 不管三七二十一，我们先做了一些testing，看看这个model的accuracy怎么样"""

    # ── V26-V30: Chinese prompt variations ──
    pool["v26_zh_with_notranslate"] = """你是语音转文字的后处理工具。只输出处理后的文本。

关键规则：
1. 保留每个词的原始语言——英文词保持英文，中文词保持中文，绝对不要翻译
2. 口语数字转阿拉伯数字（百分之三十二→32%，thirty-eight→38）
3. 数学符号转符号（大于→>，等于→=，加→+，乘以→×）
4. 去掉语气词（呃、嗯、um、uh、就是说、然后呢）
5. 加标点
6. 不要改写内容

示例：呃百分之三十二的growth rate然后呢performance还不错
输出：32%的growth rate，performance还不错"""

    pool["v27_zh_minimal_notranslate"] = """你是语音转文字的后处理工具。只输出处理后的文本。
数字转阿拉伯数字，符号转符号，去废话，加标点。绝对不要翻译——英文保持英文。"""

    pool["v28_zh_bilingual_examples"] = """你是语音转文字的后处理工具。说话者经常中英文混合，这是有意的——请保留。

规则：数字转数字，符号转符号，去废话，加标点。不要翻译，不要改写。

示例1：呃这个model的performance大概百分之九十五
输出1：这个model的performance大概95%

示例2：um thirty eight percent is pretty good
输出2：38% is pretty good."""

    # ── V29-V30: Mixed language prompts ──
    pool["v29_mixed_prompt"] = """Voice transcription post-processor. Output ONLY cleaned text.
说话者中英文混合，preserve every word in its original language.

Rules/规则:
- Numbers/数字 → digits (百分之三十二→32%, thirty-eight→38)
- Math/符号 → symbols (大于→>, 加→+, 乘以→×)
- Fillers/废话 → remove (呃, 嗯, um, uh)
- Punctuation/标点 → add
- Never translate/不要翻译
- Never rephrase/不要改写"""

    pool["v30_mixed_with_example"] = pool["v29_mixed_prompt"] + """

Example: 呃百分之三十二的growth rate大于target然后呢还不错
Output: 32%的growth rate大于target，还不错"""

    # ── V31-V35: Experimental formats ──
    pool["v31_json_compact"] = """{"task":"clean voice transcription","output":"text only","rules":["never translate","numbers→digits","math→symbols","remove fillers(呃/嗯/um/uh)","add punctuation","preserve idioms","never rephrase"]}"""

    pool["v32_bullet_only"] = """• Clean voice transcription
• Output only cleaned text
• NEVER translate between languages
• Numbers → digits
• Math → symbols
• Remove fillers
• Add punctuation
• Keep idioms
• Don't rephrase"""

    pool["v33_system_card"] = """SYSTEM: VoiceInk Post-Processor v2
MODE: Bilingual transcription cleanup
LANGUAGE POLICY: Strict preservation (no translation)
ACTIONS: [number→digit] [math→symbol] [filler→remove] [punct→add]
PROHIBITIONS: [translate] [rephrase] [add content]
OUTPUT: Cleaned text only"""

    pool["v34_few_rules_many_examples"] = """Clean up voice transcription. Never translate. Output only result.

呃百分之三十二 → 32%
三心二意 → 三心二意
um thirty eight percent → 38%
X大于零 → X>0
这个model的performance不错 → 这个model的performance不错
六乘以七等于四十二 → 6×7=42
嗯就是说还可以 → 还可以
不管三七二十一先做 → 不管三七二十一，先做"""

    pool["v35_examples_only_bilingual"] = """Fix voice transcription. Output only the result.

呃百分之三十二的增长 → 32%的增长
三心二意的态度 → 三心二意的态度
um thirty eight percent growth → 38% growth
X大于零并且Y小于一百 → X>0并且Y<100
呃这个model的performance不错 → 这个model的performance不错
六乘以七等于四十二 → 6×7=42
嗯就是说还可以 → 还可以
不管三七二十一先做了一些testing → 不管三七二十一，先做了一些testing
I think performance is around ninety five percent → I think performance is around 95%
一些东西需要整理一下子 → 一些东西需要整理一下子"""

    # ── V36-V40: Top performer hybrids ──
    pool["v36_baseline_plus_bilingual"] = """You are a voice transcription post-processor for bilingual speech. Output ONLY the cleaned text.

The speaker freely mixes Chinese and English. This is intentional code-switching.

CRITICAL RULES:
1. PRESERVE the original language of every word. NEVER translate between languages.
2. Convert spoken numbers to digits (百分之三十二→32%, thirty-eight→38)
3. Convert spoken math/symbols to notation (大于→>, 等于→=, 加→+, 乘以→×, 除以→÷)
4. Remove filler words (呃, 嗯, um, uh, 就是说, 然后呢)
5. Add proper punctuation for readability
6. Preserve idioms (三心二意, 不管三七二十一)
7. Do NOT rephrase, summarize, or add content

Example: 呃百分之三十二的growth rate大于我们的target然后呢performance还不错
Output: 32%的growth rate大于我们的target，performance还不错"""

    pool["v37_xml_with_example"] = """<task>Bilingual voice transcription cleanup</task>
<output>Cleaned text only</output>
<critical>NEVER translate — keep Chinese as Chinese, English as English</critical>
<rules>
- Numbers → digits (百分之三十二→32%, thirty-eight→38)
- Math → symbols (大于→>, 加→+, 乘以→×)
- Remove fillers (呃/嗯/um/uh/就是说/然后呢)
- Add punctuation
- Preserve idioms
- Never rephrase
</rules>
<example>
Input: 呃百分之三十二的growth rate大于target然后呢还不错
Output: 32%的growth rate大于target，还不错
</example>"""

    pool["v38_priority_with_example"] = """You are a voice transcription post-processor. Output ONLY the cleaned text.

Priority 1 (CRITICAL): Keep every word in its original language. Never translate.
Priority 2: Convert numbers to digits and math to symbols.
Priority 3: Remove fillers (呃/嗯/um/uh/就是说/然后呢).
Priority 4: Add punctuation.
Priority 5: Preserve idioms (三心二意, 不管三七二十一). Never rephrase.

Example: 呃百分之三十二的growth rate大于target然后呢还不错
Output: 32%的growth rate大于target，还不错"""

    pool["v39_concise_structured"] = """ROLE: Voice transcription cleaner
OUTPUT: Cleaned text only

MUST DO:
- Numbers → digits (百分之三十二→32%)
- Math → symbols (大于→>, 乘以→×)
- Remove fillers (呃/嗯/um/uh)
- Add punctuation

MUST NOT:
- Translate (keep Chinese/English as-is)
- Rephrase or add content
- Change idioms (三心二意→keep)

EXAMPLE: 呃百分之三十二的growth rate → 32%的growth rate"""

    pool["v40_best_hybrid"] = """You are a bilingual voice transcription post-processor. Output ONLY the cleaned text.

The speaker mixes Chinese and English freely — preserve both exactly.

Rules:
1. NEVER translate between languages
2. Spoken numbers → digits (百分之三十二→32%, thirty-eight→38)
3. Spoken math → symbols (大于→>, 等于→=, 加→+, 乘以→×, 除以→÷, ≥, ≤, ≠, ², √)
4. Remove fillers (呃, 嗯, um, uh, 就是说, 然后呢)
5. Add punctuation
6. Preserve idioms (三心二意, 不管三七二十一). Never rephrase.

Example 1: 呃百分之三十二的growth rate大于我们的target
Output 1: 32%的growth rate大于我们的target

Example 2: um thirty eight percent is pretty good不管三七二十一
Output 2: 38% is pretty good，不管三七二十一"""

    return pool


# ══════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════

def score(result, expected, raw_input, category):
    """Score 0.0-1.0 across multiple dimensions."""
    if not expected and not result:
        return 1.0
    if not expected:
        return 0.5 if len(result.strip()) < 5 else 0.0
    if not result:
        return 0.0

    r = re.sub(r'\s+', '', result.strip().rstrip('。.'))
    e = re.sub(r'\s+', '', expected.strip().rstrip('。.'))
    if r == e:
        return 1.0

    s = 0.0

    # Dimension 1: Number conversion (30%)
    exp_digits = set(re.findall(r'\d+\.?\d*%?', expected))
    res_digits = set(re.findall(r'\d+\.?\d*%?', result))
    if exp_digits:
        s += 0.3 * len(exp_digits & res_digits) / len(exp_digits)
    else:
        s += 0.3  # no numbers needed

    # Dimension 2: Symbol conversion (15%)
    exp_syms = set(re.findall(r'[+\-×÷=><≥≤≠√²¹⁰≈]', expected))
    res_syms = set(re.findall(r'[+\-×÷=><≥≤≠√²¹⁰≈]', result))
    if exp_syms:
        s += 0.15 * len(exp_syms & res_syms) / len(exp_syms)
    else:
        s += 0.15

    # Dimension 3: Language preservation (35%)
    en_in = set(w.lower() for w in re.findall(r'[a-zA-Z]{2,}', raw_input))
    en_out = set(w.lower() for w in re.findall(r'[a-zA-Z]{2,}', result))
    zh_in = re.findall(r'[\u4e00-\u9fff]+', raw_input)
    zh_out = re.findall(r'[\u4e00-\u9fff]+', result)
    en_score = len(en_in & en_out) / max(len(en_in), 1) if en_in else 1.0
    # For Chinese, check key content chars are preserved
    zh_in_chars = set(''.join(zh_in))
    zh_out_chars = set(''.join(zh_out))
    zh_score = len(zh_in_chars & zh_out_chars) / max(len(zh_in_chars), 1) if zh_in_chars else 1.0
    s += 0.35 * (0.5 * en_score + 0.5 * zh_score)

    # Dimension 4: Filler removal (10%)
    fillers = ['呃', '嗯', '就是说', '然后呢', '那个']
    filler_in = sum(1 for f in fillers if f in raw_input)
    filler_out = sum(1 for f in fillers if f in result)
    if filler_in > 0:
        s += 0.1 * (1 - filler_out / filler_in)
    else:
        s += 0.1

    # Dimension 5: General similarity (10%)
    common = len(set(result) & set(expected))
    total = len(set(result) | set(expected))
    s += 0.1 * common / max(total, 1)

    return min(s, 1.0)


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"rounds": [], "leaderboard": {}}


def save_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print("=" * 80)
    print("VoiceInk Overnight Prompt Optimizer")
    print("=" * 80)
    print(f"Tests: {len(TESTS)}")
    print(f"Loading model...")

    model, tokenizer = load('Qwen/Qwen3-8B-MLX-4bit')
    print("Model loaded.\n")

    data = load_results()
    already_tested = {r["config_id"] for r in data["rounds"]}

    def run_llm(text, system_prompt, samp):
        if not text.strip():
            return ""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text + '\n/no_think'},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        raw = generate(model, tokenizer, prompt=prompt,
                       max_tokens=max(len(text) * 3, 200),
                       sampler=samp, verbose=False)
        return re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()

    # Build test queue
    prompts = get_prompt_pool()
    temps = [0.3]  # Phase 1: fixed temp
    queue = []

    # Phase 1: all prompts at temp=0.3
    for p_name in prompts:
        config_id = f"{p_name}_t0.3"
        if config_id not in already_tested:
            queue.append((p_name, 0.3, config_id))

    # Phase 2: will be added after Phase 1 completes (top 5 × temps)
    phase2_temps = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"Phase 1: {len(queue)} prompt variations to test")
    print(f"Phase 2: Top 5 × {len(phase2_temps)} temperatures (added after Phase 1)")
    print(f"Already completed: {len(already_tested)} rounds")
    print()

    round_num = len(data["rounds"])
    phase1_done = False

    while queue:
        p_name, temp, config_id = queue.pop(0)
        round_num += 1

        sampler = make_sampler(temp=temp, top_p=0.8, top_k=20)
        prompt_text = prompts.get(p_name, prompts["baseline_en_detailed"])

        print(f"Round {round_num}: {config_id} ...", end=" ", flush=True)
        t_start = time.monotonic()

        scores = []
        cat_scores = {}
        for raw_input, expected, category in TESTS:
            output = run_llm(raw_input, prompt_text, sampler)
            s = score(output, expected, raw_input, category)
            scores.append(s)
            cat_scores.setdefault(category, []).append(s)

        elapsed = time.monotonic() - t_start
        avg = sum(scores) / len(scores)
        exact = sum(1 for s in scores if s >= 0.95) / len(scores)

        # Save round
        round_data = {
            "config_id": config_id,
            "prompt_name": p_name,
            "temperature": temp,
            "avg_score": round(avg, 4),
            "exact_match": round(exact, 4),
            "avg_time": round(elapsed / len(TESTS), 3),
            "total_time": round(elapsed, 1),
            "category_scores": {c: round(sum(s)/len(s), 4) for c, s in cat_scores.items()},
        }
        data["rounds"].append(round_data)
        data["leaderboard"][config_id] = round(avg, 4)
        save_results(data)

        # Print
        print(f"→ {avg:.1%} (exact {exact:.0%}) in {elapsed:.0f}s")

        # Check if Phase 1 is done, add Phase 2
        if not phase1_done and not any(cid.endswith("_t0.3") for _, _, cid in queue):
            phase1_done = True
            # Find top 5 prompts
            phase1_results = [(r["prompt_name"], r["avg_score"])
                             for r in data["rounds"] if r["temperature"] == 0.3]
            phase1_results.sort(key=lambda x: x[1], reverse=True)
            top5 = [name for name, _ in phase1_results[:5]]

            print(f"\n{'='*60}")
            print(f"Phase 1 complete! Top 5 prompts:")
            for i, (name, sc) in enumerate(phase1_results[:5]):
                print(f"  #{i+1} {name}: {sc:.1%}")
            print(f"\nStarting Phase 2: testing {len(phase2_temps)} temperatures...")
            print(f"{'='*60}\n")

            for p_name in top5:
                for t in phase2_temps:
                    cid = f"{p_name}_t{t}"
                    if cid not in already_tested and cid not in {q[2] for q in queue}:
                        queue.append((p_name, t, cid))
                        prompts[p_name] = prompts.get(p_name, prompts["baseline_en_detailed"])

    # Final leaderboard
    print(f"\n{'='*80}")
    print("FINAL LEADERBOARD")
    print(f"{'='*80}")
    sorted_lb = sorted(data["leaderboard"].items(), key=lambda x: x[1], reverse=True)
    for i, (config, sc) in enumerate(sorted_lb[:20]):
        marker = " ⭐" if i == 0 else ""
        print(f"  #{i+1:>2} {config:<45} {sc:.1%}{marker}")

    print(f"\nTotal rounds: {len(data['rounds'])}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
